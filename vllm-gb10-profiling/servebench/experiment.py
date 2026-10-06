"""Run one experiment end-to-end: launch -> health -> load sweeps -> agent -> profile -> teardown.

An experiment is a YAML file. ``extends:`` pulls in a base file and the two
are deep-merged, so each experiment states only the one or two knobs it
changes relative to the baseline. That keeps before/after comparisons honest.
"""

from __future__ import annotations

import asyncio
import copy
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from servebench import __version__, launcher, profiling
from servebench.agent import AgentConfig, dump_jsonl, run_agent
from servebench.loadgen import run_level
from servebench.workloads import WorkloadSpec


def _log(*args: Any) -> None:
    print(*args, flush=True)


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_experiment(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(load_experiment(path.parent / parent), data)
    data.setdefault("name", path.stem)
    for key in ("model", "served_model_name"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")
    return data


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


async def collect_env(base_url: str, handle: launcher.ServerHandle) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "servebench_version": __version__,
        "launcher": handle.launcher,
        "server_cmd": handle.cmd,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "nvidia_smi": _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"]
        ),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
    }
    async with httpx.AsyncClient() as c:
        for ep in ("version", "v1/models"):
            try:
                r = await c.get(f"{base_url}/{ep}", timeout=10)
                info[ep.replace("/", "_")] = r.json()
            except (httpx.HTTPError, ValueError):
                info[ep.replace("/", "_")] = None
    info["simulated"] = handle.launcher == "mock" or "simulated" in json.dumps(info.get("version") or {})
    return info


async def _wait_ready(base_url: str, handle: launcher.ServerHandle, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            if not handle.alive():
                raise RuntimeError("server process exited during startup; see server.log")
            try:
                if (await c.get(f"{base_url}/health", timeout=5)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _scale(n: int, quick: bool, floor: int = 2) -> int:
    return max(floor, n // 8) if quick else n


async def run_experiment_async(
    exp: dict[str, Any],
    out_dir: Path,
    *,
    launcher_name: str,
    port: int = 8000,
    base_url: str | None = None,
    profile: bool = True,
    quick: bool = False,
    log=_log,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.resolved.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    base_url = (base_url or f"http://127.0.0.1:{port}").rstrip("/")
    model = exp["served_model_name"]
    prof_cfg = exp.get("profile", {})
    do_profile = profile and prof_cfg.get("enabled", True)
    profile_dir = (out_dir / "profile" / "traces") if do_profile and launcher_name != "external" else None
    slo = exp.get("slo", {})

    handle = launcher.launch(exp, launcher_name, port, out_dir / "server.log", profile_dir)
    summary: dict[str, Any] = {
        "name": exp["name"],
        "description": exp.get("description", ""),
        "changes": exp.get("changes", []),
        "loadtest": [],
        "agent": [],
        "profile": None,
    }
    try:
        log(f"[{exp['name']}] waiting for server ({launcher_name}) ...")
        t_start = time.perf_counter()
        await _wait_ready(base_url, handle, exp.get("startup_timeout_s", 1800))
        summary["startup_s"] = round(time.perf_counter() - t_start, 1)
        env = await collect_env(base_url, handle)
        _write_json(out_dir / "env.json", env)
        summary["simulated"] = env["simulated"]

        for wl in exp.get("workloads", []):
            spec = WorkloadSpec.from_dict(wl)
            sweep = wl.get("sweep", {"mode": "concurrency", "levels": [1]})
            for level in sweep["levels"]:
                n = _scale(int(sweep.get("requests_per_level", 32)), quick)
                n = max(n, int(level) if sweep.get("mode", "concurrency") == "concurrency" else n)
                log(f"[{exp['name']}] {spec.name}: {sweep.get('mode', 'concurrency')}={level} n={n}")
                lr = await run_level(
                    base_url,
                    model,
                    spec,
                    mode=sweep.get("mode", "concurrency"),
                    level=level,
                    num_requests=n,
                    warmup_requests=sweep.get("warmup_requests", 2),
                    ignore_eos=wl.get("ignore_eos", True),
                    slo_ttft_ms=slo.get("ttft_ms"),
                    slo_tpot_ms=slo.get("tpot_ms"),
                    max_concurrency=sweep.get("max_concurrency"),
                )
                wl_dir = out_dir / "loadtest" / spec.name
                wl_dir.mkdir(parents=True, exist_ok=True)
                tag = f"{lr.mode}_{level:g}"
                dump_jsonl(str(wl_dir / f"{tag}.requests.jsonl"), [r.to_record() for r in lr.results])
                dump_jsonl(str(wl_dir / f"{tag}.timeseries.jsonl"), lr.timeseries)
                summary["loadtest"].append(lr.summary)
                s = lr.summary
                log(
                    f"    ok={s['num_ok']}/{s['num_requests']} out_tok/s={s['output_tok_per_s']} "
                    f"ttft_p50={s['ttft_ms']['p50']}ms tpot_p50={s['tpot_ms']['p50']}ms "
                    f"kv_peak={s['server'].get('peak_kv_cache_usage')}"
                )

        for i, ag in enumerate(exp.get("agent", [])):
            cfg = AgentConfig.from_dict(ag)
            if quick:
                cfg.episodes_per_session = min(cfg.episodes_per_session, 2)
            log(f"[{exp['name']}] agent[{i}] mode={cfg.mode} sessions={cfg.sessions}")
            res = await run_agent(base_url, model, cfg)
            adir = out_dir / "agent" / f"{cfg.mode}_s{cfg.sessions}"
            adir.mkdir(parents=True, exist_ok=True)
            dump_jsonl(str(adir / "steps.jsonl"), res["steps"])
            dump_jsonl(str(adir / "episodes.jsonl"), res["episodes"])
            entry = {"mode": cfg.mode, "sessions": cfg.sessions, **res["summary"], "server": res["server"]}
            summary["agent"].append(entry)
            log(
                f"    task_p50={entry['task_latency_p50_ms']}ms success={entry['task_success_rate']} "
                f"prefix_hit={entry['server'].get('prefix_cache_hit_rate')}"
            )

        if profile_dir is not None:
            summary["profile"] = await _profile(base_url, model, exp, profile_dir, out_dir / "profile", log)
    finally:
        log(f"[{exp['name']}] stopping server")
        handle.stop()
    _write_json(out_dir / "summary.json", summary)
    return summary


async def _profile(
    base_url: str, model: str, exp: dict[str, Any], trace_dir: Path, out: Path, log=_log
) -> dict[str, Any] | None:
    pc = exp.get("profile", {})
    wl_name = pc.get("workload")
    wl = next((w for w in exp.get("workloads", []) if w["name"] == wl_name), None) or exp["workloads"][0]
    spec = WorkloadSpec.from_dict({**wl, "seed": wl.get("seed", 1234) + 777})
    log(f"[{exp['name']}] profiling {spec.name} at concurrency {pc.get('concurrency', 8)}")
    await profiling.start_profile(base_url)
    try:
        await run_level(
            base_url,
            model,
            spec,
            mode="concurrency",
            level=pc.get("concurrency", 8),
            num_requests=pc.get("requests", 16),
            warmup_requests=0,
        )
    finally:
        await profiling.stop_profile(base_url)
    # Workers flush traces asynchronously after stop_profile returns.
    traces: list[Path] = []
    for _ in range(pc.get("flush_wait_s", 120)):
        traces = profiling.find_traces(trace_dir)
        if traces:
            break
        await asyncio.sleep(1)
    await asyncio.sleep(2)
    traces = profiling.find_traces(trace_dir)
    if not traces:
        log("    no traces found (is --profiler-config honoured by this vLLM build?)")
        return None
    sums = [profiling.summarize_trace(t) for t in traces]
    (out / "kernel_summary.md").write_text(profiling.render_markdown(sums))
    _write_json(out / "kernel_summary.json", sums)
    log(f"    {len(traces)} trace(s); open in https://ui.perfetto.dev")
    return {
        "traces": [str(t.relative_to(out.parent)) for t in traces],
        "summaries": [{k: v for k, v in s.items() if k != "top_kernels"} for s in sums],
    }


def run_experiment(exp: dict[str, Any], out_dir: Path, **kw: Any) -> dict[str, Any]:
    return asyncio.run(run_experiment_async(exp, out_dir, **kw))
