"""Command line entry point: ``python -m servebench <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from servebench import __version__


def _cmd_run(a: argparse.Namespace) -> int:
    from servebench.experiment import load_experiment, run_experiment

    out_root = Path(a.out)
    failed = []
    for cfg in a.configs:
        exp = load_experiment(cfg)
        try:
            run_experiment(
                exp,
                out_root / exp["name"],
                launcher_name=a.launcher,
                port=a.port,
                base_url=a.base_url,
                profile=not a.no_profile,
                quick=a.quick,
            )
        except Exception as e:  # keep going: one bad config shouldn't lose a night of runs
            print(f"[{exp['name']}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(exp["name"])
            if a.fail_fast:
                break
    if not a.no_report:
        from servebench.report import build_report

        try:
            print(f"report: {build_report(out_root, baseline=a.baseline)}")
        except FileNotFoundError as e:
            print(f"no report: {e}", file=sys.stderr)
    if failed:
        print(f"failed experiments: {failed}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_loadtest(a: argparse.Namespace) -> int:
    from servebench.loadgen import run_level
    from servebench.workloads import WorkloadSpec

    spec = WorkloadSpec(
        name=a.kind,
        kind=a.kind,
        input_tokens=a.input_tokens,
        output_tokens=a.output_tokens,
        shared_prefix_tokens=a.shared_prefix_tokens,
        num_prefixes=a.num_prefixes,
    )
    rows = []
    for lvl in [float(x) for x in a.levels.split(",")]:
        lr = asyncio.run(
            run_level(
                a.base_url,
                a.model,
                spec,
                mode=a.mode,
                level=lvl,
                num_requests=a.num_requests,
                slo_ttft_ms=a.slo_ttft_ms,
                slo_tpot_ms=a.slo_tpot_ms,
            )
        )
        s = lr.summary
        rows.append(s)
        print(
            f"{a.mode}={lvl:g}: ok={s['num_ok']}/{s['num_requests']} out_tok/s={s['output_tok_per_s']} "
            f"TTFT p50/p99={s['ttft_ms']['p50']}/{s['ttft_ms']['p99']}ms TPOT p50={s['tpot_ms']['p50']}ms "
            f"goodput={s['goodput_rps']} req/s"
        )
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=2))
    return 0


def _cmd_agent(a: argparse.Namespace) -> int:
    from servebench.agent import AgentConfig, run_agent

    cfg = AgentConfig(
        mode=a.mode,
        sessions=a.sessions,
        episodes_per_session=a.episodes,
        system_prompt_tokens=a.system_prompt_tokens,
        replay_steps=a.steps,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {},
    )
    res = asyncio.run(run_agent(a.base_url, a.model, cfg))
    print(json.dumps({"summary": res["summary"], "server": res["server"]}, indent=2))
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from servebench.report import build_report

    print(build_report(Path(a.results_dir), baseline=a.baseline, headline_level=a.headline_level))
    return 0


def _cmd_kv_math(a: argparse.Namespace) -> int:
    from servebench.mock.engine import BLOCK, SimConfig

    cfg = SimConfig(
        model=a.model,
        gpu_memory_utilization=a.gpu_memory_utilization,
        kv_cache_dtype=a.kv_cache_dtype,
        quantization=a.quantization,
        total_mem_gb=a.total_mem_gb,
        activation_reserve_gb=a.activation_reserve_gb,
    )
    tokens = cfg.num_blocks * BLOCK
    per_tok = cfg.kv_bytes_per_token
    print(f"model                 : {a.model}")
    print(f"weights               : {cfg.weight_bytes / 1e9:.1f} GB")
    print(
        f"KV bytes / token      : {per_tok:,.0f} B ({per_tok / 1024:.0f} KiB)  "
        f"= 2 x layers x kv_heads x head_dim x dtype_bytes"
    )
    print(f"KV budget             : {tokens * per_tok / 1e9:.1f} GB -> {tokens:,} tokens")
    for ctx in (2048, 8192, 32768, 131072):
        print(f"  full-length seqs @ {ctx:>6}: {tokens // ctx:>5}   (KV per seq {ctx * per_tok / 2**30:.2f} GiB)")
    print(
        f"decode roofline (bs=1): {cfg.mem_bw_gbs * 1e9 / cfg.weight_bytes:.1f} tok/s upper bound "
        f"at {cfg.mem_bw_gbs:.0f} GB/s (weights read once per token)"
    )
    return 0


def _cmd_trace(a: argparse.Namespace) -> int:
    from servebench.profiling import render_markdown, summarize_trace

    print(render_markdown([summarize_trace(Path(p)) for p in a.traces]))
    return 0


def _cmd_links(a: argparse.Namespace) -> int:
    """Print shareable Perfetto links for traces committed to GitHub."""
    import subprocess
    from urllib.parse import quote

    from servebench.profiling import find_traces

    root = Path(a.results_dir).resolve()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=root
    ).stdout.strip()
    for t in find_traces(root):
        rel = t.relative_to(top).as_posix()
        raw = f"https://raw.githubusercontent.com/{a.repo}/{a.ref}/{rel}"
        perfetto = f"https://ui.perfetto.dev/#!/?url={quote(raw, safe='')}"
        print(f"{t.relative_to(root)}\n  raw:      {raw}\n  perfetto: {perfetto}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="servebench", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run experiment YAML(s) end-to-end and build the report")
    r.add_argument("configs", nargs="+")
    r.add_argument("--launcher", choices=["docker", "local", "mock", "external"], default="docker")
    r.add_argument("--out", required=True, help="results root, e.g. results/gb10")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--base-url", default=None, help="for --launcher external")
    r.add_argument("--baseline", default=None)
    r.add_argument("--no-profile", action="store_true")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--quick", action="store_true", help="1/8 of the requests (smoke test)")
    r.add_argument("--fail-fast", action="store_true")
    r.set_defaults(fn=_cmd_run)

    lt = sub.add_parser("loadtest", help="ad-hoc load test against a running server")
    lt.add_argument("--base-url", default="http://127.0.0.1:8000")
    lt.add_argument("--model", required=True)
    lt.add_argument("--kind", choices=["random", "shared_prefix", "long_context"], default="random")
    lt.add_argument("--mode", choices=["concurrency", "rate"], default="concurrency")
    lt.add_argument("--levels", default="1,4,16")
    lt.add_argument("--num-requests", type=int, default=32)
    lt.add_argument("--input-tokens", type=int, default=512)
    lt.add_argument("--output-tokens", type=int, default=128)
    lt.add_argument("--shared-prefix-tokens", type=int, default=2048)
    lt.add_argument("--num-prefixes", type=int, default=1)
    lt.add_argument("--slo-ttft-ms", type=float, default=None)
    lt.add_argument("--slo-tpot-ms", type=float, default=None)
    lt.add_argument("--json-out")
    lt.set_defaults(fn=_cmd_loadtest)

    ag = sub.add_parser("agent", help="ad-hoc agent-loop latency test")
    ag.add_argument("--base-url", default="http://127.0.0.1:8000")
    ag.add_argument("--model", required=True)
    ag.add_argument("--mode", choices=["replay", "react"], default="replay")
    ag.add_argument("--sessions", type=int, default=1)
    ag.add_argument("--episodes", type=int, default=5)
    ag.add_argument("--steps", type=int, default=6)
    ag.add_argument("--system-prompt-tokens", type=int, default=2000)
    ag.add_argument("--no-thinking", action="store_true", help="Qwen3: disable <think> via chat_template_kwargs")
    ag.set_defaults(fn=_cmd_agent)

    rp = sub.add_parser("report", help="build REPORT.md + plots from a results dir")
    rp.add_argument("results_dir")
    rp.add_argument("--baseline")
    rp.add_argument("--headline-level", type=float)
    rp.set_defaults(fn=_cmd_report)

    kv = sub.add_parser("kv-math", help="KV-cache capacity arithmetic for a model on GB10")
    kv.add_argument("--model", default="Qwen/Qwen3-8B")
    kv.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    kv.add_argument("--kv-cache-dtype", default="auto")
    kv.add_argument("--quantization", default=None)
    kv.add_argument("--total-mem-gb", type=float, default=128.0)
    kv.add_argument("--activation-reserve-gb", type=float, default=6.0)
    kv.set_defaults(fn=_cmd_kv_math)

    tr = sub.add_parser("trace", help="summarise torch-profiler trace(s) by kernel category")
    tr.add_argument("traces", nargs="+")
    tr.set_defaults(fn=_cmd_trace)

    ln = sub.add_parser("links", help="print Perfetto links for traces pushed to GitHub")
    ln.add_argument("results_dir")
    ln.add_argument("--repo", required=True, help="owner/name")
    ln.add_argument("--ref", default="main", help="branch, tag or commit SHA (SHA = permanent link)")
    ln.set_defaults(fn=_cmd_links)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)
