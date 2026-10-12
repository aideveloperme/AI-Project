"""Profiling: capture vLLM torch-profiler traces and summarise GPU kernel time.

vLLM 0.31 enables ``/start_profile`` and ``/stop_profile`` when the server is
launched with ``--profiler-config '{"profiler": "torch", "torch_profiler_dir": ...}'``.
Traces are Chrome-trace JSON (gzipped) that open directly in https://ui.perfetto.dev.

The summariser buckets kernels into attention / GEMM / communication /
norm+activation / sampling / other, which is the breakdown you need to
answer "is this config compute-, bandwidth- or overhead-bound?".
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"flash|attn|attention|fmha|paged|mla|reshape_and_cache|unified_attention", re.I)),
    ("gemm", re.compile(r"gemm|matmul|cutlass|cublas|sm\d+_xmma|nvjet|scaled_mm|fp8|w8a8|marlin|gemv|_mm_", re.I)),
    ("communication", re.compile(r"nccl|allreduce|all_reduce|allgather|custom_ar", re.I)),
    ("norm_act", re.compile(r"rms_?norm|layernorm|silu|gelu|act_and_mul|rotary|rope", re.I)),
    ("sampling", re.compile(r"sampl|topk|top_k|softmax|argmax|sort|scatter|gather", re.I)),
]


def categorize(kernel_name: str) -> str:
    for cat, pat in CATEGORIES:
        if pat.search(kernel_name):
            return cat
    return "other"


async def start_profile(base_url: str) -> None:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{base_url.rstrip('/')}/start_profile", timeout=60)
        r.raise_for_status()


async def stop_profile(base_url: str) -> None:
    # Flushing a trace to disk can take a while for large captures.
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{base_url.rstrip('/')}/stop_profile", timeout=600)
        r.raise_for_status()


def _load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        return json.load(f)


def find_traces(directory: Path) -> list[Path]:
    pats = ("*.json", "*.json.gz")
    return sorted(
        {p for pat in pats for p in directory.rglob(pat) if "trace" in p.name or p.name.endswith(".pt.trace.json.gz")}
    )


def summarize_trace(path: Path, top_k: int = 15) -> dict[str, Any]:
    trace = _load_trace(path)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    by_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    by_cat: dict[str, float] = defaultdict(float)
    t_min, t_max = float("inf"), float("-inf")
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if str(ev.get("cat", "")).lower() not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        name = str(ev.get("name", "?"))
        dur = float(ev.get("dur", 0.0))  # microseconds
        ts = float(ev.get("ts", 0.0))
        t_min, t_max = min(t_min, ts), max(t_max, ts + dur)
        slot = by_kernel[name]
        slot[0] += dur
        slot[1] += 1
        by_cat[categorize(name) if ev.get("cat") == "kernel" else "memcpy/memset"] += dur
    total = sum(by_cat.values())
    span = (t_max - t_min) if total else 0.0
    top = sorted(by_kernel.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return {
        "trace": str(path),
        "gpu_kernel_time_ms": round(total / 1000, 3),
        "gpu_span_ms": round(span / 1000, 3),
        # Busy fraction < ~0.8 during decode usually means CPU/launch overhead dominates.
        "gpu_busy_frac": round(total / span, 3) if span else None,
        "by_category_pct": {k: round(100 * v / total, 2) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])}
        if total
        else {},
        "top_kernels": [
            {
                "name": n[:120],
                "total_ms": round(d / 1000, 3),
                "calls": int(c),
                "category": categorize(n),
                "pct": round(100 * d / total, 2) if total else 0.0,
            }
            for n, (d, c) in top
        ],
    }


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = ["# Kernel summary", ""]
    for s in summaries:
        lines += [
            f"## `{Path(s['trace']).name}`",
            "",
            f"- GPU kernel time: **{s['gpu_kernel_time_ms']} ms** over a {s['gpu_span_ms']} ms span "
            f"(busy fraction {s['gpu_busy_frac']})",
            "",
            "| category | % of GPU time |",
            "|---|---:|",
        ]
        lines += [f"| {k} | {v} |" for k, v in s["by_category_pct"].items()]
        lines += ["", "| kernel | category | calls | total ms | % |", "|---|---|---:|---:|---:|"]
        lines += [
            f"| `{k['name']}` | {k['category']} | {k['calls']} | {k['total_ms']} | {k['pct']} |"
            for k in s["top_kernels"]
        ]
        lines.append("")
    return "\n".join(lines)
