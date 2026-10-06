"""Aggregate per-request results into latency/throughput/goodput statistics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from servebench.client import RequestResult

PCTS = (50, 90, 99)


def _stats(values: Iterable[float | None], scale: float = 1000.0) -> dict[str, float | None]:
    arr = np.array([v for v in values if v is not None], dtype=float) * scale
    if arr.size == 0:
        return {"mean": None, **{f"p{p}": None for p in PCTS}, "max": None}
    out: dict[str, float | None] = {"mean": round(float(arr.mean()), 3)}
    for p in PCTS:
        out[f"p{p}"] = round(float(np.percentile(arr, p)), 3)
    out["max"] = round(float(arr.max()), 3)
    return out


def summarize(
    results: list[RequestResult],
    wall_s: float,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
) -> dict[str, Any]:
    """Summarise a batch of requests that ran over ``wall_s`` seconds.

    Latencies are reported in milliseconds. *Goodput* counts only requests
    meeting both SLOs - the number that matters for capacity planning,
    because raw throughput keeps rising long after users are unhappy.
    """
    ok = [r for r in results if r.ok]
    out_tok = sum(r.completion_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    cached = [r.cached_tokens for r in ok if r.cached_tokens is not None]
    itls = [x for r in ok for x in r.itl_s]

    def meets(r: RequestResult) -> bool:
        if slo_ttft_ms is not None and (r.ttft_s is None or r.ttft_s * 1000 > slo_ttft_ms):
            return False
        tpot = r.tpot_s
        if slo_tpot_ms is not None and tpot is not None and tpot * 1000 > slo_tpot_ms:
            return False
        return True

    good = [r for r in ok if meets(r)]
    wall = max(wall_s, 1e-9)
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = f"{r.status_code}: {(r.error or '')[:80]}"
            errors[key] = errors.get(key, 0) + 1
    return {
        "num_requests": len(results),
        "num_ok": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else None,
        "errors": errors,
        "wall_s": round(wall_s, 3),
        "request_throughput_rps": round(len(ok) / wall, 4),
        "output_tok_per_s": round(out_tok / wall, 2),
        "total_tok_per_s": round((out_tok + in_tok) / wall, 2),
        "mean_prompt_tokens": round(in_tok / len(ok), 1) if ok else None,
        "mean_output_tokens": round(out_tok / len(ok), 1) if ok else None,
        "cached_prompt_token_frac": round(sum(cached) / in_tok, 4) if cached and in_tok else None,
        "ttft_ms": _stats(r.ttft_s for r in ok),
        "tpot_ms": _stats(r.tpot_s for r in ok),
        "itl_ms": _stats(itls),
        "e2e_ms": _stats(r.e2e_s for r in ok),
        "slo": {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms},
        "goodput_rps": round(len(good) / wall, 4),
        "slo_attainment": round(len(good) / len(results), 4) if results else None,
    }
