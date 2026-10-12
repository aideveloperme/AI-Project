"""Minimal Prometheus scraper for vLLM's ``/metrics`` endpoint.

Metric names are taken from vLLM v1 (``vllm/v1/metrics/loggers.py``).
Counters are exposed with a ``_total`` suffix by prometheus_client.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

import httpx

GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}
COUNTERS = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached_total",
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Return ``{metric_name: value summed over all label sets}``."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
            v = float(value)
        except ValueError:
            continue
        if math.isnan(v):
            continue
        name = name_part.split("{", 1)[0]
        out[name] = out.get(name, 0.0) + v
    return out


def pick(snapshot: dict[str, float]) -> dict[str, float | None]:
    names = {**GAUGES, **COUNTERS}
    return {k: snapshot.get(v) for k, v in names.items()}


async def scrape(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    try:
        r = await client.get(f"{base_url.rstrip('/')}/metrics", timeout=5)
        r.raise_for_status()
        return parse_prometheus(r.text)
    except httpx.HTTPError:
        return {}


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    d: dict[str, float | None] = {}
    for key, metric in COUNTERS.items():
        if metric in before and metric in after:
            d[key] = after[metric] - before[metric]
        else:
            d[key] = None
    q, h = d.get("prefix_cache_queries"), d.get("prefix_cache_hits")
    d["prefix_cache_hit_rate"] = (h / q) if q and h is not None else None
    return d


@dataclass
class MetricsSampler:
    """Background task sampling gauges once per ``interval_s``."""

    client: httpx.AsyncClient
    base_url: str
    interval_s: float = 1.0
    rows: list[dict[str, float | None]] = field(default_factory=list)
    _task: asyncio.Task | None = None

    async def _run(self) -> None:
        t0 = time.perf_counter()
        while True:
            snap = await scrape(self.client, self.base_url)
            row: dict[str, float | None] = {"t_s": round(time.perf_counter() - t0, 3)}
            row.update({k: snap.get(v) for k, v in GAUGES.items()})
            self.rows.append(row)
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def peak(self, key: str) -> float | None:
        vals = [r[key] for r in self.rows if r.get(key) is not None]
        return max(vals) if vals else None  # type: ignore[type-var]
