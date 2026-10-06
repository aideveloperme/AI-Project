"""Load generation: closed-loop (fixed concurrency) and open-loop (Poisson rate).

Closed loop answers "how does latency scale with N concurrent users?".
Open loop answers "what happens at R req/s?" - it keeps sending even when
the server falls behind, so it exposes queueing collapse that closed loop hides.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from servebench import prom
from servebench.client import RequestResult, stream_chat
from servebench.metrics import summarize
from servebench.workloads import Workload, WorkloadSpec, for_level


@dataclass
class LevelResult:
    workload: str
    mode: str
    level: float
    results: list[RequestResult]
    wall_s: float
    summary: dict[str, Any]
    server_counters: dict[str, Any]
    timeseries: list[dict[str, Any]] = field(default_factory=list)


def make_client(max_conn: int = 1024) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
    return httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0, connect=10.0))


async def wait_ready(base_url: str, timeout_s: float = 1800.0, poll_s: float = 2.0) -> None:
    """Block until ``/health`` returns 200 (model load + CUDA graph capture can take minutes)."""
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                r = await c.get(f"{base_url.rstrip('/')}/health", timeout=5)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll_s)
    raise TimeoutError(f"server at {base_url} not healthy after {timeout_s}s")


async def run_level(
    base_url: str,
    model: str,
    spec: WorkloadSpec,
    *,
    mode: str = "concurrency",
    level: float = 1,
    num_requests: int = 32,
    warmup_requests: int = 2,
    ignore_eos: bool = True,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    max_concurrency: int | None = None,
    sample_interval_s: float = 1.0,
) -> LevelResult:
    if mode not in {"concurrency", "rate"}:
        raise ValueError("mode must be 'concurrency' or 'rate'")
    wl = Workload(for_level(spec, level, salt=0 if mode == "concurrency" else 1))
    prompts = [wl.next() for _ in range(num_requests)]
    extra = dict(spec.extra_body)

    async with make_client() as client:
        # Warmup with *different* prompts so they don't seed the prefix cache
        # for the measured random workload (shared_prefix warmup is intended:
        # a production system prompt is always warm).
        warm_wl = Workload(for_level(spec, level, salt=99991))
        await asyncio.gather(
            *[
                stream_chat(
                    client,
                    base_url,
                    model,
                    p.messages,
                    request_id=f"warmup-{i}",
                    max_tokens=min(p.max_tokens, 16),
                    extra_body=extra,
                )
                for i, p in enumerate(warm_wl.next() for _ in range(warmup_requests))
            ]
        )

        before = await prom.scrape(client, base_url)
        sampler = prom.MetricsSampler(client, base_url, sample_interval_s)
        sampler.start()
        tag = f"{spec.name}-{mode}{level:g}"

        async def one(i: int) -> RequestResult:
            p = prompts[i]
            return await stream_chat(
                client,
                base_url,
                model,
                p.messages,
                request_id=f"{tag}-{i}",
                max_tokens=p.max_tokens,
                ignore_eos=ignore_eos,
                extra_body=extra,
            )

        t0 = time.perf_counter()
        if mode == "concurrency":
            sem = asyncio.Semaphore(int(level))

            async def gated(i: int) -> RequestResult:
                async with sem:
                    return await one(i)

            results = await asyncio.gather(*[gated(i) for i in range(num_requests)])
        else:
            rng = random.Random(spec.seed)
            sem = asyncio.Semaphore(max_concurrency or 10**6)
            tasks = []
            for i in range(num_requests):

                async def gated(i: int = i) -> RequestResult:
                    async with sem:
                        return await one(i)

                tasks.append(asyncio.create_task(gated()))
                await asyncio.sleep(rng.expovariate(level))
            results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        await sampler.stop()
        after = await prom.scrape(client, base_url)

    counters = prom.counter_delta(before, after)
    counters["peak_kv_cache_usage"] = sampler.peak("kv_cache_usage")
    counters["peak_running"] = sampler.peak("running")
    counters["peak_waiting"] = sampler.peak("waiting")
    summary = summarize(list(results), wall, slo_ttft_ms, slo_tpot_ms)
    summary.update({"workload": spec.name, "mode": mode, "level": level, "server": counters})
    return LevelResult(spec.name, mode, level, list(results), wall, summary, counters, sampler.rows)
