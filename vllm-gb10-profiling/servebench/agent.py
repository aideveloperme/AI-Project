"""Agent-loop latency benchmark.

An agent turn is a *chain* of LLM calls where each call's prompt is the
previous prompt plus the model's output plus a tool result. Two properties
make agents a distinct serving workload:

1. Latency compounds: task latency = sum over steps of (TTFT + decode + tool).
2. Prompts grow monotonically and share almost everything with the previous
   step, so prefix caching turns an O(n^2) prefill bill into O(n).

Modes
-----
react   Real tool calling (OpenAI ``tools`` API; vLLM needs
        ``--enable-auto-tool-choice --tool-call-parser hermes`` for Qwen3).
        Measures latency *and* task success, so it also catches quality
        regressions from e.g. quantisation.
replay  Fixed-shape trajectory: N steps, fixed output length, canned tool
        results. Model behaviour cannot change the prompt shape, so latency is
        comparable across server configs bit-for-bit. Use this for before/after.
"""

from __future__ import annotations

import asyncio
import json
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from servebench import prom
from servebench.agent_tools import TASKS, TOOL_SCHEMAS, Task, execute_tool
from servebench.client import stream_chat
from servebench.loadgen import make_client
from servebench.workloads import make_text

HANDBOOK = """You are OpsAgent, a customer-operations assistant for an online electronics store.
You answer questions about orders, refunds, taxes and shipping by calling tools.
Rules:
- Never guess order data: always call lookup_order first.
- Use calculator for any arithmetic, including tax (8 percent of subtotal).
- Use search_docs for policy questions before answering.
- Use get_shipping_eta for delivery estimates.
- When you have the answer, reply with a short final answer containing the number or yes/no.
- Do not call more tools than necessary.
"""


def system_prompt(target_tokens: int, seed: int = 7) -> str:
    """Handbook + a deterministic reference appendix padding to ~target_tokens.

    Production agent prompts are typically 1-4k tokens (instructions, tool
    specs, few-shot examples). The padding stands in for that bulk.
    """
    import random

    pad = max(0, target_tokens - 150)
    appendix = make_text(random.Random(seed), pad) if pad else ""
    return HANDBOOK + ("\nReference appendix:\n" + appendix if appendix else "")


@dataclass
class StepRecord:
    session: int
    episode: int
    task_id: str
    step: int
    prompt_tokens: int
    cached_tokens: int | None
    completion_tokens: int
    ttft_s: float | None
    llm_s: float | None
    tool_s: float
    tool_calls: list[str] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


@dataclass
class EpisodeRecord:
    session: int
    episode: int
    task_id: str
    steps: int
    total_s: float
    llm_s: float
    tool_s: float
    ttft_sum_s: float
    success: bool | None
    final_answer: str = ""
    error: str | None = None


@dataclass
class AgentConfig:
    mode: str = "replay"  # replay | react
    sessions: int = 1
    episodes_per_session: int = 5
    max_steps: int = 6
    system_prompt_tokens: int = 2000
    # replay-mode shape
    replay_steps: int = 6
    step_output_tokens: int = 64
    tool_output_tokens: int = 300
    final_output_tokens: int = 128
    # react-mode
    max_tokens_per_step: int = 512
    tool_latency_ms: float = 0.0  # simulated external API latency per tool call
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _ticket(cfg: AgentConfig, session: int, episode: int) -> str:
    # Distinct across agent runs on the same server, so no run is served from another run's cache.
    return f"{cfg.mode}-x{cfg.sessions}-{session}-{episode}"


async def _react_episode(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    cfg: AgentConfig,
    session: int,
    episode: int,
    task: Task,
    sys_prompt: str,
    steps_out: list[StepRecord],
) -> EpisodeRecord:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"[ticket {_ticket(cfg, session, episode)}] {task.question}"},
    ]
    t0 = time.perf_counter()
    llm_s = tool_s = ttft_sum = 0.0
    final = ""
    err = None
    step = 0
    for step in range(1, cfg.max_steps + 1):
        r = await stream_chat(
            client,
            base_url,
            model,
            messages,
            request_id=f"agent-s{session}-e{episode}-{step}",
            max_tokens=cfg.max_tokens_per_step,
            tools=TOOL_SCHEMAS,
            extra_body=cfg.extra_body,
        )
        llm_s += r.e2e_s or 0.0
        ttft_sum += r.ttft_s or 0.0
        rec = StepRecord(
            session,
            episode,
            task.task_id,
            step,
            r.prompt_tokens,
            r.cached_tokens,
            r.completion_tokens,
            r.ttft_s,
            r.e2e_s,
            0.0,
            ok=r.ok,
            error=r.error,
        )
        steps_out.append(rec)
        if not r.ok:
            err = r.error
            break
        if not r.tool_calls:
            final = r.text
            break
        messages.append(
            {
                "role": "assistant",
                "content": r.text or None,
                "tool_calls": [
                    {"id": tc["id"] or f"call_{step}_{i}", "type": "function", "function": tc["function"]}
                    for i, tc in enumerate(r.tool_calls)
                ],
            }
        )
        t_tool = time.perf_counter()
        for i, tc in enumerate(r.tool_calls):
            if cfg.tool_latency_ms:
                await asyncio.sleep(cfg.tool_latency_ms / 1000)
            out = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{step}_{i}", "content": out})
            rec.tool_calls.append(tc["function"]["name"])
        rec.tool_s = time.perf_counter() - t_tool
        tool_s += rec.tool_s
    else:
        err = f"no final answer within {cfg.max_steps} steps"
    low = final.lower()
    success = bool(final) and any(e.lower() in low for e in task.expected)
    return EpisodeRecord(
        session,
        episode,
        task.task_id,
        step,
        time.perf_counter() - t0,
        llm_s,
        tool_s,
        ttft_sum,
        success,
        final[:300],
        err,
    )


async def _replay_episode(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    cfg: AgentConfig,
    session: int,
    episode: int,
    task: Task,
    sys_prompt: str,
    steps_out: list[StepRecord],
) -> EpisodeRecord:
    import random

    rng = random.Random(zlib.crc32(_ticket(cfg, session, episode).encode()))
    # Unique per-episode question so episodes share only the system prompt.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"[ticket {_ticket(cfg, session, episode)}] {task.question}"},
    ]
    t0 = time.perf_counter()
    llm_s = tool_s = ttft_sum = 0.0
    err = None
    final = ""
    for step in range(1, cfg.replay_steps + 1):
        last = step == cfg.replay_steps
        r = await stream_chat(
            client,
            base_url,
            model,
            messages,
            request_id=f"replay-s{session}-e{episode}-{step}",
            max_tokens=cfg.final_output_tokens if last else cfg.step_output_tokens,
            ignore_eos=True,
            extra_body=cfg.extra_body,
        )
        llm_s += r.e2e_s or 0.0
        ttft_sum += r.ttft_s or 0.0
        rec = StepRecord(
            session,
            episode,
            task.task_id,
            step,
            r.prompt_tokens,
            r.cached_tokens,
            r.completion_tokens,
            r.ttft_s,
            r.e2e_s,
            0.0,
            ok=r.ok,
            error=r.error,
        )
        steps_out.append(rec)
        if not r.ok:
            err = r.error
            break
        if last:
            final = r.text
            break
        messages.append({"role": "assistant", "content": r.text or "(tool call)"})
        t_tool = time.perf_counter()
        if cfg.tool_latency_ms:
            await asyncio.sleep(cfg.tool_latency_ms / 1000)
        tool_text = make_text(rng, cfg.tool_output_tokens)
        messages.append({"role": "user", "content": f"Tool result (step {step}): {tool_text}"})
        rec.tool_s = time.perf_counter() - t_tool
        rec.tool_calls.append("replay_tool")
        tool_s += rec.tool_s
    return EpisodeRecord(
        session, episode, task.task_id, step, time.perf_counter() - t0, llm_s, tool_s, ttft_sum, None, final[:120], err
    )


async def run_agent(base_url: str, model: str, cfg: AgentConfig) -> dict[str, Any]:
    if cfg.mode not in {"react", "replay"}:
        raise ValueError("agent mode must be 'react' or 'replay'")
    sys_prompt = system_prompt(cfg.system_prompt_tokens)
    steps: list[StepRecord] = []
    episode_fn = _react_episode if cfg.mode == "react" else _replay_episode

    async with make_client() as client:
        before = await prom.scrape(client, base_url)

        async def session(s: int) -> list[EpisodeRecord]:
            out = []
            for e in range(cfg.episodes_per_session):
                task = TASKS[(s + e) % len(TASKS)]
                out.append(await episode_fn(client, base_url, model, cfg, s, e, task, sys_prompt, steps))
            return out

        t0 = time.perf_counter()
        per_session = await asyncio.gather(*[session(s) for s in range(cfg.sessions)])
        wall = time.perf_counter() - t0
        after = await prom.scrape(client, base_url)

    episodes = [e for sess in per_session for e in sess]
    return {
        "config": asdict(cfg),
        "wall_s": wall,
        "summary": summarize_agent(episodes, steps, wall),
        "server": prom.counter_delta(before, after),
        "episodes": [asdict(e) for e in episodes],
        "steps": [asdict(s) for s in steps],
    }


def summarize_agent(episodes: list[EpisodeRecord], steps: list[StepRecord], wall_s: float) -> dict[str, Any]:
    import numpy as np

    def pct(vals: list[float], p: float) -> float | None:
        return round(float(np.percentile(vals, p)) * 1000, 2) if vals else None

    ok_eps = [e for e in episodes if e.error is None]
    totals = [e.total_s for e in ok_eps]
    ok_steps = [s for s in steps if s.ok and s.ttft_s is not None]
    by_step: dict[int, list[StepRecord]] = {}
    for s in ok_steps:
        by_step.setdefault(s.step, []).append(s)
    per_step = []
    for k in sorted(by_step):
        group = by_step[k]
        cached = [s.cached_tokens for s in group if s.cached_tokens is not None]
        per_step.append(
            {
                "step": k,
                "n": len(group),
                "mean_prompt_tokens": round(sum(s.prompt_tokens for s in group) / len(group), 1),
                "mean_cached_tokens": round(sum(cached) / len(cached), 1) if cached else None,
                "ttft_p50_ms": pct([s.ttft_s for s in group], 50),  # type: ignore[misc]
                "ttft_p90_ms": pct([s.ttft_s for s in group], 90),  # type: ignore[misc]
                "llm_p50_ms": pct([s.llm_s for s in group if s.llm_s is not None], 50),
            }
        )
    scored = [e for e in episodes if e.success is not None]
    llm = sum(e.llm_s for e in ok_eps)
    tool = sum(e.tool_s for e in ok_eps)
    return {
        "episodes": len(episodes),
        "episodes_ok": len(ok_eps),
        "task_success_rate": round(sum(e.success for e in scored) / len(scored), 3) if scored else None,
        "mean_steps": round(sum(e.steps for e in ok_eps) / len(ok_eps), 2) if ok_eps else None,
        "task_latency_p50_ms": pct(totals, 50),
        "task_latency_p90_ms": pct(totals, 90),
        "task_latency_p99_ms": pct(totals, 99),
        "ttft_share_of_llm_time": round(sum(e.ttft_sum_s for e in ok_eps) / llm, 3) if llm else None,
        "tool_share_of_task_time": round(tool / (llm + tool), 3) if llm + tool else None,
        "episodes_per_min": round(len(ok_eps) / wall_s * 60, 2) if wall_s else None,
        "per_step": per_step,
        "errors": [e.error for e in episodes if e.error][:5],
    }


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
