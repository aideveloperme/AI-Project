"""Streaming OpenAI-compatible chat client with per-token timing.

Every request is streamed so we can measure what users actually feel:

* TTFT  - time to first token (queueing + prefill)
* ITL   - inter-token latency between streamed chunks (decode cadence)
* TPOT  - (e2e - ttft) / (output_tokens - 1), the per-token decode cost
* E2E   - full request latency

Token counts come from the server's ``usage`` block (``stream_options.include_usage``)
rather than from counting chunks, because one chunk may carry several tokens
(async scheduling, speculative decoding).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class RequestResult:
    request_id: str
    ok: bool = False
    error: str | None = None
    status_code: int | None = None
    start_ts: float = 0.0  # wall-clock epoch seconds, for timelines
    ttft_s: float | None = None
    e2e_s: float | None = None
    itl_s: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    finish_reason: str | None = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.e2e_s is None or self.completion_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.completion_tokens - 1)

    def to_record(self, include_text: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d["tpot_s"] = self.tpot_s
        if not include_text:
            d.pop("text", None)
        return d


def _merge_tool_call_delta(acc: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    """Accumulate streamed tool-call fragments (OpenAI delta format)."""
    for d in deltas:
        idx = d.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        slot = acc[idx]
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    request_id: str,
    max_tokens: int,
    temperature: float = 0.0,
    ignore_eos: bool = False,
    tools: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = 600.0,
) -> RequestResult:
    """Send one streaming chat completion and time it."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True  # vLLM extension: forces exactly max_tokens outputs
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra_body:
        body.update(extra_body)

    res = RequestResult(request_id=request_id, start_ts=time.time())
    t0 = time.perf_counter()
    last_tok_t: float | None = None
    text_parts: list[str] = []
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=body,
            timeout=timeout_s,
            headers={"X-Request-Id": request_id},
        ) as resp:
            res.status_code = resp.status_code
            if resp.status_code != 200:
                res.error = (await resp.aread()).decode(errors="replace")[:500]
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    u = chunk["usage"]
                    res.prompt_tokens = u.get("prompt_tokens", 0) or 0
                    res.completion_tokens = u.get("completion_tokens", 0) or 0
                    details = u.get("prompt_tokens_details") or {}
                    if details.get("cached_tokens") is not None:
                        res.cached_tokens = details["cached_tokens"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    got_token = bool(
                        delta.get("content")
                        or delta.get("tool_calls")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if got_token:
                        now = time.perf_counter()
                        if res.ttft_s is None:
                            res.ttft_s = now - t0
                        else:
                            res.itl_s.append(now - last_tok_t)  # type: ignore[operator]
                        last_tok_t = now
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    if delta.get("tool_calls"):
                        _merge_tool_call_delta(res.tool_calls, delta["tool_calls"])
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
        res.e2e_s = time.perf_counter() - t0
        res.text = "".join(text_parts)
        res.ok = res.ttft_s is not None
        if not res.ok and res.error is None:
            res.error = "stream ended without any token"
    except (httpx.HTTPError, json.JSONDecodeError) as e:  # network / server crash
        res.error = f"{type(e).__name__}: {e}"
        res.e2e_s = time.perf_counter() - t0
    return res
