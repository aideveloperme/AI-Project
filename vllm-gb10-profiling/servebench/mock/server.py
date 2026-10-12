"""OpenAI-compatible HTTP front-end for the simulator.

Accepts the subset of ``vllm serve`` flags that the simulator models and
ignores the rest, so the *same* experiment YAML runs on the device and in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import time
import uuid
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from servebench.mock.engine import SimConfig, SimEngine, SimRequest, chat_tokens
from servebench.workloads import WORDS


def _out_plan(body: dict[str, Any], prompt_len: int) -> tuple[list[dict[str, Any]], str]:
    """Decide the generated deltas up-front (deterministic per prompt)."""
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
    rng = random.Random(prompt_len * 7919 + max_tokens)
    msgs = body.get("messages", [])
    tools = body.get("tools") or []
    n_tool_msgs = sum(1 for m in msgs if m.get("role") == "tool")
    if tools and n_tool_msgs < 2 and body.get("tool_choice") != "none":
        fn = tools[n_tool_msgs % len(tools)]["function"]
        req = fn.get("parameters", {}).get("required", [])
        args = json.dumps({k: "A-1042" if "order" in k else "test" for k in req})
        n = min(max_tokens, 12)
        pieces = [args[i * len(args) // n : (i + 1) * len(args) // n] for i in range(n)]
        plan = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": pieces[0]},
                    }
                ]
            }
        ]
        plan += [{"tool_calls": [{"index": 0, "function": {"arguments": p}}]} for p in pieces[1:]]
        return plan, "tool_calls"
    n = max_tokens if body.get("ignore_eos") else min(max_tokens, 24 + rng.randrange(64))
    return [{"content": " " + rng.choice(WORDS)} for _ in range(max(1, n))], ("length" if n == max_tokens else "stop")


def build_app(cfg: SimConfig, profile_dir: str | None, served_name: str) -> Starlette:
    engine = SimEngine(cfg)
    if cfg.max_model_len > engine.capacity_tokens():
        raise SystemExit(
            f"KV cache holds {engine.capacity_tokens()} tokens < max_model_len {cfg.max_model_len}: "
            "lower --max-model-len or raise --gpu-memory-utilization (same failure as vLLM)"
        )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        engine.start()
        yield

    async def health(_: Request) -> Response:
        return Response(status_code=200)

    async def version(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "version": "simulated-gb10",
                "simulated": True,
                "kv_cache_tokens": engine.capacity_tokens(),
                "kv_bytes_per_token": cfg.kv_bytes_per_token,
                "weight_gb": round(cfg.weight_bytes / 1e9, 2),
            }
        )

    async def models(_: Request) -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": served_name, "object": "model", "max_model_len": cfg.max_model_len}]}
        )

    async def metrics(_: Request) -> PlainTextResponse:
        return PlainTextResponse(engine.prometheus())

    async def start_profile(_: Request) -> Response:
        if not profile_dir:
            return JSONResponse({"error": "profiler not enabled"}, status_code=404)
        engine.start_profile()
        return Response(status_code=200)

    async def stop_profile(_: Request) -> Response:
        if not profile_dir:
            return JSONResponse({"error": "profiler not enabled"}, status_code=404)
        engine.stop_profile(profile_dir)
        return Response(status_code=200)

    async def chat(request: Request) -> Response:
        body = await request.json()
        if body.get("model") != served_name:
            return JSONResponse({"error": {"message": f"model {body.get('model')} not found"}}, status_code=404)
        prompt = chat_tokens(body.get("messages", []), body.get("tools"))
        plan, finish = _out_plan(body, len(prompt))
        if len(prompt) + len(plan) > cfg.max_model_len:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"This model's maximum context length is {cfg.max_model_len} tokens. "
                            f"However, you requested {len(prompt) + len(plan)} tokens "
                            f"({len(prompt)} in the messages, {len(plan)} in the completion)."
                        ),
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
                status_code=400,
            )
        q: asyncio.Queue = asyncio.Queue()
        req = SimRequest(
            rid=request.headers.get("x-request-id", uuid.uuid4().hex),
            prompt=prompt,
            max_tokens=len(plan),
            out_plan=plan,
            queue=q,
        )
        engine.submit(req)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta: dict[str, Any], fin: str | None = None) -> str:
            return (
                "data: "
                + json.dumps(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": served_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
                    }
                )
                + "\n\n"
            )

        if not body.get("stream"):
            parts: list[dict[str, Any]] = []
            while True:
                ev = await q.get()
                parts.append(ev["delta"])
                if ev["done"]:
                    break
            text = "".join(p.get("content", "") for p in parts)
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "model": served_name,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": len(parts),
                        "total_tokens": len(prompt) + len(parts),
                    },
                }
            )

        async def gen():
            yield chunk({"role": "assistant", "content": ""})
            n = 0
            while True:
                ev = await q.get()
                n += 1
                yield chunk(ev["delta"], finish if ev["done"] else None)
                if ev["done"]:
                    break
            if (body.get("stream_options") or {}).get("include_usage"):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "model": served_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(prompt),
                                "completion_tokens": n,
                                "total_tokens": len(prompt) + n,
                                "prompt_tokens_details": {"cached_tokens": req.cached_tokens},
                            },
                        }
                    )
                    + "\n\n"
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    routes = [
        Route("/health", health),
        Route("/version", version),
        Route("/v1/models", models),
        Route("/metrics", metrics),
        Route("/start_profile", start_profile, methods=["POST"]),
        Route("/stop_profile", stop_profile, methods=["POST"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(prog="servebench.mock", description="GB10-like vLLM simulator")
    p.add_argument("model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--quantization", default=None)
    p.add_argument("--profiler-config", default=None)
    time_scale = float(os.environ.get("SERVEBENCH_MOCK_TIME_SCALE", "1.0"))
    for f, d in (("mem-bw-gbs", 273.0), ("dense-tflops", 100.0), ("time-scale", time_scale), ("step-overhead-ms", 4.0)):
        p.add_argument(f"--sim-{f}", type=float, default=d)
    return p.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, unknown = parse_args(argv)
    if unknown:
        print(f"[mock] ignoring flags not modelled by the simulator: {unknown}")
    cfg = SimConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        mem_bw_gbs=args.sim_mem_bw_gbs,
        dense_tflops=args.sim_dense_tflops,
        time_scale=args.sim_time_scale,
        step_overhead_ms=args.sim_step_overhead_ms,
    )
    profile_dir = None
    if args.profiler_config:
        profile_dir = json.loads(args.profiler_config).get("torch_profiler_dir") or None
    app = build_app(cfg, profile_dir, args.served_model_name or args.model)
    print(
        f"[mock] SIMULATED GB10: weights={cfg.weight_bytes / 1e9:.1f}GB "
        f"kv/token={cfg.kv_bytes_per_token / 1024:.0f}KiB kv_capacity={cfg.num_blocks * 16} tokens",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
