"""A discrete-time simulator of a vLLM-v1-style engine on a GB10-class device.

**This is not a performance claim about GB10.** It exists so that the
harness, report pipeline and CI can run without a GPU, and so that the
*direction* of each knob's effect can be reasoned about before spending
time on the device. Real numbers come only from ``results/gb10``.

What it models (deliberately simple, first-order):

* Roofline step time: ``max(FLOPs / eff_compute, bytes / eff_bandwidth) + overhead``
  where bytes = weights (read once per step) + KV read by attention.
  GB10 decode is memory-bandwidth bound (LPDDR5x, ~273 GB/s), so weight
  bytes per step dominate: halving weight bytes (FP8) ~ halves TPOT at low
  batch.
* Paged KV cache with 16-token blocks, capacity derived from
  ``--gpu-memory-utilization``, model geometry and ``--kv-cache-dtype``.
* Continuous batching with a per-step token budget (``--max-num-batched-tokens``),
  chunked prefill, a ``--max-num-seqs`` cap and recompute-preemption when KV runs out.
* Hash-chained automatic prefix caching with LRU eviction of free blocks.
* ``--max-model-len`` admission control (HTTP 400 like vLLM).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import time
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BLOCK = 16

# Geometry from the public HF config.json of each model.
MODELS: dict[str, dict[str, float]] = {
    "qwen3-8b": {"params": 8.19e9, "layers": 36, "kv_heads": 8, "head_dim": 128},
    "qwen3-4b": {"params": 4.02e9, "layers": 36, "kv_heads": 8, "head_dim": 128},
    "llama-3.1-8b": {"params": 8.03e9, "layers": 32, "kv_heads": 8, "head_dim": 128},
}


def model_geometry(name: str) -> dict[str, float]:
    low = name.lower()
    for key, geo in MODELS.items():
        if key in low.replace("_", "-").replace("meta-", ""):
            return geo
    return MODELS["qwen3-8b"]


def weight_bytes_per_param(model: str, quantization: str | None) -> float:
    q = (quantization or "").lower()
    low = model.lower()
    if q in {"awq", "gptq", "nvfp4", "modelopt_fp4"} or "awq" in low or "fp4" in low or "int4" in low:
        return 0.5
    if q in {"fp8", "modelopt"} or "fp8" in low:
        return 1.0
    return 2.0


@dataclass
class SimConfig:
    model: str
    max_model_len: int = 32768
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.65
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"
    quantization: str | None = None
    # Hardware model ("GB10-like"). Bandwidth is the published LPDDR5x figure;
    # compute and efficiencies are assumptions, tunable from the CLI.
    total_mem_gb: float = 128.0
    mem_bw_gbs: float = 273.0
    bw_efficiency: float = 0.75
    dense_tflops: float = 100.0
    mfu: float = 0.45
    step_overhead_ms: float = 4.0
    activation_reserve_gb: float = 6.0
    time_scale: float = 1.0  # <1 runs faster than "real" simulated time (CI)

    @property
    def geo(self) -> dict[str, float]:
        return model_geometry(self.model)

    @property
    def weight_bytes(self) -> float:
        return self.geo["params"] * weight_bytes_per_param(self.model, self.quantization)

    @property
    def kv_bytes_per_token(self) -> float:
        g = self.geo
        dtype_bytes = 1.0 if self.kv_cache_dtype.startswith("fp8") else 2.0
        return 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * dtype_bytes

    @property
    def num_blocks(self) -> int:
        budget = (
            self.total_mem_gb * 1e9 * self.gpu_memory_utilization - self.weight_bytes - self.activation_reserve_gb * 1e9
        )
        if budget <= 0:
            raise ValueError("no memory left for KV cache: raise --gpu-memory-utilization")
        return int(budget // (self.kv_bytes_per_token * BLOCK))


def tokenize(text: str) -> list[int]:
    return [zlib.crc32(w.encode()) for w in text.split()]


def chat_tokens(messages: list[dict[str, Any]], tools: list[Any] | None) -> list[int]:
    toks: list[int] = []
    if tools:
        toks += tokenize(json.dumps(tools))
    for m in messages:
        toks += [1, zlib.crc32(str(m.get("role")).encode())]  # role header, ~ chat template
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        toks += tokenize(str(content))
        if m.get("tool_calls"):
            toks += tokenize(json.dumps(m["tool_calls"]))
        toks.append(2)
    return toks


@dataclass
class SimRequest:
    rid: str
    prompt: list[int]
    max_tokens: int
    out_plan: list[dict[str, Any]]  # one delta per generated token
    queue: asyncio.Queue
    arrival: float = field(default_factory=time.perf_counter)
    computed: int = 0  # prompt tokens whose KV exists
    generated: int = 0
    cached_tokens: int = 0
    hashed_blocks: list[int] = field(default_factory=list)
    plain_blocks: int = 0
    preemptions: int = 0
    admitted_once: bool = False

    @property
    def context_len(self) -> int:
        return self.computed + self.generated

    @property
    def prefilling(self) -> bool:
        return self.computed < len(self.prompt)


class BlockManager:
    """Paged KV with hash-chained prefix caching and LRU eviction."""

    def __init__(self, num_blocks: int, prefix_caching: bool):
        self.total = num_blocks
        self.prefix_caching = prefix_caching
        self.free_plain = num_blocks
        self.cache: OrderedDict[int, int] = OrderedDict()  # hash -> refcount (0 => evictable)

    def evictable(self) -> int:
        return sum(1 for r in self.cache.values() if r == 0)

    def available(self) -> int:
        return self.free_plain + self.evictable()

    def usage(self) -> float:
        return 1.0 - self.available() / self.total

    def _take(self, n: int) -> bool:
        if self.available() < n:
            return False
        from_free = min(n, self.free_plain)
        self.free_plain -= from_free
        n -= from_free
        for h in [h for h, r in self.cache.items() if r == 0][:n]:  # LRU order
            del self.cache[h]
        return True

    @staticmethod
    def block_hashes(tokens: list[int]) -> list[int]:
        hashes, prev = [], 0
        for i in range(len(tokens) // BLOCK):
            prev = zlib.crc32(json.dumps([prev, tokens[i * BLOCK : (i + 1) * BLOCK]]).encode())
            hashes.append(prev)
        return hashes

    def admit(self, req: SimRequest) -> bool:
        """Allocate blocks for the whole prompt (+1 token). Returns False if it doesn't fit."""
        hashes = self.block_hashes(req.prompt) if self.prefix_caching else []
        hit = 0
        for h in hashes:
            if h in self.cache:
                hit += 1
            else:
                break
        # Always recompute at least the last prompt token (needed for logits).
        hit = min(hit, (len(req.prompt) - 1) // BLOCK)
        need_total = math.ceil((len(req.prompt) + 1) / BLOCK)
        new_hashed = hashes[hit:]
        need_new = need_total - hit
        # Pinning hit blocks removes them from the evictable pool.
        pinned_evictable = sum(1 for h in hashes[:hit] if self.cache[h] == 0)
        if self.available() - pinned_evictable < need_new:
            return False
        for h in hashes[:hit]:
            self.cache[h] += 1
            self.cache.move_to_end(h)
        assert self._take(need_new)
        for h in new_hashed:
            self.cache[h] = self.cache.get(h, 0) + 1
        req.hashed_blocks = hashes[:hit] + new_hashed
        req.plain_blocks = need_new - len(new_hashed)
        req.cached_tokens = hit * BLOCK
        req.computed = hit * BLOCK
        return True

    def grow(self, req: SimRequest) -> bool:
        """Ensure a slot for the next generated token."""
        have = (len(req.hashed_blocks) + req.plain_blocks) * BLOCK
        if req.context_len + 1 <= have:
            return True
        if not self._take(1):
            return False
        req.plain_blocks += 1
        return True

    def release(self, req: SimRequest) -> None:
        for h in req.hashed_blocks:
            if h in self.cache:
                self.cache[h] -= 1
                if self.cache[h] <= 0:
                    if self.prefix_caching:
                        self.cache[h] = 0
                        self.cache.move_to_end(h)
                    else:
                        del self.cache[h]
                        self.free_plain += 1
        self.free_plain += req.plain_blocks
        req.hashed_blocks, req.plain_blocks = [], 0


class SimEngine:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.blocks = BlockManager(cfg.num_blocks, cfg.enable_prefix_caching)
        self.waiting: deque[SimRequest] = deque()
        self.running: list[SimRequest] = []
        self.counters = {
            k: 0
            for k in (
                "prefix_cache_queries",
                "prefix_cache_hits",
                "num_preemptions",
                "prompt_tokens",
                "generation_tokens",
                "prompt_tokens_cached",
                "request_success",
            )
        }
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.profiling = False
        self.profile_events: list[dict[str, Any]] = []
        self._profile_clock = 0.0

    # ---- public API -------------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def submit(self, req: SimRequest) -> None:
        self.waiting.append(req)
        self._wake.set()

    def capacity_tokens(self) -> int:
        return self.blocks.total * BLOCK

    # ---- cost model -------------------------------------------------------
    def step_time_s(self, n_prefill: int, n_decode: int, kv_read_tokens: int, attn_prefill_ctx: int) -> float:
        c = self.cfg
        tokens = n_prefill + n_decode
        flops = (
            2 * c.geo["params"] * tokens
            + 4 * c.geo["layers"] * c.geo["kv_heads"] * 4 * c.geo["head_dim"] * attn_prefill_ctx
        )
        # Blackwell tensor cores: FP8 runs at 2x and FP4 at 4x the BF16 rate when weights are quantised.
        speedup = 2.0 / weight_bytes_per_param(c.model, c.quantization)
        t_compute = flops / (c.dense_tflops * speedup * 1e12 * c.mfu)
        bytes_moved = c.weight_bytes + kv_read_tokens * c.kv_bytes_per_token
        t_mem = bytes_moved / (c.mem_bw_gbs * 1e9 * c.bw_efficiency)
        return max(t_compute, t_mem) + c.step_overhead_ms / 1000

    # ---- scheduler --------------------------------------------------------
    def _preempt_one(self) -> bool:
        if not self.running:
            return False
        victim = self.running.pop()  # newest first, like vLLM
        self.blocks.release(victim)
        victim.computed = 0
        victim.preemptions += 1
        self.counters["num_preemptions"] += 1
        self.waiting.appendleft(victim)
        return True

    def schedule(self) -> tuple[list[tuple[SimRequest, int]], list[SimRequest]]:
        """Return (prefill chunks, decode requests) for one step."""
        budget = self.cfg.max_num_batched_tokens
        decodes: list[SimRequest] = []
        for req in list(self.running):
            if req.prefilling or budget <= 0:
                continue
            while not self.blocks.grow(req):
                if not self._preempt_one() or req not in self.running:
                    break
            if req in self.running:
                decodes.append(req)
                budget -= 1
        prefills: list[tuple[SimRequest, int]] = []
        for req in self.running:  # continue chunked prefills
            if req.prefilling and budget > 0:
                n = min(budget, len(req.prompt) - req.computed)
                prefills.append((req, n))
                budget -= n
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            req = self.waiting[0]
            if not self.blocks.admit(req):
                break
            self.waiting.popleft()
            if not req.admitted_once:
                req.admitted_once = True
                self.counters["prefix_cache_queries"] += len(req.prompt)
                self.counters["prefix_cache_hits"] += req.cached_tokens
                self.counters["prompt_tokens"] += len(req.prompt)
                self.counters["prompt_tokens_cached"] += req.cached_tokens
            self.running.append(req)
            n = min(budget, len(req.prompt) - req.computed)
            prefills.append((req, n))
            budget -= n
        return prefills, decodes

    async def _loop(self) -> None:
        while True:
            if not self.running and not self.waiting:
                self._wake.clear()
                await self._wake.wait()
            prefills, decodes = self.schedule()
            if not prefills and not decodes:
                await asyncio.sleep(0.001)
                continue
            n_pref = sum(n for _, n in prefills)
            kv_read = sum(r.context_len for r in decodes) + sum(r.computed for r, _ in prefills)
            attn_ctx = sum(n * (r.computed + n) for r, n in prefills)
            dt = self.step_time_s(n_pref, len(decodes), kv_read, attn_ctx)
            await asyncio.sleep(dt * self.cfg.time_scale)
            if self.profiling:
                self._record_profile(dt, n_pref, len(decodes), kv_read)
            for req, n in prefills:
                req.computed += n
                if not req.prefilling:
                    self._emit(req)  # first token comes out of the final prefill chunk
            for req in decodes:
                self._emit(req)

    def _emit(self, req: SimRequest) -> None:
        delta = req.out_plan[req.generated] if req.generated < len(req.out_plan) else {"content": " ."}
        req.generated += 1
        self.counters["generation_tokens"] += 1
        done = req.generated >= len(req.out_plan)
        req.queue.put_nowait({"delta": delta, "done": done})
        if done:
            self.counters["request_success"] += 1
            self.blocks.release(req)
            self.running.remove(req)

    # ---- profiling --------------------------------------------------------
    def _record_profile(self, dt: float, n_pref: int, n_dec: int, kv_read: int) -> None:
        c = self.cfg
        # Simulated clock: kernel durations are in simulated time, so timestamps must be too.
        ts = self._profile_clock * 1e6
        self._profile_clock += dt
        w = c.weight_bytes / (c.weight_bytes + kv_read * c.kv_bytes_per_token)
        busy = dt - c.step_overhead_ms / 1000
        parts = [
            (
                "sim_gemm_bf16" if weight_bytes_per_param(c.model, c.quantization) == 2 else "sim_scaled_mm_fp8",
                busy * w * 0.92,
            ),
            ("sim_flash_attn_paged_kv", busy * (1 - w) + busy * w * 0.03),
            ("sim_rms_norm_kernel", busy * w * 0.03),
            ("sim_topk_sampler", busy * w * 0.02),
        ]
        cur = ts
        for name, d in parts:
            self.profile_events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "ts": cur,
                    "dur": d * 1e6,
                    "pid": 0,
                    "tid": 7,
                    "args": {"prefill_tokens": n_pref, "decode_reqs": n_dec},
                }
            )
            cur += d * 1e6

    def start_profile(self) -> None:
        self.profiling, self.profile_events, self._profile_clock = True, [], 0.0

    def stop_profile(self, out_dir: str | None) -> Path | None:
        self.profiling = False
        if not out_dir:
            return None
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        f = p / f"simulated_rank0.{int(time.time())}.pt.trace.json.gz"
        with gzip.open(f, "wt") as fh:
            json.dump({"traceEvents": self.profile_events, "simulated": True}, fh)
        return f

    # ---- metrics ----------------------------------------------------------
    def prometheus(self) -> str:
        lbl = f'{{model_name="{self.cfg.model}",engine="0"}}'
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{lbl} {len(self.running)}",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{lbl} {len(self.waiting)}",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc{lbl} {self.blocks.usage():.6f}",
        ]
        for k, v in self.counters.items():
            lines += [f"# TYPE vllm:{k} counter", f"vllm:{k}_total{lbl} {float(v)}"]
        return "\n".join(lines) + "\n"
