import asyncio

from servebench.mock.engine import BLOCK, BlockManager, SimConfig, SimEngine, SimRequest


def _req(tokens, n_out=4):
    return SimRequest(
        rid="r", prompt=tokens, max_tokens=n_out, out_plan=[{"content": " x"}] * n_out, queue=asyncio.Queue()
    )


def test_prefix_cache_hit_after_release():
    bm = BlockManager(num_blocks=100, prefix_caching=True)
    toks = list(range(BLOCK * 4 + 3))
    r1 = _req(toks)
    assert bm.admit(r1) and r1.cached_tokens == 0
    bm.release(r1)
    r2 = _req(toks)
    assert bm.admit(r2)
    assert r2.cached_tokens == BLOCK * 4


def test_no_prefix_cache_when_disabled():
    bm = BlockManager(num_blocks=100, prefix_caching=False)
    toks = list(range(BLOCK * 4))
    r1 = _req(toks)
    bm.admit(r1)
    bm.release(r1)
    r2 = _req(toks)
    bm.admit(r2)
    assert r2.cached_tokens == 0 and bm.available() == 100 - 5


def test_full_hit_still_recomputes_last_token():
    bm = BlockManager(num_blocks=100, prefix_caching=True)
    toks = list(range(BLOCK * 2))
    r1 = _req(toks)
    bm.admit(r1)
    bm.release(r1)
    r2 = _req(toks)
    bm.admit(r2)
    assert r2.cached_tokens == BLOCK  # last block recomputed so logits exist


def test_admission_fails_when_full_and_eviction_frees():
    bm = BlockManager(num_blocks=4, prefix_caching=True)
    r1 = _req(list(range(BLOCK * 3)))
    assert bm.admit(r1)
    assert not bm.admit(_req(list(range(1000, 1000 + BLOCK * 3))))
    bm.release(r1)  # blocks become evictable
    assert bm.admit(_req(list(range(1000, 1000 + BLOCK * 3))))


def test_kv_math_fp8_doubles_capacity_per_byte():
    bf16 = SimConfig(model="Qwen/Qwen3-8B")
    fp8 = SimConfig(model="Qwen/Qwen3-8B-FP8", kv_cache_dtype="fp8")
    assert bf16.kv_bytes_per_token == 2 * 36 * 8 * 128 * 2
    assert fp8.kv_bytes_per_token * 2 == bf16.kv_bytes_per_token
    assert fp8.weight_bytes * 2 == bf16.weight_bytes
    assert fp8.num_blocks > 2 * bf16.num_blocks


def test_decode_is_bandwidth_bound_on_gb10():
    e = SimEngine.__new__(SimEngine)
    e.cfg = SimConfig(model="Qwen/Qwen3-8B")
    t_bf16 = e.step_time_s(0, 1, 1000, 0)
    e.cfg = SimConfig(model="Qwen/Qwen3-8B-FP8")
    t_fp8 = e.step_time_s(0, 1, 1000, 0)
    assert 1.7 < t_bf16 / t_fp8 < 2.1


async def test_preemption_under_kv_pressure():
    cfg = SimConfig(model="Qwen/Qwen3-8B", time_scale=0.0, gpu_memory_utilization=0.2)
    cfg_blocks = cfg.num_blocks
    eng = SimEngine(cfg)
    eng.start()
    n_tok = (cfg_blocks // 3) * BLOCK  # three of these nearly fill KV; decode growth forces preemption
    reqs = [
        SimRequest(
            rid=str(i),
            prompt=list(range(i * 10**6, i * 10**6 + n_tok)),
            max_tokens=64,
            out_plan=[{"content": " x"}] * 64,
            queue=asyncio.Queue(),
        )
        for i in range(4)
    ]
    for r in reqs:
        eng.submit(r)

    async def drain(r):
        while not (await r.queue.get())["done"]:
            pass

    await asyncio.wait_for(asyncio.gather(*[drain(r) for r in reqs]), timeout=30)
    assert eng.counters["request_success"] == 4
    assert eng.blocks.available() == cfg_blocks
