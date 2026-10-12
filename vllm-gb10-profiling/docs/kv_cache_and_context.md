# KV cache and context handling on GB10

This note works through the memory maths behind every experiment in this repo.
Every number here is **computed** from public model configs and hardware specs
(`python -m servebench kv-math` reproduces them). None of them is measured.
The measured numbers are in `results/gb10/REPORT.md`.

## 1. What the KV cache is

During decoding, each new token attends over the keys and values of every
earlier token. vLLM stores them instead of recomputing them. That stored
data is the KV cache. Per token it costs:

```
kv_bytes_per_token = 2 (K and V) x num_layers x num_kv_heads x head_dim x bytes_per_element
```

For **Qwen3-8B** (36 layers, 8 KV heads via GQA, head_dim 128):

| KV dtype | bytes/token | per 16-token block | 8k-token sequence | 32k-token sequence |
|---|---:|---:|---:|---:|
| BF16 (`auto`) | 147,456 B (144 KiB) | 2.25 MiB | 1.13 GiB | 4.5 GiB |
| FP8 (`--kv-cache-dtype fp8`) | 73,728 B (72 KiB) | 1.13 MiB | 0.56 GiB | 2.25 GiB |

GQA does most of the work here. With full multi-head attention (32 KV heads)
every row would be 4x larger.

## 2. How big the pool is on a DGX Spark

GB10 has **128 GB of unified LPDDR5x** shared by the CPU, the OS and the GPU,
with about **273 GB/s** of bandwidth. vLLM takes
`--gpu-memory-utilization x total` and subtracts weights and peak activations
(it measures these with a profiling forward pass at startup). The rest becomes
the KV pool, carved into 16-token **blocks** (PagedAttention).

Using `--gpu-memory-utilization 0.65` and about 6 GB for activations and CUDA graphs:

| config | weights | KV pool | tokens in pool | 32k seqs that fit | 2k seqs that fit |
|---|---:|---:|---:|---:|---:|
| BF16 weights + BF16 KV | 16.4 GB | ~61 GB | ~412k | 12 | 201 |
| FP8 weights + FP8 KV | 8.2 GB | ~69 GB | ~936k | 28 | 457 |

FP8 therefore wins on capacity twice: the weights get smaller, so the pool
gets bigger, *and* each token takes half the bytes. Together that is **2.3x
more cached tokens**.

> **Unified-memory caveat.** On a discrete GPU, `gpu-memory-utilization 0.9` is
> normal. On Spark the same memory also holds the OS, page cache, Docker and
> the benchmark client. Setting it too high causes host swapping or OOM-kills,
> not a clean CUDA OOM. 0.6-0.7 is a safe starting point. Check that
> `free -g` still has headroom while the server is under load.

## 3. Context limit (`--max-model-len`) is an admission control, not a capacity knob

People often assume that lowering `--max-model-len` "frees KV memory". In
vLLM v1 the **pool size is set by memory**, not by the context limit. Blocks
are allocated on demand, so a 500-token request uses 32 blocks whether the
limit is 8k or 128k. What the limit actually does:

* **Rejects requests** whose `prompt + max_tokens` exceeds it, with HTTP 400.
  That is a correctness and SLO decision: one 100k-token request takes the KV
  space of about 50 normal requests and holds it for the whole decode.
* **Startup check:** vLLM refuses to start if the pool can't hold even one
  max-length sequence. On Spark with BF16 at 128k this is tight: 18 GiB for
  one sequence.
* It has a small effect on profiling and activation sizing.

Experiment `04_context_limit` tests exactly this. The expected result is that
short-prompt workloads don't change, while the 12k-token `long_context`
workload is rejected entirely (error rate 1.0). The report records that as an
admission policy, not a speed-up. To fit more concurrent long contexts, the
real levers are FP8 KV (`02`) and `--gpu-memory-utilization`/`--kv-cache-memory-bytes`.

## 4. When the pool runs out: preemption

The scheduler admits requests while free blocks exist. Each decode step can
need a new block. When none is free, vLLM **preempts** the newest running
request: it frees that request's blocks and later recomputes its prefill. You
can see this as:

* `vllm:num_preemptions_total` rising (the harness reports the delta for each level)
* `vllm:kv_cache_usage_perc` pinned near 1.0 in `*.timeseries.jsonl`
* TTFT/E2E p99 blowing up while throughput plateaus

`long_context` at concurrency 8 is the workload built to approach this. 8 x
12k tokens is about 96k tokens, roughly a quarter of the BF16 pool. Add
`rag_shared_prefix` at concurrency 32 to raise the pressure further.

## 5. Prefix caching

With `--enable-prefix-caching`, each *full* 16-token block is identified by a
hash chained from all the blocks before it. A new request that starts with the
same tokens reuses those blocks and skips their prefill.

* Only **full, leading** blocks match. A one-token difference at position 0
  (a timestamp in the system prompt, say) invalidates *everything* after it.
  Put dynamic content **last**.
* The last prompt token is always recomputed (logits are needed), so a 100%
  hit is impossible.
* When a request finishes, its blocks stay in the cache as *evictable*. They
  are reclaimed LRU-first only when the pool needs room. Caching therefore
  costs no capacity.
* Metrics: `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`
  count **tokens**. The harness reports `hits / queries` for each run, and
  per-request `cached_tokens` via `--enable-prompt-tokens-details`.

### Why agents are the best case

An agent step's prompt is the previous step's prompt plus its output plus the
tool result. Without caching, step *k* re-prefills the whole conversation, so
total prefill across *n* steps grows as **O(n²)**. With caching each step
prefills only its new tokens, so the growth is **O(n)**. In the `replay` agent
(2k-token system prompt, 6 steps, about 360 new tokens per step) the last step
holds about 4k tokens, and about 90% of them should be served from cache. The
per-step TTFT chart in the report shows this directly. TTFT rises with the step
number when caching is off and stays flat when it is on.

## 6. Chunked prefill and the step token budget

`--max-num-batched-tokens` caps how many tokens (prefill + decode) one engine
step processes. Long prompts are split into chunks, and the chunks share steps
with ongoing decodes.

* **Small budget (2k, baseline):** decodes are never stalled for long, so ITL
  stays smooth. A 12k prompt needs at least 6 steps, which raises TTFT under load.
* **Large budget (8k, `03_batching`):** fewer, bigger prefill steps give better
  TTFT and prefill throughput. Decodes that share a step with a big prefill
  chunk see an **ITL spike**, so ITL p99 is the number to watch.

`--max-num-seqs` caps concurrent sequences. Raising it only helps if the KV
pool can actually hold that many contexts (see §2).

## 7. The decode roofline on GB10

At small batch, decode is **memory-bandwidth bound**: each step streams every
weight once.

```
max tokens/s per sequence (batch 1) ~ bandwidth / weight_bytes
  BF16: 273 GB/s / 16.4 GB ~ 16.7 tok/s  (~60 ms TPOT floor)
  FP8 : 273 GB/s /  8.2 GB ~ 33.3 tok/s  (~30 ms TPOT floor)
```

Measured TPOT will be somewhat higher because achievable bandwidth is below
the peak. That makes **FP8 weights the single biggest latency lever on
Spark**, much more than on an HBM GPU. Batching amortises the weight read:
at batch 32 the same weight pass serves 32 tokens. However, KV reads grow with
`batch x context`, and that is where FP8 KV pays off a second time.

## 8. Hypotheses to check against the GB10 run

| experiment | expected direction | metric to check |
|---|---|---|
| `01_prefix_caching` | RAG and agent TTFT drop sharply; `chat` unchanged | prefix hit rate, per-step agent TTFT |
| `02_fp8_quant` | TPOT about 1.6-1.9x lower at concurrency 1; more capacity | TPOT p50, peak KV usage, react task success (quality) |
| `03_batching` | better TTFT for long prompts under load; worse ITL p99 | TTFT p50 on `long_context`, ITL p99 |
| `04_context_limit` | same speed; >8k requests rejected | error rate on `long_context` |
| `05_combined` | best goodput overall | goodput at highest concurrency |

If a measurement contradicts a hypothesis, that is the interesting part.
Record it in the results write-up rather than tuning the hypothesis to fit.
