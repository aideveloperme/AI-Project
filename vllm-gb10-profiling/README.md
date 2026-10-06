# vLLM on DGX Spark (GB10): load test, profile, tune, agent latency

A reproducible harness to **serve an open model with vLLM on an NVIDIA DGX Spark
(GB10)**, load-test it, profile it, change one or two serving knobs at a time
(prefix caching, FP8 quantisation, batching, context limit), and record
before/after results, including a **multi-step tool-calling agent loop** to
measure agent latency.

```
make gb10-setup && make gb10        # on the Spark: ~2-3 h, writes results/gb10/REPORT.md
make test && make sim               # anywhere: tests + full pipeline on a GB10 simulator
```

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` (BF16) and `Qwen/Qwen3-8B-FP8`, Apache-2.0, native tool calling |
| Server | vLLM 0.31 (`vllm serve`, OpenAI API), in Docker |
| Hardware | NVIDIA DGX Spark: GB10 Grace-Blackwell, 128 GB unified LPDDR5x (~273 GB/s) |
| Harness | `servebench`: async streaming client, closed/open-loop load gen, Prometheus scraping, agent loop, torch/nsys profiling, report generator |
| CI | lint, 32 unit + end-to-end tests, smoke run of every experiment on the simulator |

## Results status

| | status |
|---|---|
| **GB10 measurements** | **Pending a run on the device.** `make gb10` writes [`results/gb10/REPORT.md`](results/gb10/), with tables, plots and Perfetto profile links. |
| Simulated (harness validation) | [`results/simulated/REPORT.md`](results/simulated/REPORT.md): the whole pipeline end to end on an analytical GB10 model. **Not measurements.** It shows the direction of each knob and that every artefact gets produced. |

> No GB10 numbers appear anywhere in this repo until they have been measured on a
> GB10. The simulator exists so that the harness is tested and CI-able without a
> GPU. Its outputs carry a "SIMULATED" banner in every generated report.

## Experiments

Each experiment `extends: base.yaml` and changes only the knobs listed. A unit
test enforces this, so before/after comparisons stay fair.

| id | change vs baseline | question it answers |
|---|---|---|
| `00_baseline` | BF16 weights + KV, prefix caching **off**, 2k-token step budget, 32k context | reference |
| `01_prefix_caching` | `--enable-prefix-caching` | How much prefill do shared system prompts and agent histories save? |
| `02_fp8_quant` | `Qwen3-8B-FP8` + `--kv-cache-dtype fp8` | Bandwidth-bound decode: does halving the bytes halve TPOT? What happens to KV capacity and task quality? |
| `03_batching` | `--max-num-batched-tokens 8192`, `--max-num-seqs 256` | Chunked-prefill budget: TTFT against ITL-stall trade-off |
| `04_context_limit` | `--max-model-len 8192` | Is the context limit a capacity knob or an admission policy? |
| `05_combined` | 01 + 02 + bigger step budget | recommended config |

Workloads (identical in every experiment):

| workload | shape | stresses |
|---|---|---|
| `chat` | unique 512±25% in / 128 out, concurrency 1→64 | raw prefill + decode, batching |
| `rag_shared_prefix` | 4 × 2k-token shared contexts + 128 unique / 128 out | prefix cache |
| `long_context` | 12k in / 64 out, concurrency 1→8 | KV capacity, context limit, chunked prefill |
| agent `replay` | 2k system prompt, 6 steps × (64 out + 300-token tool result), 1 and 8 sessions | per-step TTFT growth, prefix reuse |
| agent `react` | real tool calls (4 tools, 5 tasks with known answers), 4 sessions | task latency **and** task success |

## What gets measured

* **Client:** TTFT, TPOT, ITL, E2E (p50/p90/p99), output tok/s, and **goodput** under
  an SLO (TTFT ≤ 2 s, TPOT ≤ 150 ms), with per-request raw rows in JSONL.
* **Server** (Prometheus): KV-cache usage (1 Hz), running/waiting, preemptions,
  prefix-cache hit rate, cached tokens per request.
* **Agent:** task latency p50/p90/p99, TTFT per step, TTFT share of LLM time,
  tool share of task time, episodes/min, task success.
* **Profiles:** a torch-profiler trace per experiment (60 steady-state engine
  iterations), with a summary of kernel categories and GPU busy fraction.
  Optional Nsight Systems capture. Shareable Perfetto links via
  `servebench links`.

See [docs/methodology.md](docs/methodology.md) for definitions and the controls
that keep results honest: fixed output lengths, per-level fresh prompts,
isolated servers and threats to validity.

## KV cache and context handling (the short version)

Full write-up: [docs/kv_cache_and_context.md](docs/kv_cache_and_context.md).

* Qwen3-8B KV = 2 × 36 layers × 8 KV heads × 128 × 2 B = **144 KiB/token** (BF16).
  A 32k context costs 4.5 GiB.
* On GB10 at `--gpu-memory-utilization 0.65`: about **412k KV tokens with BF16**
  against **936k with FP8 weights + FP8 KV**. FP8 shrinks the weights (bigger
  pool) *and* halves the bytes per token.
* Decode at batch 1 streams every weight once per token, so the roofline is
  273 GB/s ÷ 16.4 GB ≈ **17 tok/s (BF16)** against ≈ **33 tok/s (FP8)**. On
  Spark, FP8 is the biggest single-stream latency lever.
* `--max-model-len` doesn't resize the KV pool in vLLM v1. It is admission
  control (HTTP 400 above the limit). Capacity comes from memory utilisation
  and KV dtype.
* Prefix caching turns an agent's O(n²) re-prefill across steps into O(n). The
  agent per-step TTFT plot shows this directly.

```bash
python -m servebench kv-math --model Qwen/Qwen3-8B
python -m servebench kv-math --model Qwen/Qwen3-8B-FP8 --kv-cache-dtype fp8
```

## Running on the DGX Spark

```bash
git clone <this repo> && cd <repo>/vllm-gb10-profiling
make gb10-setup                       # pull image, download both models, check the GPU in-container
make gb10-quick                       # ~20-30 min sanity pass (1/8 of the requests)
make gb10                             # full suite, ~2-3 h, -> results/gb10/REPORT.md
make nsys                             # optional Nsight Systems traces (baseline vs combined)
git add results/gb10 && git commit -m "GB10 results" && git push
python -m servebench links results/gb10 --repo <owner>/<repo> --ref $(git rev-parse HEAD)
```

* Image: `VLLM_IMAGE=...` overrides the default `vllm/vllm-openai:v0.31.0`. If that
  image can't see the GB10 (setup checks this), use NVIDIA's NGC vLLM container
  for DGX Spark (`nvcr.io/nvidia/vllm:<yy.mm>-py3`).
* Unified memory: keep `gpu-memory-utilization` around 0.6–0.7. The OS and the
  client share the same 128 GB.
* Benchmarking a server you already started:
  `python -m servebench run configs/experiments/00_baseline.yaml --launcher external --base-url http://spark:8000 --out results/gb10`.

Ad-hoc tools:

```bash
python -m servebench loadtest --model qwen3-8b --kind shared_prefix --levels 1,8,32
python -m servebench agent --model qwen3-8b --mode react --sessions 4 --no-thinking
python -m servebench trace results/gb10/00_baseline/profile/traces/*.json.gz
python -m servebench report results/gb10 --baseline 00_baseline
```

## Repository layout

```
servebench/
  client.py        streaming OpenAI client: TTFT / ITL / TPOT, tool-call delta assembly
  workloads.py     deterministic chat / shared-prefix / long-context generators
  loadgen.py       closed-loop (concurrency) and open-loop (Poisson) runners + metric sampling
  prom.py          vLLM /metrics scraping (names verified against vLLM 0.31)
  metrics.py       percentiles, goodput, SLO attainment
  agent.py         agent loop: replay (fixed shape) and react (real tool calls)
  agent_tools.py   deterministic tools + tasks with known answers
  profiling.py     /start_profile, /stop_profile, trace -> kernel-category summary
  launcher.py      docker / local / mock / external server lifecycle
  experiment.py    YAML (with extends) -> full run -> results directory
  report.py        before/after REPORT.md + plots
  mock/            GB10 simulator: paged KV, prefix cache, chunked prefill, preemption, roofline timing
configs/experiments/   base.yaml + 00..05
scripts/               gb10_setup.sh, run_gb10_suite.sh, nsys_profile.sh
docs/                  methodology, KV cache & context, profiling
results/               gb10/ (device), simulated/ (harness validation)
tests/                 unit + end-to-end (HTTP/SSE against the simulator)
```

## Design notes

* **The harness is decoupled from the server.** Everything goes through the
  OpenAI API and `/metrics`, so the same harness benchmarks vLLM in Docker, a
  local install, a remote box, or the simulator.
* **The simulator models the mechanisms, not the numbers:** 16-token paged KV
  blocks, hash-chained prefix caching with LRU eviction, chunked prefill under
  a token budget, `max-num-seqs`, recompute preemption, `max-model-len`
  admission, and roofline step timing (bandwidth vs compute). That is enough
  for end-to-end CI and for sanity-checking the direction of each knob. It is
  not a substitute for the GB10 run.
* **Replay agent for comparisons, react agent for quality.** A quantised model
  that takes an extra tool step would otherwise look "slower" for the wrong
  reason. Replay fixes the shape; react catches quality regressions.
