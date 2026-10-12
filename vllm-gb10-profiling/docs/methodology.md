# Methodology

## Metrics (all client-side, streamed)

| metric | definition | why it matters |
|---|---|---|
| TTFT | request sent -> first content/tool-call token | queueing + prefill; what a user waits for |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)` | average decode speed per request |
| ITL | gap between consecutive streamed chunks | decode smoothness; p99 shows stalls from prefill interference |
| E2E | request sent -> stream closed | total latency |
| output tok/s | Σ output tokens / wall time of the level | throughput |
| goodput | requests/s that meet **both** TTFT ≤ 2 s and TPOT ≤ 150 ms | capacity you can actually sell; throughput keeps rising after users are already unhappy |

Token counts come from the server `usage` block (`stream_options.include_usage`),
not from counting chunks.

Server-side, from Prometheus `/metrics`, counter deltas are taken around each
level: prefix-cache queries/hits, preemptions, and prompt and generation tokens.
`kv_cache_usage_perc`, running and waiting are sampled once a second, and peaks
are reported.

## Load model

* **Closed loop (default):** N concurrent users, each sends its next request
  as soon as the previous one finishes. Sweep N over `levels`.
* **Open loop (`mode: rate`):** Poisson arrivals at R req/s, independent of
  completions. This exposes queueing collapse that closed loop hides. Use it to
  confirm a capacity number taken from the closed-loop sweep.

## Controls (things that silently ruin LLM benchmarks)

1. **Fixed output length:** `ignore_eos: true`. Otherwise a config that makes
   the model stop earlier looks "faster".
2. **No accidental cache hits:** every unique prompt starts with a unique tag,
   and **each sweep level gets fresh prompts** (`workloads.for_level`). A
   bug found during development: re-using one prompt set across levels let the
   prefix cache serve later levels from earlier ones, which inflated the
   cache-on results. A unit test now guards against it.
3. **Shared prefixes stay identical across levels**, and warmup requests use
   them, because a production system prompt is always warm.
4. **Warmup** requests before each level are excluded from measurement. Server
   startup time (weights load, `torch.compile`, CUDA graph capture) is recorded
   separately.
5. **Same workloads in every experiment:** a test enforces that experiment
   YAMLs differ from the baseline only in the knobs listed under `changes`.
6. **Qwen3 thinking disabled** (`chat_template_kwargs.enable_thinking=false`),
   so hidden reasoning tokens don't change the output shape.
7. **Isolation:** `run_gb10_suite.sh` warns if other GPU processes exist and
   records `nvidia-smi` clocks, power and temperature next to the results. The
   container image digest is recorded too.
8. **One server per experiment:** each config gets a clean process, so the
   cache state can't leak between experiments.

## Agent benchmark

* `replay` (used for before/after): 6 LLM calls per task. Each call adds 64
  output tokens and a 300-token canned tool result to a 2k-token system prompt.
  The shape is fixed, so model behaviour can't change the prompt lengths, and
  configs are compared like for like.
* `react`: real OpenAI-style tool calling (`--enable-auto-tool-choice
  --tool-call-parser hermes`) against four deterministic tools, on five tasks
  with known answers. It reports task latency **and** task success, so a
  quantisation-induced quality regression shows up in the same table as the
  latency win.
* Reported: task latency p50/p90/p99, TTFT p50 per step, TTFT share of LLM
  time, tool share of task time, episodes/min, and prefix hit rate. Tool
  latency can be simulated (`tool_latency_ms`) to show when the model stops
  being the bottleneck.

## Threats to validity

* Synthetic text is about 1 token per word, so input sizes are approximate.
  Actual prompt token counts are reported.
* One run per configuration: run-to-run noise on a quiet Spark is typically
  a few percent. Treat deltas under about 5% as noise, or re-run with a
  different `seed`.
* Thermal and power state: the Spark is a small-form-factor box. Long runs can
  throttle. Check the `nvidia-smi` power and clock log in `host_info.txt`.
* Client on the same host: the asyncio client is light, but it shares CPU and
  memory with the server. At very high concurrency, run the client from another
  machine (`--launcher external --base-url http://spark:8000`).
