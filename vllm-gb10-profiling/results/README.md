# Results

| directory | what | source |
|---|---|---|
| [`gb10/`](gb10/) | **Real measurements** on an NVIDIA DGX Spark (GB10) | `make gb10` on the device |
| [`simulated/`](simulated/) | Harness validation on the GB10 *simulator*. **Not measurements.** | `make sim` (any machine, no GPU) |

Layout of each experiment directory:

```
<experiment>/
  experiment.resolved.yaml      config after `extends:` merge (exactly what ran)
  env.json                      vLLM version, server command, nvidia-smi, git SHA, timestamp
  server.log                    full server stdout/stderr
  summary.json                  every metric for every workload level + agent + profile
  loadtest/<workload>/
    concurrency_<N>.requests.jsonl    one row per request (TTFT, ITL list, tokens, cached tokens, errors)
    concurrency_<N>.timeseries.jsonl  1 Hz samples of KV usage / running / waiting
  agent/<mode>_s<N>/
    steps.jsonl                 one row per LLM call in the agent loop
    episodes.jsonl              one row per task
  profile/
    traces/*.pt.trace.json.gz   torch-profiler trace -> https://ui.perfetto.dev
    kernel_summary.md           GPU time by kernel category, top kernels
REPORT.md                       before/after tables vs 00_baseline
plots/*.png
```
