# Profiling

Two complementary captures are taken for each experiment.

## 1. PyTorch profiler (automatic, every experiment)

The harness starts vLLM with

```
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles",
                    "torch_profiler_with_stack":false,"ignore_frontend":true,
                    "delay_iterations":20,"max_iterations":60}'
```

(`ProfilerConfig` in vLLM 0.31). After the load tests it calls `POST /start_profile`,
runs the `chat` workload at concurrency 8, then calls `POST /stop_profile`.
The engine skips 20 iterations and records the next 60. That keeps traces to a
few MB while covering steady-state mixed prefill and decode.

Outputs for each experiment:

* `profile/traces/*.pt.trace.json.gz`: Chrome-trace format. Open it at
  https://ui.perfetto.dev (drag and drop).
* `profile/kernel_summary.md`: GPU time by category (gemm / attention / norm_act /
  sampling / communication / other), top kernels, and **GPU busy fraction**.

### Shareable profile links

Once the results are pushed:

```bash
python -m servebench links results/gb10 --repo <owner>/<repo> --ref <commit-sha>
```

This prints a `https://ui.perfetto.dev/#!/?url=<raw.githubusercontent.com URL>` link
for every trace. Use a commit SHA as `--ref` so the link never changes. These
are the "profile links" to share. A good pair is `00_baseline` against
`05_combined`.

## 2. Nsight Systems (on demand)

```bash
make nsys      # baseline + combined
```

The server runs under `nsys profile --capture-range=cudaProfilerApi` with
`--profiler-config '{"profiler":"cuda"}'`, so `/start_profile` and `/stop_profile`
map to `cudaProfilerStart/Stop` and only the load window is captured.
`nsys stats --report cuda_gpu_kern_sum <file>.nsys-rep` gives the kernel table.
The image must contain `nsys`. NGC images do; on others, install the
`nsight-systems-cli` package.

## How to read the traces

| what you see | what it means | lever |
|---|---|---|
| GEMM time dominates decode steps; per-step time ~ weight bytes / bandwidth | bandwidth-bound decode (expected on GB10) | FP8/FP4 weights, larger batch |
| attention share grows with context length and batch | KV reads dominate | FP8 KV cache, shorter contexts, prefix caching |
| GPU busy fraction well below 1, gaps between kernels | CPU/launch overhead (scheduler, sampling, detokenise) | CUDA graphs (on by default), async scheduling, bigger batch |
| long single steps with huge GEMMs while decodes wait | prefill chunk interfering with decode | lower `--max-num-batched-tokens` (better ITL) or raise it (better TTFT) |
| `reshape_and_cache` kernels | KV writes; proportional to new tokens | prefix caching removes them for cached prefixes |

Compare the category table of `00_baseline` with `02_fp8_quant`. The GEMM
kernels should change from BF16 to FP8 `scaled_mm`/CUTLASS kernels, and their
total time should drop by about the weight-byte ratio when decode-bound.
