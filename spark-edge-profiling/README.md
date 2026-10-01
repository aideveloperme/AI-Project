# Inference Profiling & Optimisation on NVIDIA DGX Spark

An end-to-end, reproducible study of how precision, operator fusion, memory layout,
tiling and model surgery change **accuracy, latency, power and memory** for vision
models on the **NVIDIA DGX Spark** (GB10 Grace Blackwell superchip, 128 GB unified
LPDDR5x). It ends with a profiling report built around **roofline analysis**.

| Question | Experiment | Script |
|---|---|---|
| What can this GPU actually do? | measured DRAM bandwidth + GEMM peaks per precision (roofline ceilings) | `01_microbench_peaks.py` |
| FP32 vs FP16 vs INT8 PTQ vs INT8 QAT vs mixed precision | TensorRT engines + PyTorch references: top-1, p50/p99 latency, throughput, W, mJ/image, weights/engine/activation MB | `03_precision_sweep.py` |
| What does Conv+BN+ReLU fusion buy? | 4 fusion levels, **per-layer timing before/after**, CUDA kernel counts, numerics, TensorRT fused-layer names | `04_fusion_experiment.py` |
| NCHW vs NHWC, tiling, bandwidth bottlenecks | eager layout A/B per layer, TensorRT I/O format + tiling levels, Triton tile sweep, memory-bound layer list | `05_layout_bandwidth.py` |
| Real DDR traffic per layer | Nsight Compute `dram__bytes_*` vs compulsory bytes | `08_ncu_layers.py` |
| Accelerator-oriented model surgery | SiLU→ReLU6, input-resolution sweep, ViT attention-head pruning, each re-timed in TensorRT | `06_model_surgery.py` |
| Where is the time going, and why? | per-layer roofline (FLOPs, bytes, AI, attained TFLOP/s, bound) | `07_roofline.py` |
| Write it up | Markdown report + figures | `09_make_report.py` |

Models (configurable in `configs/default.yaml`): **ResNet-50** (Conv+BN+ReLU, precision,
layouts), **EfficientNet-B0** (SiLU, resolution), **ViT-B/16** (attention heads).
Dataset: **Imagenette** (10-class ImageNet subset, auto-downloaded, ~350 MB) — real images,
small enough to fine-tune every variant on the Spark in minutes.

## Quick start (on the DGX Spark)

```bash
cd spark-edge-profiling
./docker/run.sh                      # builds the NGC-based image once, opens a shell
./run_all.sh --quick                 # ~15-20 min smoke run of every step
./run_all.sh                         # full run (see docs/WORKLOAD.md for timings)
# -> reports/REPORT.md + reports/*.png, raw data in results/*.json
```

Run single steps with any config override:

```bash
python scripts/03_precision_sweep.py --model resnet50 --fp8 --set bench.iters=1000
python scripts/05_layout_bandwidth.py --parts BC          # only TensorRT layout/tiling + Triton tiles
python scripts/06_model_surgery.py --parts C              # only ViT head pruning
python scripts/08_ncu_layers.py --topk 25                 # needs `ncu` + counter permissions
```

`pytest -q tests` runs CPU-only correctness tests (fusion equivalence, head pruning
equivalence, FLOP counts) — no GPU needed.

## Layout

```
configs/default.yaml     every knob: models, data, benchmark iterations, TRT, nominal GB10 peaks
docker/                  Dockerfile (nvcr.io/nvidia/pytorch, arm64) + run.sh
docs/SETUP_DGX_SPARK.md  host prep, container, counters permission, measurement hygiene
docs/WORKLOAD.md         phase-by-phase plan of the whole project, outputs, time budget
docs/METHODOLOGY.md      the theory: roofline, fusion, layouts/tiling, quantisation, surgery
sparkprof/               library
  timing.py              CUDA-event latency + per-layer forward-hook timer
  power.py               NVML power sampler, energy per inference
  roofline.py            per-layer FLOP/byte cost model, classification, roofline plot
  fusion.py              BN folding, cuDNN conv+bias+ReLU fusion, per-unit mapping
  quant.py               ModelOpt INT8/FP8 PTQ, QAT, layer sensitivity, mixed precision
  trt_engine.py          ONNX export, TensorRT build (precision/format/tiling), runner + IProfiler
  surgery.py             activation swap, gradient-based head importance, head pruning
  tiling.py              Triton GEMM with sweepable tiles + traffic model
  ncu.py / ncu_target.py Nsight Compute driver and single-layer target
  hardware.py            device inventory, bandwidth / GEMM micro-benchmarks
scripts/00..09           the pipeline, one script per experiment
```

## Reading the results

* `results/*.json` hold every number (per-batch latency stats, per-layer rows, power traces summary).
* `reports/REPORT.md` is generated — rerun `09_make_report.py` after any step.
* The report's **Key findings** are derived from your measurements; the explanation of *why*
  each effect appears is in `docs/METHODOLOGY.md` and should be quoted/adapted in your write-up.
