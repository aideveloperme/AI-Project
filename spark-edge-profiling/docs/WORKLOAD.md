# End-to-end workload plan

The project runs as ten ordered steps. Each one writes JSON to `results/`; later steps read
earlier results, and `09_make_report.py` can be re-run at any point. Time estimates are
rough guides for a full (non `--quick`) run on a DGX Spark and are dominated by
TensorRT engine builds and fine-tuning; measure your own and note them in the report.

```
00 env ──► 01 ceilings ──► 02 fine-tune ──┬─► 03 precision ─────┐
                                          ├─► 04 fusion ────────┤
                                          ├─► 05 layout/BW ─────┤
                                          ├─► 06 surgery ───────┼─► 09 report
                                          └─► 07 roofline ─► 08 ncu ─► 07 (again)
```

## Phase 0 — Environment (`00_env_check.py`, < 1 min)
Records GPU, CUDA/cuDNN/TensorRT/ModelOpt/Triton versions, tool availability (ncu, nsys,
trtexec), NVML power backend and idle power. **Deliverable**: `results/env.json` (the
"platform" table of the report). Fix every `[MISSING]` line before continuing.

## Phase 1 — Roofline ceilings (`01_microbench_peaks.py`, ~2 min)
* DRAM: device copy, read-only reduction, triad over 2 GB buffers → achievable GB/s.
* Compute: 8192³ GEMMs in FP32 (TF32 off), TF32, FP16, BF16, INT8 (`torch._int_mm`), FP8 (`torch._scaled_mm`).
* Prints ridge points (FLOP/B) per precision.
**Deliverable**: `results/hw_peaks.json`. Expect measured bandwidth below the 273 GB/s
nominal; the gap is the first thing the report states.

## Phase 2 — Baselines (`02_train_baselines.py`, ~30–60 min)
ImageNet-pretrained ResNet-50, EfficientNet-B0, ViT-B/16 fine-tuned on Imagenette
(5 epochs, AdamW, BF16 autocast). **Deliverable**: `checkpoints/<arch>.pt` and FP32 accuracy —
the reference every other accuracy is compared with.

## Phase 3 — Precision study (`03_precision_sweep.py`, ~30–45 min)
For ResNet-50 (repeat with `--model efficientnet_b0` / `vit_b_16` if time allows):

| Variant | How |
|---|---|
| eager FP32 / FP16 | PyTorch + cuDNN, TF32 off |
| TRT FP32 / FP16 (/ BF16) | ONNX → TensorRT, builder precision flags |
| TRT INT8 PTQ | ModelOpt max calibration on 512 train images, explicit Q/DQ |
| TRT INT8 QAT | PTQ init + 2 epochs fake-quant fine-tune, same export |
| TRT mixed | INT8 except the k most sensitive layers (KL-divergence ranking) in FP16 |
| TRT FP8 (`--fp8`) | Blackwell E4M3 |

Measured for each: top-1, latency p50/p90/p99 at batch 1/8/32, throughput, average/peak
W and mJ/image under sustained load, weights MB, engine MB, activation/device MB, engine
layer-precision histogram. **Deliverable**: `results/precision_resnet50.json`, report §2.

## Phase 4 — Layer fusion (`04_fusion_experiment.py`, ~10 min)
L0 unfused → L1 BN folded → L2 cuDNN Conv+bias+ReLU → L3 `torch.compile` → TensorRT.
Per-layer timing for every fusion unit (same key before and after), CUDA kernels per
forward, max logit difference (numerics). **Deliverable**: `results/fusion_resnet50.json`,
before/after bar chart, report §3.

## Phase 5 — Layout, tiling, bandwidth (`05_layout_bandwidth.py`, ~20 min)
A. NCHW vs NHWC per layer in FP32/FP16. B. TensorRT input LINEAR vs HWC8 and tiling levels
NONE→FULL. C. Triton GEMM tile sweep (16×16 … 256×128) with traffic model. D. Per-layer
achieved GB/s vs ceiling, memory-bound list. **Deliverable**: `results/layout_resnet50.json`,
bandwidth chart, report §4.

## Phase 6 — Model surgery (`06_model_surgery.py`, ~45–90 min)
A. EfficientNet-B0 SiLU→ReLU6: accuracy before/after fine-tune, FP16 and INT8 TRT latency,
INT8 PTQ accuracy for both. B. Resolution 128…256: accuracy, GFLOPs, latency. C. ViT-B/16
head pruning at 100/75/50 % heads: importance heatmap, accuracy before/after recovery,
params, GFLOPs, latency. **Deliverable**: `results/surgery.json`, report §5.

## Phase 7–8 — Roofline + measured DDR traffic (`07_roofline.py`, `08_ncu_layers.py`, ~20–40 min)
Per-layer FLOPs, compulsory bytes, AI, attained TFLOP/s, achieved GB/s, bound; then
Nsight Compute on the 15 slowest layers per model (+ Triton tiles) for real
`dram__bytes_read/write`, L2 bytes, %-of-peak DRAM/SM. Re-running 07 overlays the measured
points. **Deliverable**: roofline plots per model/precision, report §6.

## Phase 9 — Report (`09_make_report.py`, seconds)
`reports/REPORT.md` with all tables, figures and auto-derived findings. Finish it by hand:

1. **Executive summary**: the recommended deployment config (e.g. "INT8 QAT, NHWC, 192 px")
   and its accuracy/latency/power vs the FP32 baseline.
2. **Roofline narrative**: which layers are memory-bound on GB10 and why (low reuse
   depthwise/1×1 convs, elementwise ops, small batch), which are compute-bound, and which
   optimisation moved each class of layer (fusion and lower-precision activations move
   memory-bound points right/up; INT8/FP8 lifts the compute roof).
3. **Trade-off discussion**: accuracy cost of each optimisation vs its gain; where gains
   did *not* materialise and why (e.g. INT8 at batch 1 is launch-bound; SiLU is already
   fused by TensorRT so ReLU6 gains mostly via INT8 accuracy).
4. **Threats to validity**: thermals, run-to-run variance, Imagenette vs full ImageNet,
   compulsory-bytes approximation where ncu was not run.

## Suggested extensions
* Full ImageNet validation (set `data.dataset` to a custom ImageFolder loader).
* NVFP4 / FP8 for the ViT with ModelOpt (Blackwell-specific).
* 2:4 structured sparsity (`config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)`) after ModelOpt sparsification.
* CUDA Graphs for batch-1 eager latency (removes launch overhead visible in the roofline).
* Detection model (YOLO-style) for the "prune heads" meaning *detection heads*.
