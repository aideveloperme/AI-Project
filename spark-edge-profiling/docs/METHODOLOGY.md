# Methodology and theory

## 1. Roofline model

For a kernel (here: a layer) with `F` floating-point operations that moves `Q` bytes
between DRAM and the chip, the **arithmetic intensity** is `AI = F / Q` [FLOP/B]. On a
machine with peak compute `P` [FLOP/s] and bandwidth `B` [B/s] the best possible
performance is

```
attainable(AI) = min(P, AI × B)          ridge point  AI* = P / B
```

* `AI < AI*` → **memory-bound**: time ≥ Q / B. Only moving fewer bytes helps (fusion,
  smaller activations/weights, better reuse via tiling/layout, larger batch for weight reuse).
* `AI > AI*` → **compute-bound**: time ≥ F / P. Faster math (FP16→INT8/FP8 tensor cores)
  or fewer FLOPs (pruning, lower resolution) helps.
* Distance below the roof = inefficiency (launch overhead, tail effects, poor kernels).

GB10 has a relatively low `B` (~273 GB/s nominal LPDDR5x) next to a large tensor-core `P`, so
`AI*` is high (hundreds of FLOP/B for FP16/INT8). Many CNN layers — every elementwise op,
BN, pooling, depthwise and many 1×1 convs, all layers at batch 1 — fall left of the ridge.
That is why fusion and smaller data types matter more on this platform than raw TFLOPs.

### Per-layer costs used (`sparkprof/roofline.py`)

| layer | FLOPs | compulsory bytes |
|---|---|---|
| Conv2d | `2·N·Cout·Hout·Wout·(Cin/g)·kh·kw` (+bias) | input + weight + output |
| Linear | `2·tokens·in·out` | input + weight + output |
| Attention (SDPA) | `2·N·D·3I + 4·B·H·N²·hd + 5·B·H·N² + 2·N·I·D` | x, weights, out, q/k/v/context (N×N stays on-chip with flash attention) |
| BN (eval) | `2·elements` | in + out + 4·C params |
| activations | ReLU 1, ReLU6 2, SiLU ≈5, GELU ≈8 per element | in + out |

"Compulsory" is a lower bound: real traffic is higher when tiles don't fit on-chip and
lower when L2 keeps data from the previous layer. `08_ncu_layers.py` measures the real
number (`dram__bytes_read.sum + dram__bytes_write.sum`); the **traffic ratio**
(measured / compulsory) is the bandwidth-efficiency figure to discuss per layer.

Ceilings come from `01_microbench_peaks.py` (measured), with the nominal datasheet roof
dashed for comparison. Always state which one a conclusion is based on.

## 2. Precision

| Format | Where it helps | Risk |
|---|---|---|
| FP32 (TF32 off) | reference accuracy | slowest; CUDA cores only |
| FP16 / BF16 | 2× fewer bytes, tensor cores | FP16 overflow in rare layers (BF16 avoids) |
| INT8 PTQ | 4× fewer weight bytes vs FP32, INT8 tensor cores | calibration error, outliers (SiLU, first/last layers) |
| INT8 QAT | same speed as PTQ | needs training data + a few epochs; recovers PTQ loss |
| Mixed INT8/FP16 | keeps most of INT8 speed | choose the FP16 layers well: here by isolated-layer KL sensitivity |
| FP8 (Blackwell) | INT8-like speed with float dynamic range | needs newer TensorRT/ModelOpt |

**Explicit quantisation**: ModelOpt inserts QuantizeLinear/DequantizeLinear around
Conv/Linear inputs and weights (per-channel weights, per-tensor activations, max
calibration). TensorRT fuses each Q/DQ pair into the adjacent kernel, so the ONNX file
defines exactly which layers run INT8 — making PTQ, QAT and mixed precision comparable.

Power and energy: NVML GPU power sampled at 20 Hz during sustained max-batch inference.
`energy/inference = avg W / throughput`. Lower precision usually lowers mJ/image even if W
rises slightly, because throughput rises more.

Memory: weights at each precision (params × bytes), engine file size, TensorRT activation
(device) memory, and PyTorch allocator peak for eager runs.

## 3. Conv + BN + ReLU fusion

BatchNorm in eval mode is an affine map, so it folds into the convolution:

```
s = γ / sqrt(σ² + ε)      W' = W · s      b' = (b − μ) · s + β
```

ReLU then runs in the convolution **epilogue** (on the accumulator in registers) instead of
as a separate kernel. Unfused, each unit writes and re-reads the full activation tensor
twice more (after conv and after BN): for a `N×C×H×W` tensor at FP16 that is
`4·N·C·H·W` extra bytes and two extra kernel launches. Expected per-unit savings are
therefore largest for **large activations with cheap convs** (early layers, 1×1 convs) —
exactly the memory-bound points on the roofline. The experiment reports per-unit time
(conv[+bn][+relu]) for each level so the saving can be attributed layer by layer, and the
TensorRT per-layer profile shows the same fusions done automatically (layer names joined
by ` + `).

## 4. Memory layout and tiling

* **NCHW** stores each channel plane contiguously; **NHWC** (channels-last) stores all
  channels of a pixel contiguously. Tensor-core implicit-GEMM convolutions reduce over
  `C·kh·kw`, so NHWC gives contiguous, vectorisable loads along the reduction dimension.
  With NCHW, cuDNN often inserts NCHW↔NHWC transposes — pure memory traffic.
* TensorRT picks internal formats itself (`HWC8`, `CHW32` for INT8 …). Forcing the network
  input to `HWC8` removes the input reformat layer; the engine inspector shows which
  formats were chosen.
* **Tiling**: a GEMM/conv computed in `BM×BN` output tiles re-reads A `N/BN` times and B
  `M/BM` times, so `traffic ≈ elem·(MNK/BN + MNK/BM + MN)` and `AI ≈ 2/(elem·(1/BM+1/BN))`.
  Bigger tiles → higher AI → the kernel moves right on the roofline, until registers/shared
  memory overflow (spills, lower occupancy). The Triton sweep plots this directly; Nsight
  Compute shows how much L2 reduces the traffic below the model. TensorRT ≥ 10.8's
  `tiling_optimization_level` applies L2-aware tiling across layers on Blackwell.

**Identifying memory-bound layers**: AI below the ridge **and** achieved GB/s close to the
ceiling (or ncu `dram__throughput` % of peak high while SM % is low). A layer below the
ridge but far below the bandwidth roof is *latency/launch-bound* instead — fix with
batching, CUDA graphs or fusion, not with bandwidth.

## 5. Model surgery

* **SiLU → ReLU6**: ReLU6 is a clamp that fuses into any conv epilogue and bounds the
  activation range to [0, 6], which makes INT8 calibration nearly lossless. SiLU needs a
  sigmoid and has an unbounded positive range. Swapping without retraining drops accuracy;
  a short fine-tune recovers most of it. Measure both FP16 and INT8: the FP16 gain is often
  small (TensorRT already fuses SiLU as a pointwise epilogue), the INT8 accuracy/latency
  gain is the main win.
* **Input resolution**: FLOPs and activation bytes scale with `H·W` (≈ r²); weights don't.
  Latency usually scales sub-quadratically at small sizes because fixed costs and weight
  reads dominate. Report the accuracy/latency Pareto front.
* **Attention-head pruning**: head importance `I_h = Σ |∂L/∂g_h|` with a gate `g_h` on each
  head's output (Michel et al., 2019). The top-k heads per layer are kept and qkv/proj
  weights are physically sliced, so the result is a smaller dense model (no masks) that any
  accelerator runs faster. Uniform k per layer keeps shapes regular for TensorRT kernels.
  FLOPs drop in qkv/proj and attention, not in the MLP — so expect speedups smaller than the
  head ratio; the roofline shows the MLP becoming the dominant cost.

## 6. Statistics and hygiene

* Warm-up iterations (default 50) before timing; 300 timed iterations; report mean, p50,
  p90, p99, std. Use p99 for latency SLAs, mean for throughput.
* Per-layer timings use CUDA events recorded in forward pre/post hooks (async; one sync
  per forward). They sum to slightly more than whole-model time (event overhead), so compare
  per-layer numbers with each other, and whole-model numbers with each other.
* Accuracy deltas below ~0.3 pt on Imagenette's 3,925 validation images are within noise.
