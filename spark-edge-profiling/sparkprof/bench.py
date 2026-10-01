"""High-level benchmark helpers shared by the experiment scripts.

Every variant (eager PyTorch or TensorRT engine) is reported with the same schema:
  accuracy  top-1 on the Imagenette validation split
  latency   {batch: {mean_ms, p50_ms, p99_ms, throughput_ips, ...}}
  power     sustained-load average / peak W, energy per inference (mJ)
  memory    weights MB, engine/device MB, peak allocator MB
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from .power import measure_energy
from .roofline import annotate_costs
from .timing import LayerTimer, time_callable

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def weights_mb(model: nn.Module, bytes_per_param: float | None = None) -> float:
    if bytes_per_param is None:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return sum(p.numel() for p in model.parameters()) * bytes_per_param / 1e6


def eager_model(model: nn.Module, precision: str = "fp32", channels_last: bool = False) -> nn.Module:
    m = copy.deepcopy(model).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(DTYPES[precision])
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def make_input(batch: int, res: int, precision: str = "fp32", channels_last: bool = False,
               device: str | None = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch, 3, res, res, device=device, dtype=DTYPES.get(precision, torch.float32))
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def bench_eager(model: nn.Module, cfg, res: int, precision: str = "fp32",
                channels_last: bool = False, batch_sizes=None, power: bool = False) -> dict:
    """Latency (and optionally power) of an eager model; disables TF32 for honest FP32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    m = eager_model(model, precision, channels_last)
    out = {"latency": {}, "memory": {"weights_mb": weights_mb(m)}}
    for b in batch_sizes or cfg.bench.batch_sizes:
        x = make_input(b, res, precision, channels_last)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out["latency"][b] = time_callable(lambda: m(x), cfg.bench.warmup, cfg.bench.iters, b)
        if torch.cuda.is_available():
            out["memory"][f"peak_alloc_mb_b{b}"] = torch.cuda.max_memory_allocated() / 1e6
    if power and torch.cuda.is_available():
        b = max(cfg.bench.batch_sizes)
        x = make_input(b, res, precision, channels_last)
        with torch.inference_mode():
            out["power"] = measure_energy(lambda: m(x), b, cfg.bench.power_seconds,
                                          cfg.bench.power_sample_hz)
    return out


def bench_engine(engine_path: str | Path, cfg, res: int, val_loader=None, power: bool = True,
                 batch_sizes=None) -> dict:
    from .trt_engine import TRTRunner, evaluate_engine, precision_histogram

    r = TRTRunner(engine_path)
    out = {"latency": {}, "memory": {"engine_mb": Path(engine_path).stat().st_size / 1e6,
                                     "trt_device_mb": r.device_memory_bytes() / 1e6}}
    if val_loader is not None and r.linear_input:
        out["accuracy"] = evaluate_engine(r, val_loader, cfg.trt.max_batch)
    for b in batch_sizes or cfg.bench.batch_sizes:
        out["latency"][b] = r.time(b, res, cfg.bench.warmup, cfg.bench.iters)
    if power:
        b = max(batch_sizes or cfg.bench.batch_sizes)
        r.setup(b, res)
        out["power"] = measure_energy(r.enqueue, b, cfg.bench.power_seconds,
                                      cfg.bench.power_sample_hz, sync=r.sync)
    try:
        info = r.layer_info()
        out["engine_layers"] = len(info)
        out["precision_hist"] = precision_histogram(info)
    except Exception as e:  # noqa: BLE001
        out["inspector_error"] = str(e)[:200]
    return out


def per_layer_profile(model: nn.Module, res: int, batch: int, precision: str = "fp32",
                      channels_last: bool = False, warmup: int = 10, iters: int = 50) -> list[dict]:
    m = eager_model(model, precision, channels_last)
    x = make_input(batch, res, precision, channels_last)
    with LayerTimer(m) as lt:
        rows = lt.run(x, warmup, iters)
    return annotate_costs(m, rows)


def model_cost(model: nn.Module, res: int) -> dict:
    """Total analytical GFLOPs / MB for one image (runs one CPU-or-GPU forward)."""
    rows = per_layer_profile(model, res, 1, "fp32", warmup=0, iters=1)
    return {"gflops": sum(r.get("flops", 0) for r in rows) / 1e9,
            "min_traffic_mb": sum(r.get("bytes", 0) for r in rows) / 1e6,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}


def count_cuda_kernels(model: nn.Module, x: torch.Tensor) -> int | None:
    if not torch.cuda.is_available():
        return None
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(x)
            torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")
