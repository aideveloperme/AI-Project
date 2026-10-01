"""Device inventory and micro-benchmarks that measure the real roofline ceilings.

Nominal datasheet peaks are rarely reachable; a roofline drawn with *measured*
ceilings tells you how far each layer is from what the GB10 can actually do.

  bandwidth   device-to-device copy and read-only reduction over buffers far larger
              than L2 (the GB10 shares 128 GB LPDDR5x between CPU and GPU)
  compute     large square GEMMs in FP32 (TF32 off), TF32, FP16, BF16, INT8, FP8
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import torch


def device_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "machine": platform.machine(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "sm_count": p.multi_processor_count,
                     "compute_capability": f"{p.major}.{p.minor}",
                     "total_memory_gb": p.total_memory / 1e9, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(),
                     "l2_cache_mb": getattr(p, "L2_cache_size", 0) / 1e6})
    for mod in ("tensorrt", "modelopt", "triton", "onnx", "torchvision"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                            "--format=csv,noheader"], capture_output=True, text=True)
        info["nvidia_smi"] = q.stdout.strip()
    return info


def _time_ms(fn, iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def measure_bandwidth(size_mb: int = 2048) -> dict:
    n = size_mb * (1 << 20) // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda").normal_()
    b = torch.empty_like(a)
    copy_ms = _time_ms(lambda: b.copy_(a))
    read_ms = _time_ms(lambda: a.sum())
    triad_ms = _time_ms(lambda: torch.add(a, b, alpha=2.0, out=b))
    nbytes = a.numel() * 4
    res = {"copy_gbs": 2 * nbytes / copy_ms / 1e6, "read_gbs": nbytes / read_ms / 1e6,
           "triad_gbs": 3 * nbytes / triad_ms / 1e6, "buffer_mb": size_mb}
    res["best_gbs"] = max(res["copy_gbs"], res["read_gbs"], res["triad_gbs"])
    del a, b
    torch.cuda.empty_cache()
    return res


def measure_gemm(n: int = 8192) -> dict:
    res = {}
    flops = 2.0 * n ** 3
    prev = torch.backends.cuda.matmul.allow_tf32
    for name, dt, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True),
                           ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        res[f"{name}_tflops"] = flops / _time_ms(lambda: a @ b, iters=10) / 1e9
    torch.backends.cuda.matmul.allow_tf32 = prev
    try:
        a = torch.randint(-64, 64, (n, n), device="cuda", dtype=torch.int8)
        b = torch.randint(-64, 64, (n, n), device="cuda", dtype=torch.int8).t()
        res["int8_tops"] = flops / _time_ms(lambda: torch._int_mm(a, b), iters=10) / 1e9
    except Exception as e:  # noqa: BLE001
        res["int8_tops"] = None
        res["int8_error"] = str(e)[:200]
    try:
        a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()
        one = torch.tensor(1.0, device="cuda")
        res["fp8_tflops"] = flops / _time_ms(
            lambda: torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16), iters=10) / 1e9
    except Exception as e:  # noqa: BLE001
        res["fp8_tflops"] = None
        res["fp8_error"] = str(e)[:200]
    torch.cuda.empty_cache()
    return res


def peaks(cfg, measured: dict | None, precision: str) -> tuple[float, float]:
    """(peak TFLOP/s, peak GB/s) for the roofline: measured if available, else nominal."""
    nom = cfg.hardware.nominal
    key = {"fp32": "fp32_tflops", "tf32": "tf32_tflops", "fp16": "fp16_tflops",
           "bf16": "bf16_tflops", "int8": "int8_tops", "fp8": "fp8_tflops"}[precision]
    tf = (measured or {}).get("gemm", {}).get(key) or nom.get(key) or nom.fp16_tflops
    bw = (measured or {}).get("bandwidth", {}).get("best_gbs") or nom.dram_bw_gbs
    return float(tf), float(bw)
