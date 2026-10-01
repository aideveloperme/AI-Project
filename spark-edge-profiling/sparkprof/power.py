"""GPU power / energy measurement via NVML (falls back to nvidia-smi polling).

On DGX Spark the GB10 superchip reports GPU power through NVML. Some fields that
assume discrete GPUs (e.g. framebuffer memory used) report "Not Supported" because
CPU and GPU share one 128 GB LPDDR5x pool — that is expected, and the memory
numbers in this project come from the CUDA allocator / TensorRT instead.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Callable


class PowerSampler:
    def __init__(self, hz: float = 20.0, device_index: int = 0):
        self.period = 1.0 / hz
        self.idx = device_index
        self.samples: list[tuple[float, float]] = []  # (t, watts)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.backend = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)  # probe
            self.backend = "nvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.backend = "nvidia-smi"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def _read_watts(self) -> float | None:
        try:
            if self.backend == "nvml":
                return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            if self.backend == "nvidia-smi":
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.idx}", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None
        return None

    def _loop(self):
        while not self._stop.is_set():
            w = self._read_watts()
            if w is not None:
                self.samples.append((time.perf_counter(), w))
            time.sleep(self.period)

    def __enter__(self):
        self.samples.clear()
        self._stop.clear()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self) -> dict:
        if len(self.samples) < 2:
            return {"backend": self.backend, "avg_w": None, "peak_w": None, "energy_j": None}
        ts = [t for t, _ in self.samples]
        ws = [w for _, w in self.samples]
        energy = sum((ts[i + 1] - ts[i]) * (ws[i] + ws[i + 1]) / 2 for i in range(len(ts) - 1))
        return {"backend": self.backend, "avg_w": sum(ws) / len(ws), "peak_w": max(ws),
                "energy_j": energy, "duration_s": ts[-1] - ts[0], "n_samples": len(ws)}


def idle_power(seconds: float = 3.0, hz: float = 20.0) -> float | None:
    with PowerSampler(hz) as ps:
        time.sleep(seconds)
    return ps.summary()["avg_w"]


def measure_energy(fn: Callable[[], object], batch: int, seconds: float = 15.0,
                   hz: float = 20.0, sync: Callable[[], None] | None = None) -> dict:
    """Run `fn` back-to-back for `seconds` under a power sampler.

    Returns average/peak watts, inferences/s and energy per inference (mJ).
    Energy/inference uses the *total* board power reading; subtract `idle_w` if you
    want the dynamic component only (both are reported).
    """
    import torch
    sync = sync or (torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None))
    for _ in range(10):
        fn()
    sync()
    n = 0
    with PowerSampler(hz) as ps:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(10):
                fn()
            n += 10
            sync()
        elapsed = time.perf_counter() - t0
    s = ps.summary()
    ips = n * batch / elapsed
    s["throughput_ips"] = ips
    s["mj_per_inference"] = (s["avg_w"] / ips * 1e3) if s.get("avg_w") else None
    return s
