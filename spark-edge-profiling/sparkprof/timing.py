"""Latency measurement: whole-model (CUDA events) and per-layer (forward hooks)."""

from __future__ import annotations

import statistics
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn


def summarize(samples_ms: list[float], batch: int = 1) -> dict:
    s = sorted(samples_ms)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(round(p / 100 * (n - 1))))]

    mean = statistics.fmean(s)
    return {
        "mean_ms": mean, "p50_ms": pct(50), "p90_ms": pct(90), "p99_ms": pct(99),
        "min_ms": s[0], "std_ms": statistics.pstdev(s) if n > 1 else 0.0,
        "batch": batch, "throughput_ips": batch * 1000.0 / mean if mean > 0 else 0.0, "n": n,
    }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn: Callable[[], object], warmup: int = 50, iters: int = 300,
                  batch: int = 1) -> dict:
    """Time `fn` with one CUDA-event pair per iteration (CPU timer when no GPU)."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return summarize(samples, batch)


LEAF_TYPES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.ReLU6, nn.SiLU,
              nn.GELU, nn.Hardswish, nn.Sigmoid, nn.Hardsigmoid, nn.MaxPool2d,
              nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity)


def is_profiled_leaf(m: nn.Module) -> bool:
    from .models import GatedMHA
    from .fusion import FusedConvBNReLU
    if isinstance(m, (GatedMHA, FusedConvBNReLU)):
        return True
    if isinstance(m, (nn.Dropout, nn.Identity)):
        return False
    return len(list(m.children())) == 0 and isinstance(m, LEAF_TYPES)


class LayerTimer:
    """Per-module latency via CUDA events recorded in forward pre/post hooks.

    Events are recorded asynchronously on the current stream, so the measurement
    does not serialise the GPU; we synchronise once per forward pass. Hook overhead
    (~µs of CPU) only matters when the GPU is starved — check `cpu_bound_warning`.
    """

    def __init__(self, model: nn.Module, leaf_filter: Callable[[nn.Module], bool] = is_profiled_leaf):
        self.model = model
        self.handles = []
        self.records: "OrderedDict[str, list[float]]" = OrderedDict()
        self.meta: dict[str, dict] = {}
        self._pending: list[tuple[str, object, object]] = []
        self._acc: dict[str, float] = {}
        self._cpu: dict[str, float] = {}
        self.cuda = torch.cuda.is_available()
        for name, mod in model.named_modules():
            if leaf_filter(mod) and not any(name.startswith(n + ".") for n in self.meta):
                self.meta[name] = {"type": type(mod).__name__}
                self.records[name] = []
                self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
                self.handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(mod, inp):
            if self.cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._cpu[name] = ev
            else:
                self._cpu[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) and inp else inp
            o = out[0] if isinstance(out, tuple) else out
            if "in_shape" not in self.meta[name] and torch.is_tensor(x):
                self.meta[name]["in_shape"] = list(x.shape)
                self.meta[name]["out_shape"] = list(o.shape) if torch.is_tensor(o) else None
                self.meta[name]["dtype"] = str(x.dtype).replace("torch.", "")
            if self.cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._pending.append((name, self._cpu.pop(name), ev))
            else:
                dt = (time.perf_counter() - self._cpu.pop(name)) * 1e3
                self._acc[name] = self._acc.get(name, 0.0) + dt
            self.meta[name]["_calls_cur"] = self.meta[name].get("_calls_cur", 0) + 1
        return hook

    def flush(self):
        """Close one forward pass: a module called k times gets the *sum* of its k calls."""
        if self.cuda:
            torch.cuda.synchronize()
            for name, s, e in self._pending:
                self._acc[name] = self._acc.get(name, 0.0) + s.elapsed_time(e)
            self._pending.clear()
        for name, ms in self._acc.items():
            self.records[name].append(ms)
        self._acc.clear()
        for m in self.meta.values():
            if "_calls_cur" in m:
                m["calls"] = m.pop("_calls_cur")

    def run(self, x: torch.Tensor, warmup: int = 10, iters: int = 50) -> list[dict]:
        with torch.inference_mode():
            for _ in range(warmup):
                self.model(x)
            self.flush()
            for r in self.records.values():
                r.clear()
            for _ in range(iters):
                self.model(x)
                self.flush()
        return self.table()

    def table(self) -> list[dict]:
        rows = []
        for name, samples in self.records.items():
            if not samples:
                continue
            rows.append({"layer": name, **{k: v for k, v in self.meta[name].items()
                                            if not k.startswith("_")},
                         "mean_ms": statistics.fmean(samples),
                         "p50_ms": statistics.median(samples)})
        return rows

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
