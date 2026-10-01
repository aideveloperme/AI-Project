"""ONNX export, TensorRT engine building and an engine runner with per-layer profiling.

The runner allocates I/O with PyTorch on the GPU and calls `execute_async_v3`, so there
is no extra dependency (no pycuda / cuda-python). Per-layer timing uses TensorRT's
IProfiler; layer names reveal the fusions TensorRT performed, e.g.
  "/layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu"   (BN already folded)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


def _trt():
    import tensorrt as trt
    return trt


# --------------------------------------------------------------------------- export

def export_onnx(model: nn.Module, path: str | Path, resolution: int, opset: int = 19,
                dynamic_batch: bool = True, batch: int = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    model = model.eval()
    dummy = torch.randn(batch, 3, resolution, resolution, device=dev)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              do_constant_folding=True)
    if dynamic_batch:
        kw["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}
    with torch.no_grad():
        try:
            torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kw)
        except TypeError:  # older torch without the `dynamo` kwarg
            torch.onnx.export(model, (dummy,), str(path), **kw)
    return path


# --------------------------------------------------------------------------- build

PRECISION_FLAGS = {
    # name           builder flags                      notes
    "fp32":        [],                                  # TF32 disabled below -> true FP32
    "tf32":        ["TF32"],
    "fp16":        ["FP16"],
    "bf16":        ["BF16"],
    "int8_qdq":    ["INT8", "FP16"],                    # explicit Q/DQ ONNX (PTQ, QAT, mixed)
    "fp8_qdq":     ["FP8", "FP16"],
}


def build_engine(onnx_path: str | Path, engine_path: str | Path, precision: str, resolution: int,
                 min_batch: int = 1, opt_batch: int = 8, max_batch: int = 32,
                 workspace_gb: float = 8, opt_level: int = 3, tiling_level: str | None = None,
                 io_format: str | None = None, timing_cache: str | Path | None = None,
                 verbose: bool = False) -> dict:
    """Build and serialise a TensorRT engine; returns build metadata.

    tiling_level: Blackwell/TRT>=10.8 `TilingOptimizationLevel` (NONE/FAST/MODERATE/FULL)
    io_format:    restrict the input tensor format, e.g. "LINEAR" (NCHW) or "HWC8"/"HWC"
                  (channels-last). Requires a reformat-free I/O or TRT inserts a reformat layer.
    """
    trt = _trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for f in PRECISION_FLAGS[precision]:
        config.set_flag(getattr(trt.BuilderFlag, f))
    if precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    applied_tiling = None
    if tiling_level and hasattr(config, "tiling_optimization_level"):
        config.tiling_optimization_level = getattr(trt.TilingOptimizationLevel, tiling_level)
        applied_tiling = tiling_level

    inp = network.get_input(0)
    if io_format:
        inp.allowed_formats = 1 << int(getattr(trt.TensorFormat, io_format))
        if io_format != "LINEAR":
            inp.dtype = trt.float16  # vectorised HWC formats are FP16/INT8 formats

    profile = builder.create_optimization_profile()
    shape = lambda b: (b, 3, resolution, resolution)  # noqa: E731
    profile.set_shape(inp.name, shape(min_batch), shape(opt_batch), shape(max_batch))
    config.add_optimization_profile(profile)

    cache = None
    if timing_cache:
        blob = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b""
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, ignore_mismatch=False)

    import time
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path} ({precision})")
    build_s = time.time() - t0
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_path).write_bytes(bytes(serialized))
    if cache is not None:
        Path(timing_cache).write_bytes(bytes(cache.serialize()))
    return {"engine": str(engine_path), "precision": precision, "build_s": build_s,
            "engine_mb": os.path.getsize(engine_path) / 1e6, "tiling_level": applied_tiling,
            "io_format": io_format, "trt_version": trt.__version__}


# --------------------------------------------------------------------------- run

class _LayerProfiler:
    """Implements trt.IProfiler; accumulates ms per layer across runs."""

    def __new__(cls):
        trt = _trt()

        class P(trt.IProfiler):
            def __init__(self):
                trt.IProfiler.__init__(self)
                self.times: dict[str, list[float]] = {}
                self.order: list[str] = []

            def report_layer_time(self, layer_name, ms):
                if layer_name not in self.times:
                    self.times[layer_name] = []
                    self.order.append(layer_name)
                self.times[layer_name].append(ms)

        return P()


_TRT_TO_TORCH = {"FLOAT": torch.float32, "HALF": torch.float16, "BF16": torch.bfloat16,
                 "INT8": torch.int8, "INT32": torch.int32, "INT64": torch.int64, "BOOL": torch.bool}


class TRTRunner:
    def __init__(self, engine_path: str | Path, device_index: int = 0):
        trt = _trt()
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self._bufs: dict[str, torch.Tensor] = {}
        self._batch = None

    def _dtype(self, name):
        return _TRT_TO_TORCH[self.engine.get_tensor_dtype(name).name]

    def _prepare(self, batch: int, resolution: int):
        if self._batch == batch:
            return
        inp = self.inputs[0]
        self.context.set_input_shape(inp, (batch, 3, resolution, resolution))
        for n in self.names:
            shape = list(self.context.get_tensor_shape(n))
            # vectorised formats (HWC8, CHW32, ...) pad the vectorised dim to a multiple
            # of the components per element; allocate the padded size.
            comps = self.engine.get_tensor_components_per_element(n)
            vdim = self.engine.get_tensor_vectorized_dim(n)
            if comps > 1 and vdim >= 0:
                shape[vdim] = -(-shape[vdim] // comps) * comps
            self._bufs[n] = torch.empty(shape, dtype=self._dtype(n), device="cuda")
            self.context.set_tensor_address(n, self._bufs[n].data_ptr())
        self._batch = batch

    @property
    def linear_input(self) -> bool:
        return self.engine.get_tensor_format(self.inputs[0]).name == "LINEAR"

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Copy x in, run, return a view of the output buffer (valid until the next call)."""
        if not self.linear_input:
            raise RuntimeError("infer() needs a LINEAR (NCHW) input; this engine is latency-only")
        self._prepare(x.shape[0], x.shape[-1])
        inp = self._bufs[self.inputs[0]]
        with torch.cuda.stream(self.stream):
            inp.copy_(x, non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self._bufs[self.outputs[0]]

    def enqueue(self) -> None:
        """Run on whatever is already in the input buffer (pure-GPU timing loop)."""
        self.context.execute_async_v3(self.stream.cuda_stream)

    def sync(self) -> None:
        self.stream.synchronize()

    def setup(self, batch: int, resolution: int) -> None:
        self._prepare(batch, resolution)
        self._bufs[self.inputs[0]].normal_()

    def time(self, batch: int, resolution: int, warmup: int = 50, iters: int = 300) -> dict:
        from .timing import summarize
        self.setup(batch, resolution)
        for _ in range(warmup):
            self.enqueue()
        self.sync()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record(self.stream)
            self.enqueue()
            ends[i].record(self.stream)
        self.sync()
        return summarize([s.elapsed_time(e) for s, e in zip(starts, ends)], batch)

    def layer_times(self, batch: int, resolution: int, iters: int = 50) -> list[dict]:
        self.setup(batch, resolution)
        for _ in range(10):
            self.enqueue()
        self.sync()
        prof = _LayerProfiler()
        self.context.profiler = prof
        for _ in range(iters):
            self._sync_run()
        self.context.profiler = None
        return [{"layer": n, "mean_ms": sum(v) / len(v), "calls": len(v)}
                for n in prof.order for v in [prof.times[n]]]

    def _sync_run(self):
        # IProfiler needs synchronous completion per enqueue
        self.enqueue()
        self.sync()

    def device_memory_bytes(self) -> int:
        e = self.engine
        return int(getattr(e, "device_memory_size_v2", None) or e.device_memory_size)

    def layer_info(self) -> list[dict]:
        insp = self.engine.create_engine_inspector()
        info = json.loads(insp.get_engine_information(self.trt.LayerInformationFormat.JSON))
        layers = info.get("Layers", [])
        return [l if isinstance(l, dict) else {"Name": l} for l in layers]


def precision_histogram(layer_info: list[dict]) -> dict[str, int]:
    """Count engine layers by the precision of their first output (from the inspector)."""
    hist: dict[str, int] = {}
    for l in layer_info:
        outs = l.get("Outputs") or []
        fmt = outs[0].get("Format/Datatype", "?") if outs and isinstance(outs[0], dict) else "?"
        key = next((p for p in ("Int8", "FP8", "Half", "BFloat16", "Float") if p in fmt), fmt)
        hist[key] = hist.get(key, 0) + 1
    return hist


@torch.inference_mode()
def evaluate_engine(runner: TRTRunner, loader, max_batch: int) -> dict:
    correct = total = 0
    for x, y in loader:
        for i in range(0, x.shape[0], max_batch):
            xb = x[i:i + max_batch].cuda(non_blocking=True)
            logits = runner.infer(xb).float()
            correct += (logits.argmax(1).cpu() == y[i:i + max_batch]).sum().item()
            total += xb.shape[0]
    return {"top1": 100.0 * correct / max(total, 1), "n": total}
