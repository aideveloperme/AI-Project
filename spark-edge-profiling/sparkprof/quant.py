"""INT8 / FP8 quantisation with NVIDIA TensorRT Model Optimizer (nvidia-modelopt).

Flow (explicit quantisation — the path TensorRT 10 recommends):

  FP32 model --mtq.quantize(cfg, calib loop)--> fake-quant model (Q/DQ around Conv/Linear)
     |                                               |
     |  PTQ: export as-is                            |  QAT: fine-tune with STE, then export
     v                                               v
  ONNX with QuantizeLinear/DequantizeLinear nodes -> TensorRT builds INT8 kernels exactly
  where Q/DQ pairs sit; un-quantised layers run in FP16 (=> mixed precision).

Mixed precision = INT8 everywhere except the `k` layers whose isolated quantisation
hurts the output most (sensitivity analysis below), which stay FP16.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mtq():
    try:
        import modelopt.torch.quantization as mtq
        return mtq
    except ImportError as e:
        raise ImportError("INT8/FP8 experiments need NVIDIA ModelOpt: "
                          "pip install 'nvidia-modelopt[torch]'") from e


def quant_config(kind: str) -> dict:
    mtq = _mtq()
    return copy.deepcopy({"int8": mtq.INT8_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG}[kind])


def ptq(model: nn.Module, calib_loader, kind: str = "int8", dev=None) -> nn.Module:
    """Post-training quantisation: insert quantizers and calibrate amax on `calib_loader`."""
    mtq = _mtq()
    dev = dev or next(model.parameters()).device
    model = copy.deepcopy(model).to(dev).eval()

    def forward_loop(m):
        with torch.no_grad():
            for x, _ in calib_loader:
                m(x.to(dev))

    return mtq.quantize(model, quant_config(kind), forward_loop)


def quantized_layers(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules()
            if hasattr(m, "weight_quantizer") and hasattr(m, "input_quantizer")]


def set_layer_quant(model: nn.Module, names, enabled: bool) -> None:
    mods = dict(model.named_modules())
    for n in names:
        m = mods[n]
        for q in (m.input_quantizer, m.weight_quantizer):
            q.enable() if enabled else q.disable()


@torch.no_grad()
def sensitivity(qmodel: nn.Module, fp_model: nn.Module, loader, batches: int = 4, dev=None) -> list[dict]:
    """KL(FP32 || model with ONLY layer i quantised) for every quantised layer, sorted desc.

    Isolating one layer at a time measures each layer's own damage; the most damaging
    layers are the ones worth keeping in FP16 for the mixed-precision engine.
    """
    dev = dev or next(qmodel.parameters()).device
    fp_model = fp_model.to(dev).eval()
    qmodel = qmodel.eval()
    data = [(x.to(dev), y) for i, (x, y) in enumerate(loader) if i < batches]
    ref = [F.log_softmax(fp_model(x).float(), -1) for x, _ in data]
    layers = quantized_layers(qmodel)
    set_layer_quant(qmodel, layers, False)
    out = []
    for name in layers:
        set_layer_quant(qmodel, [name], True)
        kl = 0.0
        for (x, _), r in zip(data, ref):
            lq = F.log_softmax(qmodel(x).float(), -1)
            kl += F.kl_div(lq, r, log_target=True, reduction="batchmean").item()
        out.append({"layer": name, "kl": kl / len(data)})
        set_layer_quant(qmodel, [name], False)
    set_layer_quant(qmodel, layers, True)
    return sorted(out, key=lambda d: -d["kl"])


def make_mixed(qmodel: nn.Module, keep_fp16: list[str]) -> nn.Module:
    m = copy.deepcopy(qmodel)
    set_layer_quant(m, keep_fp16, False)
    return m
