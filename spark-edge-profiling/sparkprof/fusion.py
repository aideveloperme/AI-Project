"""Conv + BatchNorm + ReLU fusion experiments in PyTorch eager mode.

Three levels are compared layer-by-layer (see scripts/04_fusion_experiment.py):

  L0 unfused   Conv2d -> BatchNorm2d -> ReLU        3 kernels, 3 activation round-trips to DRAM
  L1 bn-fold   Conv2d(+bias, BN folded) -> ReLU     2 kernels, BN math disappears
  L2 fused     cudnn conv+bias+ReLU                 1 kernel, ReLU in the conv epilogue

On top of these, torch.compile and TensorRT apply the same fusions automatically;
their per-layer timings are collected by the TensorRT profiler for comparison.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d whose weight/bias absorb the (eval-mode) BatchNorm.

        y = gamma * (W*x + b - mean) / sqrt(var + eps) + beta
          = (W * s) * x + (b - mean) * s + beta,      s = gamma / sqrt(var + eps)
    """
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode).to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(conv.weight * s.reshape(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_((b - bn.running_mean) * s + bn.bias)
    return fused


def _cudnn_conv_relu_available(device) -> bool:
    return device.type == "cuda" and hasattr(torch, "cudnn_convolution_relu")


class FusedConvBNReLU(nn.Module):
    """Single-kernel Conv+bias+ReLU (cuDNN runtime fusion); falls back to conv+relu."""

    def __init__(self, conv: nn.Conv2d, relu: bool = True):
        super().__init__()
        self.conv = conv
        self.relu = relu

    def forward(self, x):
        c = self.conv
        if self.relu and _cudnn_conv_relu_available(x.device) and c.padding_mode == "zeros" \
                and isinstance(c.padding, tuple):
            return torch.cudnn_convolution_relu(x, c.weight, c.bias, c.stride, c.padding,
                                                c.dilation, c.groups)
        y = F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
        return F.relu(y) if self.relu else y


def _find_conv_bn_relu(model: nn.Module):
    """Yield (parent, conv_name, bn_name, relu_name|None) triples from module order.

    Works on torchvision ResNets (conv1/bn1/relu at the stem; conv{i}/bn{i} inside
    Bottlenecks where a single shared `relu` follows) and on plain nn.Sequential
    Conv-BN-Act stacks (EfficientNet/MobileNet Conv2dNormActivation).
    """
    for parent in model.modules():
        names = list(parent._modules.keys())
        mods = list(parent._modules.values())
        for i, m in enumerate(mods):
            if not isinstance(m, nn.Conv2d) or i + 1 >= len(mods):
                continue
            if not isinstance(mods[i + 1], nn.BatchNorm2d):
                continue
            relu_name = None
            if i + 2 < len(mods) and isinstance(mods[i + 2], nn.ReLU):
                relu_name = names[i + 2]
            yield parent, names[i], names[i + 1], relu_name


def fold_all_bn(model: nn.Module) -> nn.Module:
    """L1: replace every Conv->BN pair by a bias-carrying Conv and the BN by Identity."""
    model = copy.deepcopy(model).eval()
    for parent, cn, bn, _ in list(_find_conv_bn_relu(model)):
        setattr(parent, cn, fold_bn_into_conv(getattr(parent, cn), getattr(parent, bn)))
        setattr(parent, bn, nn.Identity())
    return model


def fuse_resnet_conv_bn_relu(model: nn.Module) -> nn.Module:
    """L2 for torchvision ResNets: fold BN and fuse the following ReLU into the conv.

    In a Bottleneck, conv1/bn1 and conv2/bn2 are each followed by ReLU; conv3/bn3 is
    followed by the residual add *then* ReLU, so only BN is folded there. The block's
    forward is patched accordingly.
    """
    from torchvision.models.resnet import BasicBlock, Bottleneck

    model = copy.deepcopy(model).eval()
    model.conv1 = FusedConvBNReLU(fold_bn_into_conv(model.conv1, model.bn1), relu=True)
    model.bn1, model.relu = nn.Identity(), nn.Identity()

    for blk in model.modules():
        if isinstance(blk, Bottleneck):
            blk.conv1 = FusedConvBNReLU(fold_bn_into_conv(blk.conv1, blk.bn1), True)
            blk.conv2 = FusedConvBNReLU(fold_bn_into_conv(blk.conv2, blk.bn2), True)
            blk.conv3 = FusedConvBNReLU(fold_bn_into_conv(blk.conv3, blk.bn3), False)
            blk.bn1 = blk.bn2 = blk.bn3 = nn.Identity()
            blk.forward = _fused_bottleneck_forward.__get__(blk)
        elif isinstance(blk, BasicBlock):
            blk.conv1 = FusedConvBNReLU(fold_bn_into_conv(blk.conv1, blk.bn1), True)
            blk.conv2 = FusedConvBNReLU(fold_bn_into_conv(blk.conv2, blk.bn2), False)
            blk.bn1 = blk.bn2 = nn.Identity()
            blk.forward = _fused_basic_forward.__get__(blk)
        if getattr(blk, "downsample", None) is not None and len(blk.downsample) == 2:
            ds_conv, ds_bn = blk.downsample[0], blk.downsample[1]
            blk.downsample = FusedConvBNReLU(fold_bn_into_conv(ds_conv, ds_bn), False)
    return model


def _fused_bottleneck_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = self.conv3(self.conv2(self.conv1(x)))
    return self.relu(out + identity)  # post-residual ReLU (kept as a module so it is timed)


def _fused_basic_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    return self.relu(self.conv2(self.conv1(x)) + identity)


def count_ops(model: nn.Module) -> dict:
    c = {"conv": 0, "bn": 0, "relu": 0, "fused": 0}
    for m in model.modules():
        if isinstance(m, FusedConvBNReLU):
            c["fused"] += 1
        elif isinstance(m, nn.Conv2d):
            c["conv"] += 1
        elif isinstance(m, nn.BatchNorm2d):
            c["bn"] += 1
        elif isinstance(m, nn.ReLU):
            c["relu"] += 1
    c["conv"] -= c["fused"]  # don't double count convs wrapped by FusedConvBNReLU
    return c


def fused_units(rows: list[dict]) -> dict[str, float]:
    """Collapse per-layer rows (per-forward totals) into comparable *fusion units*.

    Unit keys are the conv names: `layer1.0.conv1` covers conv1 [+ bn1] [+ its ReLU call],
    so the same key exists before and after fusion. Torchvision Bottlenecks share one
    `relu` module that runs 3x per block (after bn1, after bn2, after the residual add):
    in the unfused model 1/3 of its time is attributed to conv1 and conv2 each and the
    last third stays as `<block>.relu` (post-residual), which also exists after fusion.
    Downsample paths are keyed `<block>.downsample`.
    """
    t = {r["layer"]: r["mean_ms"] for r in rows}
    calls = {r["layer"]: r.get("calls", 1) for r in rows}
    out: dict[str, float] = {}
    for name, ms in t.items():
        parent, _, leaf = name.rpartition(".")
        pre = parent + "." if parent else ""
        if leaf.startswith("conv"):
            idx = leaf[4:]
            out[name] = ms + t.get(f"{pre}bn{idx}", 0.0)
            relu = f"{pre}relu"
            if relu in t and calls.get(relu, 1) > 1 and idx in ("1", "2"):
                out[name] += t[relu] / calls[relu]
            elif relu in t and calls.get(relu, 1) == 1 and not parent:
                out[name] += t[relu]  # stem: conv1 -> bn1 -> relu
        elif parent.endswith("downsample") and leaf == "0":
            out[parent] = ms + t.get(f"{parent}.1", 0.0)
        elif leaf == "relu" and parent:
            out[name] = ms / calls.get(name, 1)  # post-residual share
        elif leaf.startswith(("bn", "relu")) or (parent.endswith("downsample") and leaf == "1"):
            continue
        else:
            out[name] = ms
    return out

