"""Accelerator-oriented model surgery.

* replace_activation   SiLU -> ReLU6 (or any act -> act). ReLU6 is a clamp that every
                       accelerator / TensorRT INT8 kernel fuses into the conv epilogue;
                       SiLU needs a sigmoid (transcendental) and keeps more dynamic range,
                       which hurts INT8 calibration.
* resolution           handled by the experiment script — CNNs are resolution-agnostic,
                       so only the input shape and the TRT optimisation profile change.
* attention-head pruning (ViT): score heads with the gradient of a per-head gate
                       (Michel et al., "Are Sixteen Heads Really Better than One?", 2019),
                       keep the top-k per layer, and physically slice qkv/proj weights so the
                       pruned model is a smaller *dense* model (no masks at inference).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GatedMHA


def replace_activation(model: nn.Module, old: type = nn.SiLU,
                       new_fn=lambda: nn.ReLU6(inplace=False)) -> tuple[nn.Module, int]:
    model = copy.deepcopy(model)
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child) is old:
                setattr(parent, name, new_fn())
                n += 1
    return model, n


def attention_modules(model: nn.Module) -> list[tuple[str, GatedMHA]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, GatedMHA)]


def head_importance(model: nn.Module, loader, device, batches: int = 8) -> dict[str, torch.Tensor]:
    """|dL/d gate_h| accumulated over `batches` labelled batches, L2-normalised per layer."""
    model = model.to(device).eval()
    attns = attention_modules(model)
    for _, m in attns:
        m.use_gate = True
        m.head_gate = torch.ones(m.num_heads, device=device, requires_grad=True)
    scores = {n: torch.zeros(m.num_heads, device=device) for n, m in attns}
    for i, (x, y) in enumerate(loader):
        if i >= batches:
            break
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, [m.head_gate for _, m in attns])
        for (n, _), g in zip(attns, grads):
            scores[n] += g.abs().detach()
    for n, m in attns:
        m.use_gate = False
        m.head_gate = torch.ones(m.num_heads, device=device)
        s = scores[n]
        scores[n] = (s / (s.norm() + 1e-12)).cpu()
    return scores


def prune_heads(attn: GatedMHA, keep: list[int]) -> GatedMHA:
    keep = sorted(keep)
    hd, D, H = attn.head_dim, attn.embed_dim, attn.num_heads
    idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in keep])
    qkv_idx = torch.cat([idx + s * H * hd for s in range(3)])
    new = GatedMHA(D, len(keep), hd, attn.dropout).to(attn.qkv.weight.device, attn.qkv.weight.dtype)
    with torch.no_grad():
        new.qkv.weight.copy_(attn.qkv.weight[qkv_idx])
        new.qkv.bias.copy_(attn.qkv.bias[qkv_idx])
        new.proj.weight.copy_(attn.proj.weight[:, idx])
        new.proj.bias.copy_(attn.proj.bias)
    return new


def prune_vit_heads(model: nn.Module, scores: dict[str, torch.Tensor],
                    keep_ratio: float) -> tuple[nn.Module, dict]:
    """Keep the top `keep_ratio` heads in every layer (uniform -> identical layer shapes)."""
    model = copy.deepcopy(model)
    mods = dict(model.named_modules())
    kept = {}
    for name, s in scores.items():
        attn = mods[name]
        k = max(1, round(attn.num_heads * keep_ratio))
        keep = torch.topk(s, k).indices.tolist()
        parent_name, _, attr = name.rpartition(".")
        setattr(mods[parent_name], attr, prune_heads(attn, keep))
        kept[name] = sorted(keep)
    return model, kept
