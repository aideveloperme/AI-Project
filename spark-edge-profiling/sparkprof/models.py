"""Model zoo (torchvision) adapted to Imagenette, plus a head-gated attention module.

torchvision's ViT uses nn.MultiheadAttention, which can neither be pruned per head
nor costed per layer. Every ViT built here therefore has its attention replaced by
`GatedMHA`: numerically identical, but with an explicit `num_heads`, a per-head
gate (used for importance scoring) and FLOP/byte bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMHA(nn.Module):
    """Multi-head self-attention with a removable per-head gate.

    Drop-in for torchvision's EncoderBlock.self_attention: called as
    ``self_attention(x, x, x, need_weights=False)`` and returns ``(out, None)``.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, head_dim
        inner = num_heads * head_dim
        self.qkv = nn.Linear(embed_dim, 3 * inner)
        self.proj = nn.Linear(inner, embed_dim)
        self.dropout = dropout
        # gate is a buffer (not a Parameter) so it never appears in state_dict optimisers;
        # head-importance scoring flips requires_grad on it temporarily.
        self.register_buffer("head_gate", torch.ones(num_heads), persistent=False)
        self.use_gate = False

    @classmethod
    def from_torch_mha(cls, mha: nn.MultiheadAttention) -> "GatedMHA":
        m = cls(mha.embed_dim, mha.num_heads, mha.head_dim, mha.dropout)
        with torch.no_grad():
            m.qkv.weight.copy_(mha.in_proj_weight)
            m.qkv.bias.copy_(mha.in_proj_bias)
            m.proj.weight.copy_(mha.out_proj.weight)
            m.proj.bias.copy_(mha.out_proj.bias)
        return m

    def forward(self, q, k=None, v=None, need_weights: bool = False, **_):
        x = q
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_, k_, v_ = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q_, k_, v_,
                                             dropout_p=self.dropout if self.training else 0.0)
        if self.use_gate:
            out = out * self.head_gate.view(1, -1, 1, 1)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(out), None


def convert_vit_attention(model: nn.Module) -> nn.Module:
    for name, mod in list(model.named_modules()):
        if hasattr(mod, "self_attention") and isinstance(mod.self_attention, nn.MultiheadAttention):
            mod.self_attention = GatedMHA.from_torch_mha(mod.self_attention)
    return model


def _replace_classifier(model: nn.Module, arch: str, num_classes: int) -> None:
    if arch.startswith(("resnet", "regnet", "resnext")):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith(("efficientnet", "mobilenet", "convnext")):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    elif arch.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"don't know how to replace the classifier of {arch}")


def build_model(arch: str, num_classes: int = 10, pretrained: bool = True,
                image_size: int = 224) -> nn.Module:
    import torchvision

    kwargs = {}
    if arch.startswith("vit"):
        kwargs["image_size"] = image_size if not pretrained else 224
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.get_model(arch, weights=weights, **kwargs)
    _replace_classifier(model, arch, num_classes)
    if arch.startswith("vit"):
        convert_vit_attention(model)
    return model


def ckpt_path(cfg, tag: str) -> Path:
    return Path(cfg.paths.ckpt_dir) / f"{tag}.pt"


def save_checkpoint(model: nn.Module, cfg, tag: str, meta: dict | None = None) -> Path:
    p = ckpt_path(cfg, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    # the whole module is saved so surgically-modified architectures reload as-is
    torch.save({"model": model, "meta": meta or {}}, p)
    return p


def load_checkpoint(cfg, tag: str, map_location="cpu") -> nn.Module:
    p = ckpt_path(cfg, tag)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run scripts/02_train_baselines.py first")
    return torch.load(p, map_location=map_location, weights_only=False)["model"]


def load_or_build(cfg, arch: str) -> nn.Module:
    """Fine-tuned checkpoint if present, otherwise a fresh (pretrained-backbone) model."""
    try:
        return load_checkpoint(cfg, arch)
    except FileNotFoundError:
        print(f"[warn] no fine-tuned checkpoint for {arch}; accuracy numbers will be meaningless")
        return build_model(arch, cfg.data.num_classes, pretrained=cfg.data.dataset != "synthetic")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
