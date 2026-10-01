"""Analytical per-layer cost model (FLOPs, minimum DRAM bytes) and roofline analysis.

For every profiled layer we compute
  FLOPs           multiply-add = 2 FLOPs
  bytes (min)     compulsory traffic: read input + weights, write output, once each
  AI              arithmetic intensity = FLOPs / bytes                  [FLOP/B]
and, combined with the measured latency t,
  attained        FLOPs / t                                              [TFLOP/s]
  achieved BW     bytes / t                                              [GB/s]
  bound           memory if AI < ridge point (peak_flops / peak_bw) else compute
  efficiency      attained / min(peak_flops, AI * peak_bw)

When Nsight Compute DRAM measurements exist (scripts/08_ncu_layers.sh) the measured
bytes replace the analytical minimum, which exposes layers whose real traffic is far
above compulsory (poor reuse / tiling, layout transposes, spills).
"""

from __future__ import annotations

import math

import torch.nn as nn

from .models import GatedMHA
from .fusion import FusedConvBNReLU

ACT_FLOPS = {"ReLU": 1, "ReLU6": 2, "SiLU": 5, "GELU": 8, "Hardswish": 4, "Sigmoid": 4,
             "Hardsigmoid": 3}
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "float8_e4m3fn": 1}


def _numel(shape) -> int:
    return math.prod(shape) if shape else 0


def module_cost(mod: nn.Module, in_shape, out_shape, elem_bytes: int,
                weight_bytes: int | None = None) -> tuple[float, float]:
    """Return (flops, min_bytes) for one call of `mod` with the given shapes."""
    wb = weight_bytes or elem_bytes
    n_in, n_out = _numel(in_shape), _numel(out_shape or in_shape)
    act_bytes = (n_in + n_out) * elem_bytes

    conv = mod.conv if isinstance(mod, FusedConvBNReLU) else mod
    if isinstance(conv, nn.Conv2d):
        kh, kw = conv.kernel_size
        flops = 2.0 * n_out * (conv.in_channels // conv.groups) * kh * kw
        if conv.bias is not None:
            flops += n_out
        if isinstance(mod, FusedConvBNReLU) and mod.relu:
            flops += n_out
        return flops, act_bytes + conv.weight.numel() * wb
    if isinstance(mod, nn.Linear):
        tokens = n_in // mod.in_features
        return 2.0 * tokens * mod.in_features * mod.out_features, act_bytes + mod.weight.numel() * wb
    if isinstance(mod, GatedMHA):
        B, N, D = in_shape
        H, hd = mod.num_heads, mod.head_dim
        inner = H * hd
        flops = 2.0 * B * N * D * 3 * inner          # qkv projection
        flops += 2.0 * 2 * B * H * N * N * hd         # QK^T and PV
        flops += 5.0 * B * H * N * N                  # softmax
        flops += 2.0 * B * N * inner * D              # output projection
        w = (mod.qkv.weight.numel() + mod.proj.weight.numel()) * wb
        # flash/SDPA keeps the NxN score matrix on-chip; q,k,v and the context still hit DRAM
        inter = 2 * 4 * B * N * inner * elem_bytes
        return flops, act_bytes + w + inter
    if isinstance(mod, nn.BatchNorm2d):
        return 2.0 * n_in, act_bytes + 4 * mod.num_features * 4
    if isinstance(mod, nn.LayerNorm):
        return 8.0 * n_in, act_bytes + 2 * _numel(mod.normalized_shape) * 4
    name = type(mod).__name__
    if name in ACT_FLOPS:
        return float(ACT_FLOPS[name] * n_in), act_bytes
    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return float(n_in), act_bytes
    return 0.0, act_bytes


def annotate_costs(model: nn.Module, rows: list[dict], weight_bytes: int | None = None) -> list[dict]:
    """Add flops / bytes / AI to LayerTimer rows (in place) and return them.

    A module called k times per forward (e.g. the shared ReLU in a ResNet Bottleneck)
    is costed k times with its first-call shape — exact for ResNet, where the three
    calls in a block see the same-sized tensors except the expansion output.
    """
    mods = dict(model.named_modules())
    for r in rows:
        mod = mods.get(r["layer"])
        if mod is None or "in_shape" not in r:
            continue
        eb = DTYPE_BYTES.get(r.get("dtype", "float32"), 4)
        f, b = module_cost(mod, r["in_shape"], r.get("out_shape"), eb, weight_bytes)
        k = r.get("calls", 1)
        r["flops"], r["bytes"] = f * k, b * k
        r["ai"] = r["flops"] / r["bytes"] if r["bytes"] else 0.0
    return rows


def classify(rows: list[dict], peak_tflops: float, peak_gbs: float,
             bytes_key: str = "bytes") -> list[dict]:
    ridge = peak_tflops * 1e12 / (peak_gbs * 1e9)
    for r in rows:
        t = r.get("mean_ms", 0) / 1e3
        by = r.get(bytes_key) or r.get("bytes", 0)
        if not t or not by:
            continue
        ai = r.get("flops", 0) / by
        r["ai_used"] = ai
        r["attained_tflops"] = r.get("flops", 0) / t / 1e12
        r["achieved_gbs"] = by / t / 1e9
        roof = min(peak_tflops, ai * peak_gbs / 1e3)
        r["roof_tflops"] = roof
        r["efficiency"] = r["attained_tflops"] / roof if roof else 0.0
        r["bound"] = "memory" if ai < ridge else "compute"
        r["bw_util"] = r["achieved_gbs"] / peak_gbs
    return rows


def summary(rows: list[dict]) -> dict:
    tot = sum(r.get("mean_ms", 0) for r in rows) or 1.0
    mem = [r for r in rows if r.get("bound") == "memory"]
    comp = [r for r in rows if r.get("bound") == "compute"]
    return {
        "layers": len(rows),
        "memory_bound_layers": len(mem),
        "compute_bound_layers": len(comp),
        "memory_bound_time_pct": 100 * sum(r["mean_ms"] for r in mem) / tot,
        "compute_bound_time_pct": 100 * sum(r["mean_ms"] for r in comp) / tot,
        "total_gflops": sum(r.get("flops", 0) for r in rows) / 1e9,
        "total_min_mb": sum(r.get("bytes", 0) for r in rows) / 1e6,
        "top_memory_bound": sorted(
            ({"layer": r["layer"], "type": r["type"], "ms": r["mean_ms"], "ai": r.get("ai_used"),
              "gbs": r.get("achieved_gbs")} for r in mem), key=lambda d: -d["ms"])[:10],
    }


def plot_roofline(series: dict[str, list[dict]], peak_tflops: float, peak_gbs: float, path: str,
                  title: str, extra_ceilings: dict[str, float] | None = None,
                  nominal: tuple[float, float] | None = None) -> None:
    """Log-log roofline. `series` maps a label (e.g. 'FP32 eager') to classified rows;
    marker area is proportional to the layer's share of runtime."""
    from . import plotstyle
    plt = plotstyle.apply()
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    all_ai = [r["ai_used"] for rows in series.values() for r in rows if r.get("ai_used")]
    lo = max(min(all_ai, default=0.1) / 3, 1e-2)
    hi = max(max(all_ai, default=1e3) * 3, peak_tflops * 1e3 / peak_gbs * 10)
    x = np.logspace(math.log10(lo), math.log10(hi), 200)

    ax.plot(x, np.minimum(peak_tflops, x * peak_gbs / 1e3), color=plotstyle.TEXT, lw=2,
            label=f"roof used ({peak_gbs:.0f} GB/s, {peak_tflops:.1f} TFLOP/s)")
    if nominal and (abs(nominal[0] - peak_tflops) > 1e-6 or abs(nominal[1] - peak_gbs) > 1e-6):
        ax.plot(x, np.minimum(nominal[0], x * nominal[1] / 1e3), color=plotstyle.TEXT2, lw=1,
                ls="--", label=f"nominal ({nominal[1]:.0f} GB/s, {nominal[0]:.0f} TFLOP/s)")
    for name, tf in (extra_ceilings or {}).items():
        ax.axhline(tf, color=plotstyle.TEXT2, lw=0.8, ls=":")
        ax.text(hi, tf, f" {name}", va="bottom", ha="right", fontsize=8, color=plotstyle.TEXT2)
    ridge = peak_tflops * 1e3 / peak_gbs
    ax.axvline(ridge, color=plotstyle.GRID, lw=1)
    ax.text(ridge, peak_tflops * 1.15, f"ridge {ridge:.0f} FLOP/B",
            ha="center", fontsize=8, color=plotstyle.TEXT2)

    for i, (label, rows) in enumerate(series.items()):
        pts = [r for r in rows if r.get("ai_used") and r.get("attained_tflops")]
        if not pts:
            continue
        tot = sum(r["mean_ms"] for r in pts)
        ax.scatter([r["ai_used"] for r in pts], [r["attained_tflops"] for r in pts],
                   s=[max(10, 900 * r["mean_ms"] / tot) for r in pts],
                   color=plotstyle.SERIES[i % 3], alpha=0.7, edgecolor=plotstyle.SURFACE,
                   linewidth=1.5, label=label, zorder=3)
        for r in sorted(pts, key=lambda r: -r["mean_ms"])[:4]:
            ax.annotate(r["layer"], (r["ai_used"], r["attained_tflops"]), fontsize=7,
                        color=plotstyle.TEXT2, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity [FLOP / byte of DRAM traffic]")
    ax.set_ylabel("Attained performance [TFLOP/s]")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
