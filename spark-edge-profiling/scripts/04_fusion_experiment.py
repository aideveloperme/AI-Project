#!/usr/bin/env python3
"""Step 4 — Conv+BN+ReLU fusion: per-layer timing before/after, kernel counts, numerics.

Levels (ResNet-family model, eager PyTorch):
  L0 unfused      Conv -> BN -> ReLU             (3 kernels per unit)
  L1 bn_folded    Conv(+bias) -> ReLU            (BN folded into weights)
  L2 fused        cuDNN conv+bias+ReLU            (1 kernel per unit)
  L3 compiled     torch.compile(max-autotune) of L1 (Inductor fuses pointwise ops)
  TRT             TensorRT FP32/FP16 engine; per-layer timings from IProfiler show the
                  fused layer names TensorRT produced
"""

from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sparkprof import plotstyle
from sparkprof.bench import count_cuda_kernels, eager_model, make_input, per_layer_profile
from sparkprof.config import base_parser, config_from_args, save_json
from sparkprof.fusion import count_ops, fold_all_bn, fuse_resnet_conv_bn_relu, fused_units
from sparkprof.models import load_or_build
from sparkprof.timing import time_callable


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-trt", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    arch = args.model or cfg.models.cnn
    assert arch.startswith("resnet"), "the per-unit fusion mapping is written for torchvision ResNets"
    res, B = cfg.data.resolution, args.batch
    base = load_or_build(cfg, arch).eval()
    levels = {"L0_unfused": base, "L1_bn_folded": fold_all_bn(base),
              "L2_fused": fuse_resnet_conv_bn_relu(base)}
    out = {"arch": arch, "batch": B, "resolution": res, "ops": {k: count_ops(v) for k, v in levels.items()},
           "precisions": {}}

    for prec in args.precisions:
        P = out["precisions"][prec] = {"whole_model": {}, "per_layer": {}, "units": {}, "kernels": {},
                                       "max_abs_diff_vs_L0": {}}
        x = make_input(B, res, prec)
        ref = None
        for name, m in levels.items():
            em = eager_model(m, prec)
            with torch.inference_mode():
                y = em(x).float()
                ref = y if ref is None else ref
                P["max_abs_diff_vs_L0"][name] = (y - ref).abs().max().item()
                P["whole_model"][name] = time_callable(lambda: em(x), cfg.bench.warmup, cfg.bench.iters, B)
            P["kernels"][name] = count_cuda_kernels(em, x)
            rows = per_layer_profile(m, res, B, prec, iters=max(20, cfg.bench.iters // 5))
            P["per_layer"][name] = rows
            P["units"][name] = fused_units(rows)
            print(f"[{prec}] {name:13s} {P['whole_model'][name]['mean_ms']:.3f} ms  "
                  f"kernels/forward={P['kernels'][name]}  max|dy|={P['max_abs_diff_vs_L0'][name]:.2e}")

        if not args.no_compile:
            try:
                cm = torch.compile(eager_model(levels["L1_bn_folded"], prec), mode="max-autotune")
                with torch.inference_mode():
                    cm(x)
                    P["whole_model"]["L3_compiled"] = time_callable(lambda: cm(x), cfg.bench.warmup,
                                                                    cfg.bench.iters, B)
                P["kernels"]["L3_compiled"] = count_cuda_kernels(cm, x)
                print(f"[{prec}] L3_compiled   {P['whole_model']['L3_compiled']['mean_ms']:.3f} ms  "
                      f"kernels/forward={P['kernels']['L3_compiled']}")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] torch.compile failed: {e}")

        if not args.no_trt:
            try:
                from sparkprof.trt_engine import TRTRunner, build_engine, export_onnx
                edir = Path(cfg.paths.engine_dir) / arch
                onnx = edir / "fp32.onnx"
                if not onnx.exists():
                    export_onnx(base.to("cuda"), onnx, res)
                eng = edir / f"{prec}.engine"
                if not eng.exists():
                    build_engine(onnx, eng, prec, res, 1, cfg.trt.opt_batch, cfg.trt.max_batch,
                                 cfg.trt.workspace_gb, timing_cache=edir / "timing.cache")
                r = TRTRunner(eng)
                P["whole_model"]["TRT"] = r.time(B, res, cfg.bench.warmup, cfg.bench.iters)
                P["trt_layers"] = r.layer_times(B, res, iters=30)
                print(f"[{prec}] TRT           {P['whole_model']['TRT']['mean_ms']:.3f} ms  "
                      f"engine layers={len(P['trt_layers'])}")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] TensorRT step skipped: {e}")

        _plot_units(P, arch, prec, Path(cfg.paths.report_dir) / f"fusion_{arch}_{prec}.png")

    # torch.compile objects / tensors are not JSON; per-layer rows are plain dicts
    save_json(out, f"{cfg.paths.results_dir}/fusion_{arch}.json")
    print(f"wrote {cfg.paths.results_dir}/fusion_{arch}.json")


def _plot_units(P, arch, prec, path):
    plt = plotstyle.apply()
    u0, u2 = P["units"]["L0_unfused"], P["units"]["L2_fused"]
    keys = sorted((k for k in u0 if k in u2), key=lambda k: -u0[k])[:20]
    if not keys:
        return
    import numpy as np
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(keys) + 1.5))
    ax.barh(y - 0.2, [u0[k] for k in keys], 0.38, label="unfused (Conv, BN, ReLU)",
            color=plotstyle.SERIES[0])
    ax.barh(y + 0.2, [u2[k] for k in keys], 0.38, label="fused (Conv+BN+ReLU)",
            color=plotstyle.SERIES[1])
    ax.set_yticks(y, keys, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("time per forward [ms]")
    ax.set_title(f"{arch} {prec}: 20 slowest fusion units, before vs after", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
