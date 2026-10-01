#!/usr/bin/env python3
"""Step 6 — accelerator-oriented model surgery and what each change buys.

  A. SiLU -> ReLU6 (EfficientNet-B0): accuracy right after the swap, after a short
     recovery fine-tune, and TensorRT FP16 + INT8 latency before/after.
  B. Input resolution sweep (128..256): accuracy vs latency vs GFLOPs.
  C. ViT attention-head pruning: gradient-based head importance, keep 100/75/50 % of
     heads per layer, physically slice qkv/proj, recover with a short fine-tune.

Every variant is costed (GFLOPs, params) and timed through TensorRT so the gains are
what the accelerator actually sees, not just FLOP counts.
"""

from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sparkprof.bench import model_cost
from sparkprof.config import base_parser, config_from_args, load_json, save_json, seed_everything
from sparkprof.data import calibration_loader, get_loader
from sparkprof.models import load_or_build, save_checkpoint
from sparkprof.surgery import head_importance, prune_vit_heads, replace_activation
from sparkprof.train import evaluate, fit


def trt_latency(cfg, model, tag, res, precisions=("fp16",), calib=None):
    """Export + build + time; returns {precision: {batch: stats}}. INT8 uses ModelOpt PTQ."""
    from sparkprof.trt_engine import TRTRunner, build_engine, export_onnx
    edir = Path(cfg.paths.engine_dir) / "surgery"
    out = {}
    for prec in precisions:
        try:
            m = model.to("cuda").eval()
            if prec == "int8":
                from sparkprof.quant import ptq
                m = ptq(m, calib, "int8")
            onnx = export_onnx(m, edir / f"{tag}_{prec}.onnx", res)
            eng = edir / f"{tag}_{prec}_{res}.engine"
            build_engine(onnx, eng, "int8_qdq" if prec == "int8" else prec, res, 1,
                         cfg.trt.opt_batch, cfg.trt.max_batch, cfg.trt.workspace_gb,
                         timing_cache=edir / "timing.cache")
            r = TRTRunner(eng)
            out[prec] = {b: r.time(b, res, cfg.bench.warmup, cfg.bench.iters) for b in cfg.bench.batch_sizes}
        except Exception as e:  # noqa: BLE001
            out[prec] = {"error": str(e)[:300]}
            print(f"  [warn] TRT {tag} {prec}: {e}")
    return out


def _ms(lat, prec, b=1):
    try:
        return lat[prec][b]["mean_ms"]
    except (KeyError, TypeError):
        return float("nan")


def part_a(cfg, out, val, train):
    arch = cfg.models.silu
    base = load_or_build(cfg, arch).eval()
    calib = calibration_loader(cfg)
    relu6, n = replace_activation(base)
    rec = {"arch": arch, "replaced": n}
    rec["baseline"] = {"acc": evaluate(base, val)["top1"], **model_cost(base, cfg.data.resolution),
                       "trt": trt_latency(cfg, base, f"{arch}_silu", cfg.data.resolution, ("fp16", "int8"), calib)}
    rec["relu6_no_ft"] = {"acc": evaluate(relu6, val)["top1"]}
    hist = fit(relu6, train, val, cfg.train.surgery_epochs, cfg.train.lr / 2, cfg.train.weight_decay,
               cfg.train.label_smoothing, cfg.train.amp, log_prefix="[relu6 FT] ")
    save_checkpoint(relu6.cpu(), cfg, f"{arch}_relu6")
    rec["relu6_ft"] = {"acc": evaluate(relu6, val)["top1"], "history": hist,
                       **model_cost(relu6, cfg.data.resolution),
                       "trt": trt_latency(cfg, relu6, f"{arch}_relu6", cfg.data.resolution, ("fp16", "int8"), calib)}
    # INT8 accuracy is where ReLU6 usually pays off most (bounded activation range)
    try:
        from sparkprof.quant import ptq
        rec["baseline"]["int8_fakequant_acc"] = evaluate(ptq(base.cuda(), calib), val)["top1"]
        rec["relu6_ft"]["int8_fakequant_acc"] = evaluate(ptq(relu6.cuda(), calib), val)["top1"]
    except ImportError as e:
        print(f"  [warn] {e}")
    out["A_silu_to_relu6"] = rec
    b, r = rec["baseline"], rec["relu6_ft"]
    print(f"[A] SiLU {b['acc']:.2f}% -> ReLU6 {rec['relu6_no_ft']['acc']:.2f}% (no FT) -> "
          f"{r['acc']:.2f}% (FT).  TRT fp16 bs1 {_ms(b['trt'], 'fp16'):.3f} -> {_ms(r['trt'], 'fp16'):.3f} ms, "
          f"int8 bs1 {_ms(b['trt'], 'int8'):.3f} -> {_ms(r['trt'], 'int8'):.3f} ms")


def part_b(cfg, out):
    arch = cfg.models.silu
    model = load_or_build(cfg, arch).eval()
    rows = []
    for res in cfg.surgery.resolutions:
        val = get_loader(cfg, "val", resolution=res, limit=cfg.data.eval_limit, shuffle=False)
        row = {"resolution": res, "acc": evaluate(model, val)["top1"], **model_cost(model, res),
               "trt": trt_latency(cfg, model, f"{arch}_res{res}", res, ("fp16",))}
        rows.append(row)
        print(f"[B] {arch} @{res}: top1 {row['acc']:.2f}%  {row['gflops']:.2f} GFLOPs  "
              f"TRT fp16 bs1 {_ms(row['trt'], 'fp16'):.3f} ms")
    out["B_resolution"] = {"arch": arch, "rows": rows}


def part_c(cfg, out, val, train):
    arch = cfg.models.vit
    base = load_or_build(cfg, arch).eval()
    dev = torch.device("cuda")
    imp_loader = get_loader(cfg, "train", batch_size=16, limit=512, shuffle=False)
    scores = head_importance(base, imp_loader, dev, batches=16 if not cfg.get("_quick") else 2)
    rows = []
    for keep in cfg.surgery.vit_head_keep_ratios:
        pruned, kept = prune_vit_heads(base, scores, keep) if keep < 1 else (base, None)
        row = {"keep_ratio": keep, "acc_no_ft": evaluate(pruned, val)["top1"],
               **model_cost(pruned, cfg.data.resolution)}
        if keep < 1:
            row["history"] = fit(pruned, train, val, cfg.train.surgery_epochs, cfg.train.lr / 5,
                                 cfg.train.weight_decay, cfg.train.label_smoothing, cfg.train.amp,
                                 log_prefix=f"[heads {keep:.2f} FT] ")
            save_checkpoint(pruned.cpu(), cfg, f"{arch}_heads{int(keep * 100)}")
        row["acc_ft"] = evaluate(pruned, val)["top1"]
        row["trt"] = trt_latency(cfg, pruned, f"{arch}_heads{int(keep * 100)}", cfg.data.resolution, ("fp16",))
        rows.append(row)
        print(f"[C] keep {keep:.2f}: params {row['params_m']:.1f}M  {row['gflops']:.2f} GFLOPs  "
              f"top1 {row['acc_no_ft']:.2f}% -> {row['acc_ft']:.2f}% (FT)  "
              f"TRT fp16 bs1 {_ms(row['trt'], 'fp16'):.3f} ms")
    out["C_head_pruning"] = {"arch": arch, "rows": rows,
                             "importance": {k: v.tolist() for k, v in scores.items()}}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--parts", default="ABC")
    args = ap.parse_args()
    cfg = config_from_args(args)
    cfg["_quick"] = args.quick
    seed_everything(0)
    val = get_loader(cfg, "val", limit=cfg.data.eval_limit, shuffle=False)
    train = get_loader(cfg, "train")
    path = f"{cfg.paths.results_dir}/surgery.json"
    out = load_json(path, {})  # parts can be run separately and merge into one file
    if "A" in args.parts:
        part_a(cfg, out, val, train)
        save_json(out, path)
    if "B" in args.parts:
        part_b(cfg, out)
        save_json(out, path)
    if "C" in args.parts:
        part_c(cfg, out, val, train)
        save_json(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
