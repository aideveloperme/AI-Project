#!/usr/bin/env python3
"""Step 3 — FP32 vs FP16 vs INT8 PTQ vs INT8 QAT vs mixed precision (accuracy, latency, power, memory).

Variants (TensorRT unless marked eager):
  eager_fp32 / eager_fp16   PyTorch reference (cuDNN, no TensorRT)
  trt_fp32                  TF32 disabled -> true FP32 math
  trt_fp16                  FP16 kernels, FP32 accumulation where TRT chooses
  trt_bf16                  (optional) BF16
  trt_int8_ptq              ModelOpt max-calibrated INT8, explicit Q/DQ
  trt_int8_qat              PTQ init + quantisation-aware fine-tune
  trt_mixed                 INT8 except the top-k most sensitive layers kept FP16
  trt_fp8_ptq               (optional, --fp8) Blackwell FP8 E4M3
"""

from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sparkprof.bench import bench_eager, bench_engine, weights_mb
from sparkprof.config import base_parser, config_from_args, save_json, seed_everything
from sparkprof.data import calibration_loader, get_loader
from sparkprof.models import load_or_build
from sparkprof.train import evaluate, fit
from sparkprof.trt_engine import build_engine, export_onnx

ALL = ["eager_fp32", "eager_fp16", "trt_fp32", "trt_fp16", "trt_int8_ptq", "trt_mixed", "trt_int8_qat"]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", default=None, help="architecture (default models.cnn)")
    ap.add_argument("--variants", nargs="*", default=ALL)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild engines even if cached")
    ap.add_argument("--no-power", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(0)
    arch = args.model or cfg.models.cnn
    res = cfg.data.resolution
    variants = list(args.variants) + (["trt_bf16"] if args.bf16 else []) + (["trt_fp8_ptq"] if args.fp8 else [])
    dev = torch.device("cuda")
    edir = Path(cfg.paths.engine_dir) / arch
    tcache = edir / "timing.cache"
    power = not args.no_power

    model = load_or_build(cfg, arch).to(dev).eval()
    val = get_loader(cfg, "val", limit=cfg.data.eval_limit, shuffle=False)
    calib = calibration_loader(cfg)
    results: dict = {"arch": arch, "resolution": res, "variants": {}}
    n_params = sum(p.numel() for p in model.parameters())

    def engine(tag, precision, onnx):
        p = edir / f"{tag}.engine"
        if args.rebuild or not p.exists():
            print(f"  building {p.name} ...", flush=True)
            meta = build_engine(onnx, p, precision, res, 1, cfg.trt.opt_batch, cfg.trt.max_batch,
                                cfg.trt.workspace_gb, cfg.trt.builder_optimization_level,
                                timing_cache=tcache)
            print(f"  built in {meta['build_s']:.0f}s ({meta['engine_mb']:.1f} MB)")
        return p

    def record(name, r, **extra):
        r.update(extra)
        results["variants"][name] = r
        lat1 = r["latency"].get(1, r["latency"].get("1", {}))
        print(f"[{name:14s}] top1 {r.get('accuracy', {}).get('top1', float('nan')):6.2f}%  "
              f"bs1 {lat1.get('mean_ms', float('nan')):.3f} ms  "
              f"power {((r.get('power') or {}).get('avg_w') or float('nan')):.1f} W", flush=True)
        save_json(results, f"{cfg.paths.results_dir}/precision_{arch}.json")

    # ---------------------------------------------------------------- eager references
    for prec in ("fp32", "fp16"):
        if f"eager_{prec}" in variants:
            r = bench_eager(model, cfg, res, prec, power=power)
            r["accuracy"] = evaluate(model.to(torch.float16 if prec == "fp16" else torch.float32),
                                     val, dev, dtype=torch.float16 if prec == "fp16" else None)
            model.float()
            record(f"eager_{prec}", r, runtime="pytorch", precision=prec)

    # ---------------------------------------------------------------- float engines
    onnx_fp = edir / "fp32.onnx"
    if any(v in variants for v in ("trt_fp32", "trt_fp16", "trt_bf16")):
        if args.rebuild or not onnx_fp.exists():
            export_onnx(model, onnx_fp, res)
        for prec in ("fp32", "fp16", "bf16"):
            if f"trt_{prec}" in variants:
                r = bench_engine(engine(f"{prec}", prec, onnx_fp), cfg, res, val, power)
                r["memory"]["weights_mb"] = n_params * {"fp32": 4, "fp16": 2, "bf16": 2}[prec] / 1e6
                record(f"trt_{prec}", r, runtime="tensorrt", precision=prec)

    # ---------------------------------------------------------------- INT8 PTQ / mixed / QAT
    need_q = any(v in variants for v in ("trt_int8_ptq", "trt_mixed", "trt_int8_qat"))
    if need_q:
        from sparkprof.quant import make_mixed, ptq, sensitivity

        qmodel = ptq(model, calib, "int8", dev)
        fq_acc = evaluate(qmodel, val, dev)
        print(f"  fake-quant INT8 PTQ top-1 (PyTorch): {fq_acc['top1']:.2f}%")

        if "trt_int8_ptq" in variants:
            onnx = export_onnx(qmodel, edir / "int8_ptq.onnx", res)
            r = bench_engine(engine("int8_ptq", "int8_qdq", onnx), cfg, res, val, power)
            r["memory"]["weights_mb"] = n_params / 1e6
            record("trt_int8_ptq", r, runtime="tensorrt", precision="int8", fakequant_top1=fq_acc["top1"])

        if "trt_mixed" in variants:
            sens = sensitivity(qmodel, model, calib, cfg.quant.sensitivity_batches, dev)
            keep = [s["layer"] for s in sens[: cfg.quant.mixed_keep_fp16_topk]]
            print(f"  most quantisation-sensitive layers (kept FP16): {keep}")
            mixed = make_mixed(qmodel, keep)
            m_acc = evaluate(mixed, val, dev)
            onnx = export_onnx(mixed, edir / "mixed.onnx", res)
            r = bench_engine(engine("mixed", "int8_qdq", onnx), cfg, res, val, power)
            kept_params = sum(p.numel() for n, mod in mixed.named_modules() if n in keep
                              for p in mod.parameters(recurse=False))
            r["memory"]["weights_mb"] = (n_params - kept_params + 2 * kept_params) / 1e6
            record("trt_mixed", r, runtime="tensorrt", precision="int8+fp16",
                   fp16_layers=keep, sensitivity=sens, fakequant_top1=m_acc["top1"])

        if "trt_int8_qat" in variants:
            qat = ptq(model, calib, "int8", dev)
            train = get_loader(cfg, "train")
            hist = fit(qat, train, val, cfg.train.qat_epochs, cfg.train.qat_lr,
                       cfg.train.weight_decay, cfg.train.label_smoothing, amp=False,
                       log_prefix="[QAT] ")
            onnx = export_onnx(qat.eval(), edir / "int8_qat.onnx", res)
            r = bench_engine(engine("int8_qat", "int8_qdq", onnx), cfg, res, val, power)
            r["memory"]["weights_mb"] = n_params / 1e6
            record("trt_int8_qat", r, runtime="tensorrt", precision="int8", qat_history=hist)

    if "trt_fp8_ptq" in variants:
        from sparkprof.quant import ptq
        q8 = ptq(model, calib, "fp8", dev)
        onnx = export_onnx(q8, edir / "fp8_ptq.onnx", res)
        r = bench_engine(engine("fp8_ptq", "fp8_qdq", onnx), cfg, res, val, power)
        r["memory"]["weights_mb"] = n_params / 1e6
        record("trt_fp8_ptq", r, runtime="tensorrt", precision="fp8")

    results["weights_fp32_mb"] = weights_mb(model)
    save_json(results, f"{cfg.paths.results_dir}/precision_{arch}.json")
    print(f"\nwrote {cfg.paths.results_dir}/precision_{arch}.json")


if __name__ == "__main__":
    main()
