#!/usr/bin/env python3
"""Step 2 — fine-tune ImageNet-pretrained models on Imagenette (FP32 reference checkpoints)."""

import _bootstrap  # noqa: F401

from sparkprof.config import base_parser, config_from_args, save_json, seed_everything
from sparkprof.data import get_loader
from sparkprof.models import build_model, count_params, save_checkpoint
from sparkprof.train import evaluate, fit


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="architectures (default: models.cnn, models.silu, models.vit)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(0)
    archs = args.models or list(dict.fromkeys([cfg.models.cnn, cfg.models.silu, cfg.models.vit]))
    train = get_loader(cfg, "train")
    val = get_loader(cfg, "val", limit=cfg.data.eval_limit)
    for arch in archs:
        print(f"\n=== fine-tuning {arch} ===")
        model = build_model(arch, cfg.data.num_classes, pretrained=True)
        hist = fit(model, train, val, cfg.train.epochs, cfg.train.lr, cfg.train.weight_decay,
                   cfg.train.label_smoothing, cfg.train.amp, log_prefix=f"[{arch}] ")
        acc = evaluate(model, val)
        save_checkpoint(model.cpu(), cfg, arch, {"val": acc})
        save_json({"arch": arch, "params_m": count_params(model) / 1e6, "history": hist, "val": acc},
                  f"{cfg.paths.results_dir}/train_{arch}.json")
        print(f"[{arch}] final top-1 {acc['top1']:.2f}%")


if __name__ == "__main__":
    main()
