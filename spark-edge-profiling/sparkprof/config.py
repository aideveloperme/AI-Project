"""YAML config loading with dotted command-line overrides and shared CLI helpers."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Cfg(dict):
    """dict with attribute access (cfg.bench.iters)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_override(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"override must look like a.b=value, got {assignment!r}")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, Cfg())
    node[parts[-1]] = _wrap(_parse_value(raw))


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Cfg:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = _wrap(yaml.safe_load(f))
    for o in overrides or []:
        apply_override(cfg, o)
    # resolve paths relative to the project root
    for k, v in cfg.paths.items():
        p = Path(v)
        cfg.paths[k] = str(p if p.is_absolute() else PROJECT_ROOT / p)
        Path(cfg.paths[k]).mkdir(parents=True, exist_ok=True)
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="override a config key, e.g. --set bench.iters=100")
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test mode: tiny iteration counts and data subsets")
    return ap


QUICK_OVERRIDES = [
    "bench.warmup=5", "bench.iters=20", "bench.batch_sizes=[1,8]", "bench.power_seconds=3",
    "train.epochs=1", "train.qat_epochs=1", "train.surgery_epochs=1",
    "data.eval_limit=256", "data.calib_images=64", "quant.sensitivity_batches=1",
    "surgery.resolutions=[160,224]",
]


def config_from_args(args: argparse.Namespace) -> Cfg:
    overrides = (QUICK_OVERRIDES if args.quick else []) + list(args.overrides)
    return load_config(args.config, overrides)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | os.PathLike, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with open(path) as f:
        return json.load(f)
