"""Imagenette data pipeline (10-class ImageNet subset) with a synthetic fallback.

Imagenette is small enough to download and fine-tune on in minutes on DGX Spark,
yet uses real ImageNet images at realistic resolutions, so accuracy deltas
between FP32 / FP16 / INT8 / pruned models are meaningful.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticImages(Dataset):
    """Deterministic random tensors — for latency-only runs without a dataset."""

    def __init__(self, n: int, resolution: int, num_classes: int):
        self.n, self.res, self.nc = n, resolution, num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i)
        return torch.randn(3, self.res, self.res, generator=g), i % self.nc


def _transforms(resolution: int, train: bool):
    from torchvision import transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.35, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(resolution * 256 / 224)),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataset(cfg, split: str, resolution: int | None = None, eval_tf: bool = False) -> Dataset:
    res = resolution or cfg.data.resolution
    if cfg.data.dataset == "synthetic":
        return SyntheticImages(2048 if split == "train" else 512, res, cfg.data.num_classes)
    from torchvision.datasets import Imagenette

    # download=True is a no-op once the extracted folder exists
    return Imagenette(root=cfg.paths.data_root, split=split, size=cfg.data.imagenette_size,
                      download=True, transform=_transforms(res, train=(split == "train") and not eval_tf))


def get_loader(cfg, split: str, resolution: int | None = None, batch_size: int | None = None,
               limit: int | None = None, shuffle: bool | None = None) -> DataLoader:
    ds = get_dataset(cfg, split, resolution)
    if limit is not None and limit < len(ds):
        g = torch.Generator().manual_seed(0)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.data.batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=cfg.data.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=False,
    )


def calibration_loader(cfg, resolution: int | None = None, batch_size: int = 32) -> DataLoader:
    """Training images with *eval* transforms — what INT8 calibration should see."""
    ds = get_dataset(cfg, "train", resolution, eval_tf=True)
    g = torch.Generator().manual_seed(1)
    idx = torch.randperm(len(ds), generator=g)[: cfg.data.calib_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=cfg.data.workers)
