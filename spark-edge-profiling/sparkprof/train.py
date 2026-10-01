"""Fine-tuning and evaluation loops (shared by baseline training, QAT and surgery recovery)."""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate(model: nn.Module, loader, dev=None, dtype: torch.dtype | None = None,
             channels_last: bool = False, max_batches: int | None = None) -> dict:
    dev = dev or device()
    model = model.to(dev).eval()
    correct = total = 0
    loss_sum = 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if dtype is not None:
            x = x.to(dtype)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"top1": 100.0 * correct / max(total, 1), "loss": loss_sum / max(total, 1), "n": total}


def fit(model: nn.Module, train_loader, val_loader, epochs: int, lr: float,
        weight_decay: float = 0.05, label_smoothing: float = 0.1, amp: bool = True,
        dev=None, log_prefix: str = "") -> list[dict]:
    """AdamW + cosine schedule with 1-epoch-fraction warmup. Returns per-epoch history."""
    dev = dev or device()
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(train_loader))
    warm = max(1, min(200, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = amp and dev.type == "cuda"
    history = []
    for ep in range(epochs):
        model.train()
        t0, seen, loss_acc = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            loss_acc += loss.item() * y.numel()
            seen += y.numel()
        val = evaluate(model, val_loader, dev)
        rec = {"epoch": ep + 1, "train_loss": loss_acc / max(seen, 1), "val_top1": val["top1"],
               "val_loss": val["loss"], "sec": time.time() - t0}
        history.append(rec)
        print(f"{log_prefix}epoch {ep + 1}/{epochs}  loss {rec['train_loss']:.3f}  "
              f"val top1 {rec['val_top1']:.2f}%  ({rec['sec']:.0f}s)", flush=True)
    return history
