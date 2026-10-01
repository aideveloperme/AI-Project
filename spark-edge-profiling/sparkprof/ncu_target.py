"""Target program for Nsight Compute: run ONE layer (or one Triton tile config) in isolation.

    ncu --profile-from-start off ... python -m sparkprof.ncu_target --arch resnet50 --layer layer1.0.conv2

The full model runs once to capture the layer's real input (outside the profiled range),
then the layer alone runs `--iters` times between cudaProfilerStart/Stop.
"""

from __future__ import annotations

import argparse

import torch

from .bench import eager_model, make_input
from .config import load_config
from .models import load_or_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arch")
    ap.add_argument("--layer")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--tile", default=None, help="BMxBNxBK: profile the Triton GEMM instead")
    ap.add_argument("--gemm-n", type=int, default=4096)
    a = ap.parse_args()

    if a.tile:
        from .tiling import tiled_matmul
        bm, bn, bk = map(int, a.tile.split("x"))
        x = torch.randn(a.gemm_n, a.gemm_n, device="cuda", dtype=torch.float16)
        y = torch.randn(a.gemm_n, a.gemm_n, device="cuda", dtype=torch.float16)
        tiled_matmul(x, y, bm, bn, bk)  # compile outside the profiled range
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(a.iters):
            tiled_matmul(x, y, bm, bn, bk)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    cfg = load_config(a.config)
    model = eager_model(load_or_build(cfg, a.arch), a.precision, a.channels_last)
    layer = dict(model.named_modules())[a.layer]
    captured = {}
    h = layer.register_forward_pre_hook(lambda m, inp: captured.setdefault("x", inp))
    with torch.inference_mode():
        model(make_input(a.batch, a.res, a.precision, a.channels_last))
        h.remove()
        inp = captured["x"]
        for _ in range(3):
            layer(*inp)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(a.iters):
            layer(*inp)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
