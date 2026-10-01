#!/usr/bin/env python3
"""Step 1 — measure achievable DRAM bandwidth and GEMM throughput (the roofline ceilings)."""

import _bootstrap  # noqa: F401
import torch

from sparkprof.config import base_parser, config_from_args, save_json
from sparkprof.hardware import measure_bandwidth, measure_gemm


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--buffer-mb", type=int, default=2048)
    args = ap.parse_args()
    cfg = config_from_args(args)
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    n = 2048 if args.quick else args.gemm_n
    bw = measure_bandwidth(256 if args.quick else args.buffer_mb)
    gemm = measure_gemm(n)
    nom = cfg.hardware.nominal
    res = {"bandwidth": bw, "gemm": gemm, "gemm_n": n, "nominal": dict(nom)}
    print(f"DRAM bandwidth  copy {bw['copy_gbs']:.0f}  read {bw['read_gbs']:.0f}  "
          f"triad {bw['triad_gbs']:.0f} GB/s   (nominal {nom.dram_bw_gbs} GB/s, "
          f"{100 * bw['best_gbs'] / nom.dram_bw_gbs:.0f}% achieved)")
    for k, v in gemm.items():
        if k.endswith(("tflops", "tops")) and v:
            print(f"GEMM {k:12s} {v:8.1f}")
    for prec, key in (("fp32", "fp32_tflops"), ("fp16", "fp16_tflops"), ("int8", "int8_tops")):
        tf = gemm.get(key)
        if tf:
            print(f"ridge point {prec}: {tf * 1e3 / bw['best_gbs']:.0f} FLOP/byte")
    save_json(res, f"{cfg.paths.results_dir}/hw_peaks.json")


if __name__ == "__main__":
    main()
