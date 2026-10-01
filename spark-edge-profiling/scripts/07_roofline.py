#!/usr/bin/env python3
"""Step 7 — per-layer roofline analysis for every model (FP32 and FP16, eager).

For each layer: analytical FLOPs and compulsory bytes, measured latency, attained
TFLOP/s, achieved GB/s, compute- vs memory-bound classification against the measured
ceilings from step 1. If step 8 (Nsight Compute) results exist, measured DRAM bytes
are used instead of the compulsory minimum ("ncu" series in the plot).
"""

from pathlib import Path

import _bootstrap  # noqa: F401

from sparkprof.bench import per_layer_profile
from sparkprof.config import base_parser, config_from_args, load_json, save_json
from sparkprof.hardware import peaks
from sparkprof.models import load_or_build
from sparkprof.roofline import classify, plot_roofline, summary


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    hw = load_json(f"{cfg.paths.results_dir}/hw_peaks.json")
    if hw is None:
        print("[warn] results/hw_peaks.json missing — using nominal peaks (run step 1 first)")
    archs = args.models or list(dict.fromkeys([cfg.models.cnn, cfg.models.silu, cfg.models.vit]))
    res = cfg.data.resolution
    for arch in archs:
        model = load_or_build(cfg, arch).eval()
        out = {"arch": arch, "batch": args.batch, "precisions": {}}
        for prec in args.precisions:
            tf, bw = peaks(cfg, hw, prec)
            rows = per_layer_profile(model, res, args.batch, prec, channels_last=arch != cfg.models.vit,
                                     iters=max(20, cfg.bench.iters // 5))
            classify(rows, tf, bw)
            series = {f"{prec} (compulsory bytes)": rows}
            ncu = load_json(f"{cfg.paths.results_dir}/ncu_{arch}_{prec}.json")
            if ncu:
                measured = {d["layer"]: d for d in ncu.get("layers", []) if d.get("dram_bytes")}
                nrows = []
                for r in rows:
                    if r["layer"] in measured:
                        n = dict(r)
                        n["ncu_bytes"] = measured[r["layer"]]["dram_bytes"]
                        r["ncu_bytes"] = n["ncu_bytes"]
                        r["traffic_ratio"] = n["ncu_bytes"] / r["bytes"] if r.get("bytes") else None
                        nrows.append(n)
                classify(nrows, tf, bw, bytes_key="ncu_bytes")
                series[f"{prec} (Nsight DRAM bytes)"] = nrows
            nom_key = {"fp32": "fp32_tflops", "fp16": "fp16_tflops"}[prec]
            plot_roofline(series, tf, bw, str(Path(cfg.paths.report_dir) / f"roofline_{arch}_{prec}.png"),
                          f"{arch} {prec} batch {args.batch} — per-layer roofline (GB10)",
                          nominal=(cfg.hardware.nominal[nom_key], cfg.hardware.nominal.dram_bw_gbs))
            s = summary(rows)
            out["precisions"][prec] = {"peak_tflops": tf, "peak_gbs": bw, "ridge": tf * 1e3 / bw,
                                       "summary": s, "rows": rows}
            print(f"[{arch} {prec}] {s['total_gflops']:.1f} GFLOP, ridge {tf * 1e3 / bw:.0f} FLOP/B: "
                  f"{s['memory_bound_layers']}/{s['layers']} layers memory-bound "
                  f"({s['memory_bound_time_pct']:.0f}% of time)")
        save_json(out, f"{cfg.paths.results_dir}/roofline_{arch}.json")


if __name__ == "__main__":
    main()
