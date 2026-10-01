#!/usr/bin/env python3
"""Step 8 — measure real DRAM (LPDDR5x) traffic per layer with Nsight Compute.

Profiles the top-K slowest layers of each model in isolation and the Triton tile sweep.
Compares measured DRAM bytes with the compulsory minimum: ratio >> 1 means poor reuse
(tiling/L2 misses, layout reformat, im2col materialisation); ratio < 1 means L2 kept
part of the working set from a previous call (weights of small layers fit in L2).

Requires `ncu` and permission to read GPU performance counters (see docs/SETUP_DGX_SPARK.md).
"""

import sys

import _bootstrap  # noqa: F401

from sparkprof import ncu
from sparkprof.config import DEFAULT_CONFIG, base_parser, config_from_args, load_json, save_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if not ncu.available():
        sys.exit("ncu not found — install Nsight Compute or use the NGC container")
    archs = args.models or list(dict.fromkeys([cfg.models.cnn, cfg.models.silu, cfg.models.vit]))
    py = [sys.executable, "-m", "sparkprof.ncu_target", "--config", args.config or str(DEFAULT_CONFIG)]
    for arch in archs:
        roof = load_json(f"{cfg.paths.results_dir}/roofline_{arch}.json")
        if not roof:
            print(f"[skip] {arch}: run 07_roofline.py first to know which layers matter")
            continue
        rows = roof["precisions"][args.precision]["rows"]
        top = sorted(rows, key=lambda r: -r["mean_ms"])[: (3 if args.quick else args.topk)]
        out = {"arch": arch, "precision": args.precision, "layers": []}
        for r in top:
            cmd = py + ["--arch", arch, "--layer", r["layer"], "--precision", args.precision,
                        "--batch", str(args.batch), "--res", str(cfg.data.resolution),
                        "--iters", str(args.iters)]
            if arch != cfg.models.vit:
                cmd.append("--channels-last")
            try:
                agg = ncu.aggregate(ncu.run(cmd), calls=args.iters)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {r['layer']}: {e}")
                continue
            agg.update({"layer": r["layer"], "type": r["type"], "compulsory_bytes": r.get("bytes"),
                        "flops": r.get("flops")})
            agg["traffic_ratio"] = agg["dram_bytes"] / r["bytes"] if r.get("bytes") else None
            out["layers"].append(agg)
            print(f"  {arch} {r['layer']:28s} DRAM {agg['dram_bytes'] / 1e6:8.2f} MB "
                  f"(compulsory {r.get('bytes', 0) / 1e6:7.2f} MB, x{agg['traffic_ratio'] or 0:.2f})  "
                  f"DRAM {agg['dram_pct_peak']:.0f}% / SM {agg['sm_pct_peak']:.0f}% of peak")
            save_json(out, f"{cfg.paths.results_dir}/ncu_{arch}_{args.precision}.json")

    if not args.no_tiles:
        from sparkprof.tiling import DEFAULT_TILES, model_traffic_bytes
        n = 1024 if args.quick else 4096
        tiles = []
        for bm, bn, bk in DEFAULT_TILES:
            cmd = py + ["--tile", f"{bm}x{bn}x{bk}", "--gemm-n", str(n), "--iters", str(args.iters)]
            try:
                agg = ncu.aggregate(ncu.run(cmd), calls=args.iters)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] tile {bm}x{bn}x{bk}: {e}")
                continue
            agg.update({"tile": f"{bm}x{bn}x{bk}", "model_bytes": model_traffic_bytes(n, n, n, bm, bn)})
            tiles.append(agg)
            print(f"  tile {agg['tile']:12s} DRAM {agg['dram_bytes'] / 1e6:8.1f} MB  "
                  f"L2 {agg['l2_bytes'] / 1e6:9.1f} MB  model(no-L2) {agg['model_bytes'] / 1e6:9.1f} MB")
        save_json({"n": n, "tiles": tiles}, f"{cfg.paths.results_dir}/ncu_tiles.json")


if __name__ == "__main__":
    main()
