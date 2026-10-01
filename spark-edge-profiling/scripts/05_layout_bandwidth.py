#!/usr/bin/env python3
"""Step 5 — memory layouts (NCHW vs NHWC), tiling, and bandwidth bottlenecks.

  A. PyTorch eager NCHW vs NHWC (channels_last), FP32 and FP16: whole-model and per-layer
     latency; NHWC lets cuDNN pick tensor-core kernels without NCHW<->NHWC transposes.
  B. TensorRT: network input in LINEAR (NCHW) vs HWC8 (vectorised NHWC, FP16) format —
     the difference is the reformat layer TRT must insert — and the Blackwell tiling
     optimisation levels (TRT >= 10.8).
  C. Triton GEMM tile sweep: how tile size sets arithmetic intensity and DRAM traffic.
  D. Bandwidth bottlenecks: per-layer achieved GB/s vs the measured DRAM ceiling; layers
     below the ridge point that run near the bandwidth roof are memory-bound.
"""

from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from sparkprof import plotstyle
from sparkprof.bench import bench_eager, per_layer_profile
from sparkprof.config import base_parser, config_from_args, load_json, save_json
from sparkprof.hardware import peaks
from sparkprof.models import load_or_build
from sparkprof.roofline import classify, summary


def part_a(cfg, model, arch, B, out):
    res = cfg.data.resolution
    A = out["A_eager_layout"] = {}
    for prec in ("fp32", "fp16"):
        for cl in (False, True):
            tag = f"{prec}_{'nhwc' if cl else 'nchw'}"
            r = bench_eager(model, cfg, res, prec, channels_last=cl, batch_sizes=[1, B])
            rows = per_layer_profile(model, res, B, prec, channels_last=cl,
                                     iters=max(20, cfg.bench.iters // 5))
            A[tag] = {"latency": r["latency"], "per_layer": rows}
            print(f"[A] {arch} {tag}: bs1 {r['latency'][1]['mean_ms']:.3f} ms  "
                  f"bs{B} {r['latency'][B]['mean_ms']:.3f} ms")
    # per-layer speedup NHWC/NCHW in FP16, biggest movers first
    nchw = {r["layer"]: r for r in A["fp16_nchw"]["per_layer"]}
    movers = []
    for r in A["fp16_nhwc"]["per_layer"]:
        b = nchw.get(r["layer"])
        if b and r["mean_ms"] > 0:
            movers.append({"layer": r["layer"], "type": r["type"], "nchw_ms": b["mean_ms"],
                           "nhwc_ms": r["mean_ms"], "speedup": b["mean_ms"] / r["mean_ms"]})
    A["fp16_layer_movers"] = sorted(movers, key=lambda d: -abs(d["nchw_ms"] - d["nhwc_ms"]))[:25]


def part_b(cfg, model, arch, B, out):
    from sparkprof.trt_engine import TRTRunner, build_engine, export_onnx
    res = cfg.data.resolution
    edir = Path(cfg.paths.engine_dir) / arch / "layout"
    onnx = Path(cfg.paths.engine_dir) / arch / "fp32.onnx"
    if not onnx.exists():
        export_onnx(model.to("cuda"), onnx, res)
    Bo = out["B_trt"] = {"io_format": {}, "tiling": {}}
    for fmt in ("LINEAR", "HWC8"):
        eng = edir / f"fp16_{fmt}.engine"
        try:
            build_engine(onnx, eng, "fp16", res, 1, cfg.trt.opt_batch, cfg.trt.max_batch,
                         cfg.trt.workspace_gb, io_format=fmt)
            r = TRTRunner(eng)
            Bo["io_format"][fmt] = {"latency": {b: r.time(b, res, cfg.bench.warmup, cfg.bench.iters)
                                                for b in (1, B)},
                                    "layers": [l["layer"] for l in r.layer_times(B, res, iters=5)]}
            print(f"[B] TRT fp16 input {fmt:6s}: bs1 {Bo['io_format'][fmt]['latency'][1]['mean_ms']:.3f} ms")
        except Exception as e:  # noqa: BLE001
            Bo["io_format"][fmt] = {"error": str(e)[:300]}
            print(f"[B] {fmt} failed: {e}")
    for lvl in cfg.trt.tiling_levels:
        eng = edir / f"fp16_tiling_{lvl}.engine"
        try:
            meta = build_engine(onnx, eng, "fp16", res, 1, cfg.trt.opt_batch, cfg.trt.max_batch,
                                cfg.trt.workspace_gb, tiling_level=lvl)
            if meta["tiling_level"] is None:
                Bo["tiling"]["unsupported"] = "this TensorRT has no tiling_optimization_level"
                print("[B] tiling optimisation not supported by this TensorRT version")
                break
            r = TRTRunner(eng)
            Bo["tiling"][lvl] = {"build_s": meta["build_s"],
                                 "latency": {b: r.time(b, res, cfg.bench.warmup, cfg.bench.iters)
                                             for b in (1, B)}}
            print(f"[B] TRT tiling {lvl:8s}: bs1 {Bo['tiling'][lvl]['latency'][1]['mean_ms']:.3f} ms "
                  f"(build {meta['build_s']:.0f}s)")
        except Exception as e:  # noqa: BLE001
            Bo["tiling"][lvl] = {"error": str(e)[:300]}


def part_c(cfg, out, n: int):
    from sparkprof.tiling import DEFAULT_TILES, HAVE_TRITON, model_traffic_bytes, tiled_matmul
    from sparkprof.timing import time_callable
    if not (HAVE_TRITON and torch.cuda.is_available()):
        print("[C] needs triton and a CUDA GPU — skipping tile sweep")
        return
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    ref = a @ b
    C = out["C_tiling"] = {"n": n, "cublas": time_callable(lambda: a @ b, 10, 50)["mean_ms"], "tiles": []}
    for bm, bn, bk in DEFAULT_TILES:
        try:
            c = tiled_matmul(a, b, bm, bn, bk)
            err = (c.float() - ref.float()).abs().max().item()
            ms = time_callable(lambda: tiled_matmul(a, b, bm, bn, bk), 5, 30)["mean_ms"]
            traffic = model_traffic_bytes(n, n, n, bm, bn)
            rec = {"tile": f"{bm}x{bn}x{bk}", "bm": bm, "bn": bn, "bk": bk, "ms": ms,
                   "tflops": 2 * n ** 3 / ms / 1e9, "model_traffic_gb": traffic / 1e9,
                   "model_ai": 2 * n ** 3 / traffic, "max_err": err}
            C["tiles"].append(rec)
            print(f"[C] tile {rec['tile']:12s} {ms:7.3f} ms  {rec['tflops']:6.1f} TFLOP/s  "
                  f"model AI {rec['model_ai']:7.1f} FLOP/B")
        except Exception as e:  # noqa: BLE001  (e.g. tile too large for shared memory)
            C["tiles"].append({"tile": f"{bm}x{bn}x{bk}", "error": str(e)[:200]})
    print(f"[C] cuBLAS reference {C['cublas']:.3f} ms ({2 * n ** 3 / C['cublas'] / 1e9:.1f} TFLOP/s)")


def part_d(cfg, out, arch):
    hw = load_json(f"{cfg.paths.results_dir}/hw_peaks.json")
    rows = out["A_eager_layout"]["fp16_nhwc"]["per_layer"]
    tf, bw = peaks(cfg, hw, "fp16")
    classify(rows, tf, bw)
    s = summary(rows)
    by_bw = sorted((r for r in rows if r.get("achieved_gbs")), key=lambda r: -r["mean_ms"])
    out["D_bandwidth"] = {"peak_tflops": tf, "peak_gbs": bw, "summary": s,
                          "layers": [{k: r.get(k) for k in ("layer", "type", "mean_ms", "flops", "bytes",
                                                             "ai_used", "achieved_gbs", "bw_util", "bound")}
                                     for r in by_bw]}
    print(f"[D] {s['memory_bound_layers']}/{s['layers']} layers memory-bound, "
          f"{s['memory_bound_time_pct']:.0f}% of runtime")
    _plot_bw(out["D_bandwidth"], arch, Path(cfg.paths.report_dir) / f"bandwidth_{arch}.png")


def _plot_bw(D, arch, path):
    plt = plotstyle.apply()
    rows = [r for r in D["layers"] if r.get("achieved_gbs")][:25]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(rows) + 1.5))
    colors = [plotstyle.SERIES[1] if r["bound"] == "memory" else plotstyle.SERIES[0] for r in rows]
    ax.barh(range(len(rows)), [r["achieved_gbs"] for r in rows], color=colors)
    ax.axvline(D["peak_gbs"], color=plotstyle.TEXT, lw=1.5)
    ax.text(D["peak_gbs"], -0.8, f" measured peak {D['peak_gbs']:.0f} GB/s", fontsize=8)
    ax.set_yticks(range(len(rows)), [f"{r['layer']} ({r['type']})" for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("achieved DRAM bandwidth (compulsory bytes / time) [GB/s]")
    ax.set_title(f"{arch} FP16 NHWC — 25 slowest layers (orange = memory-bound)", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gemm-n", type=int, default=4096)
    ap.add_argument("--parts", default="ABCD")
    args = ap.parse_args()
    cfg = config_from_args(args)
    arch = args.model or cfg.models.cnn
    model = load_or_build(cfg, arch).eval()
    out = {"arch": arch, "batch": args.batch}
    if "A" in args.parts or "D" in args.parts:
        part_a(cfg, model, arch, args.batch, out)
    if "B" in args.parts:
        part_b(cfg, model, arch, args.batch, out)
    if "C" in args.parts:
        part_c(cfg, out, 1024 if args.quick else args.gemm_n)
    if "D" in args.parts:
        part_d(cfg, out, arch)
    save_json(out, f"{cfg.paths.results_dir}/layout_{arch}.json")
    print(f"wrote {cfg.paths.results_dir}/layout_{arch}.json")


if __name__ == "__main__":
    main()
