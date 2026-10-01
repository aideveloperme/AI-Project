#!/usr/bin/env python3
"""Step 9 — assemble every results/*.json into reports/REPORT.md (+ figures)."""

import datetime as dt
import math
from pathlib import Path

import _bootstrap  # noqa: F401

from sparkprof import plotstyle
from sparkprof.config import base_parser, config_from_args, load_json


def f(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.{nd}f}{unit}"


def lat(entry, b):
    L = (entry or {}).get("latency") or entry or {}
    return L.get(b) or L.get(str(b)) or {}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Report:
    def __init__(self, cfg):
        self.cfg = cfg
        self.R = Path(cfg.paths.results_dir)
        self.D = Path(cfg.paths.report_dir)
        self.md: list[str] = []
        self.findings: list[str] = []

    def j(self, name):
        return load_json(self.R / name)

    def h(self, text):
        self.md.append(f"\n{text}\n")

    def img(self, name, alt):
        if (self.D / name).exists():
            self.md.append(f"\n![{alt}]({name})\n")

    # ------------------------------------------------------------------ sections
    def environment(self):
        env, hw = self.j("env.json"), self.j("hw_peaks.json")
        if not (env or hw):
            return
        self.h("## 1. Platform and measured ceilings")
        if env:
            keys = ["gpu", "compute_capability", "sm_count", "total_memory_gb", "cuda", "cudnn",
                    "torch", "tensorrt", "modelopt", "triton", "machine", "power_backend", "idle_power_w"]
            self.md.append(table(["item", "value"], [[k, env.get(k)] for k in keys if k in env]))
        if hw:
            nom = self.cfg.hardware.nominal
            bw, g = hw["bandwidth"], hw["gemm"]
            rows = [["DRAM bandwidth (GB/s)", nom.dram_bw_gbs, f(bw["best_gbs"], 0),
                     f(100 * bw["best_gbs"] / nom.dram_bw_gbs, 0, "%")]]
            for k, nk in (("fp32_tflops", "fp32_tflops"), ("tf32_tflops", None), ("fp16_tflops", "fp16_tflops"),
                          ("bf16_tflops", None), ("int8_tops", "int8_tops"), ("fp8_tflops", "fp8_tflops")):
                if g.get(k):
                    n = nom.get(nk) if nk else None
                    rows.append([k, n or "–", f(g[k], 1), f(100 * g[k] / n, 0, "%") if n else "–"])
            self.h("Measured with `01_microbench_peaks.py` (copy/read/triad over 2 GB buffers; "
                   f"{hw['gemm_n']}³ GEMMs). These measured values are the roofline ceilings.")
            self.md.append(table(["ceiling", "nominal", "measured", "achieved"], rows))
            fp16 = g.get("fp16_tflops")
            if fp16:
                self.findings.append(
                    f"Ridge point (FP16): **{fp16 * 1e3 / bw['best_gbs']:.0f} FLOP/B** — any layer with "
                    f"lower arithmetic intensity cannot use the tensor cores fully on GB10's "
                    f"{bw['best_gbs']:.0f} GB/s LPDDR5x.")

    def precision(self):
        arch = self.cfg.models.cnn
        P = self.j(f"precision_{arch}.json")
        if not P:
            return
        self.h(f"## 2. Precision comparison — {arch} @ {P['resolution']}px")
        V = P["variants"]
        ref = (V.get("trt_fp32") or V.get("eager_fp32") or {}).get("accuracy", {}).get("top1")
        bmax = max(self.cfg.bench.batch_sizes)
        rows = []
        for name, r in V.items():
            acc = (r.get("accuracy") or {}).get("top1")
            pw = r.get("power") or {}
            m = r.get("memory", {})
            rows.append([name, f(acc), f(acc - ref if acc is not None and ref is not None else None),
                         f(lat(r, 1).get("p50_ms"), 3), f(lat(r, 1).get("p99_ms"), 3),
                         f(lat(r, bmax).get("throughput_ips"), 0), f(pw.get("avg_w"), 1),
                         f(pw.get("mj_per_inference"), 2), f(m.get("weights_mb"), 1),
                         f(m.get("engine_mb"), 1), f(m.get("trt_device_mb") or m.get(f"peak_alloc_mb_b{bmax}"), 1)])
        self.md.append(table(["variant", "top-1 %", "Δ vs FP32", "bs1 p50 ms", "bs1 p99 ms",
                              f"bs{bmax} img/s", "avg W", "mJ/img", "weights MB", "engine MB",
                              "activation/device MB"], rows))
        self.md.append("\n*Power is the GPU power reported by NVML under sustained max-batch load; "
                       "mJ/img = avg W ÷ throughput. Weights MB is the parameter storage at that precision.*")
        mixed = V.get("trt_mixed")
        if mixed and mixed.get("fp16_layers"):
            self.md.append("\n**Mixed precision**: INT8 everywhere except the most quantisation-sensitive "
                           "layers (KL divergence when quantised alone), kept in FP16:\n")
            self.md.append(table(["layer", "KL(FP32‖INT8-only-this-layer)"],
                                 [[s["layer"], f(s["kl"], 5)] for s in mixed["sensitivity"][:10]]))
        self._plot_precision(V, arch, bmax)
        self.img(f"precision_{arch}.png", "precision trade-off")
        # findings
        trt = {k: v for k, v in V.items() if k.startswith("trt") and lat(v, 1)}
        if "trt_fp32" in trt:
            base = lat(trt["trt_fp32"], bmax).get("throughput_ips")
            best = max(trt, key=lambda k: lat(trt[k], bmax).get("throughput_ips", 0))
            sp = lat(trt[best], bmax).get("throughput_ips", 0) / base if base else None
            acc = (trt[best].get("accuracy") or {}).get("top1")
            self.findings.append(f"Fastest engine: **{best}**, {f(sp, 1)}× FP32 throughput at batch {bmax} "
                                 f"with top-1 {f(acc)}% (FP32 {f(ref)}%).")
        if "trt_int8_ptq" in V and "trt_int8_qat" in V:
            a1 = (V["trt_int8_ptq"].get("accuracy") or {}).get("top1")
            a2 = (V["trt_int8_qat"].get("accuracy") or {}).get("top1")
            if a1 is not None and a2 is not None:
                self.findings.append(f"QAT vs PTQ: {f(a2)}% vs {f(a1)}% top-1 ({f(a2 - a1, 2)} pts) at "
                                     "identical latency — QAT matters when PTQ loses > ~0.5 pt.")

    def _plot_precision(self, V, arch, bmax):
        plt = plotstyle.apply()
        pts = [(k, (v.get("accuracy") or {}).get("top1"), lat(v, bmax).get("throughput_ips"),
                (v.get("power") or {}).get("mj_per_inference")) for k, v in V.items()]
        pts = [p for p in pts if p[1] is not None and p[2]]
        if not pts:
            return
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.scatter([p[2] for p in pts], [p[1] for p in pts], s=60, color=plotstyle.SERIES[0],
                   edgecolor=plotstyle.SURFACE, linewidth=2, zorder=3)
        for k, acc, ips, _ in pts:
            a1.annotate(k, (ips, acc), xytext=(5, 4), textcoords="offset points", fontsize=8,
                        color=plotstyle.TEXT2)
        a1.set_xlabel(f"throughput at batch {bmax} [img/s]")
        a1.set_ylabel("top-1 accuracy [%]")
        a1.set_title("Accuracy vs throughput", loc="left")
        e = [p for p in pts if p[3]]
        if e:
            a2.barh([p[0] for p in e], [p[3] for p in e], color=plotstyle.SERIES[0])
            for i, p in enumerate(e):
                a2.text(p[3], i, f" {p[3]:.2f}", va="center", fontsize=8, color=plotstyle.TEXT2)
            a2.invert_yaxis()
            a2.set_xlabel("energy per image [mJ]")
            a2.set_title("Energy per inference", loc="left")
        else:
            a2.set_axis_off()
        fig.tight_layout()
        fig.savefig(self.D / f"precision_{arch}.png")
        plt.close(fig)

    def fusion(self):
        arch = self.cfg.models.cnn
        F = self.j(f"fusion_{arch}.json")
        if not F:
            return
        self.h(f"## 3. Layer fusion: Conv + BN + ReLU — {arch}, batch {F['batch']}")
        self.md.append(table(["level", "Conv", "BN", "ReLU", "fused Conv+BN+ReLU"],
                             [[k, v["conv"], v["bn"], v["relu"], v["fused"]] for k, v in F["ops"].items()]))
        for prec, P in F["precisions"].items():
            self.h(f"### {prec.upper()}")
            base = P["whole_model"].get("L0_unfused", {}).get("mean_ms")
            rows = [[k, f(v["mean_ms"], 3), f(base / v["mean_ms"] if base else None, 2, "×"),
                     P["kernels"].get(k) or "–", f(P["max_abs_diff_vs_L0"].get(k), 6) if k in P["max_abs_diff_vs_L0"] else "–"]
                    for k, v in P["whole_model"].items()]
            self.md.append(table(["level", "latency ms", "speedup vs L0", "CUDA kernels / forward",
                                  "max |Δlogit| vs L0"], rows))
            u0, u2 = P["units"]["L0_unfused"], P["units"]["L2_fused"]
            u1 = P["units"].get("L1_bn_folded", {})
            keys = sorted((k for k in u0 if k in u2), key=lambda k: -u0[k])[:15]
            self.md.append("\nPer-layer timing, 15 slowest fusion units (unit = conv [+bn] [+relu]):\n")
            self.md.append(table(["unit", "unfused ms", "BN-folded ms", "fused ms", "saved"],
                                 [[k, f(u0[k], 4), f(u1.get(k), 4), f(u2[k], 4),
                                   f(100 * (1 - u2[k] / u0[k]) if u0[k] else None, 0, "%")] for k in keys]))
            tot0, tot2 = sum(u0.values()), sum(u2.values())
            self.md.append(f"\nSum over all units: {f(tot0, 3)} ms → {f(tot2, 3)} ms "
                           f"({f(100 * (tot2 / tot0 - 1) if tot0 else None, 0, '%')} change).")
            if P.get("trt_layers"):
                fused = [l for l in P["trt_layers"] if "+" in l["layer"]][:8]
                self.md.append(f"\nTensorRT built **{len(P['trt_layers'])}** engine layers. Examples of fused layers "
                               "(names joined by `+` are single kernels):\n")
                self.md.append(table(["TensorRT layer", "ms"], [[f"`{l['layer'][:110]}`", f(l["mean_ms"], 4)] for l in fused]))
            self.img(f"fusion_{arch}_{prec}.png", f"fusion {prec}")
            if base and "L2_fused" in P["whole_model"]:
                k0, k2 = P["kernels"].get("L0_unfused"), P["kernels"].get("L2_fused")
                kern = f"kernels/forward {k0} → {k2} and " if k0 and k2 else ""
                self.findings.append(
                    f"Fusion ({prec}): Conv+BN+ReLU fusion changes {kern}latency "
                    f"{f(base, 3)} → {f(P['whole_model']['L2_fused']['mean_ms'], 3)} ms.")

    def layout(self):
        arch = self.cfg.models.cnn
        L = self.j(f"layout_{arch}.json")
        if not L:
            return
        B = L["batch"]
        self.h(f"## 4. Memory layout, tiling and bandwidth — {arch}")
        A = L.get("A_eager_layout")
        if A:
            self.h("### 4a. NCHW vs NHWC (PyTorch/cuDNN)")
            rows = [[k, f(lat(v, 1).get("mean_ms"), 3), f(lat(v, B).get("mean_ms"), 3),
                     f(lat(v, B).get("throughput_ips"), 0)] for k, v in A.items() if k != "fp16_layer_movers"]
            self.md.append(table(["config", "bs1 ms", f"bs{B} ms", f"bs{B} img/s"], rows))
            self.md.append("\nLayers whose time changes most when switching FP16 NCHW → NHWC:\n")
            self.md.append(table(["layer", "type", "NCHW ms", "NHWC ms", "speedup"],
                                 [[m["layer"], m["type"], f(m["nchw_ms"], 4), f(m["nhwc_ms"], 4),
                                   f(m["speedup"], 2, "×")] for m in A["fp16_layer_movers"][:12]]))
            n1, n2 = lat(A.get("fp16_nchw"), B).get("mean_ms"), lat(A.get("fp16_nhwc"), B).get("mean_ms")
            if n1 and n2:
                self.findings.append(f"Layout: FP16 NHWC is {f(n1 / n2, 2)}× NCHW at batch {B} — tensor-core "
                                     "convolutions want channels innermost; NCHW pays for transposes.")
        Bt = L.get("B_trt")
        if Bt:
            self.h("### 4b. TensorRT I/O format and tiling optimisation")
            rows = [[k, f(lat(v, 1).get("mean_ms"), 3), f(lat(v, B).get("mean_ms"), 3), v.get("error", "")[:60]]
                    for k, v in Bt["io_format"].items()]
            self.md.append(table(["input format", "bs1 ms", f"bs{B} ms", "note"], rows))
            if Bt["tiling"]:
                rows = [[k, f(lat(v, 1).get("mean_ms"), 3) if isinstance(v, dict) else v,
                         f(lat(v, B).get("mean_ms"), 3) if isinstance(v, dict) else "",
                         f(v.get("build_s"), 0) if isinstance(v, dict) else ""] for k, v in Bt["tiling"].items()]
                self.md.append("\n" + table(["tiling level", "bs1 ms", f"bs{B} ms", "build s"], rows))
        C = L.get("C_tiling")
        if C:
            self.h(f"### 4c. GEMM tile-size sweep (Triton, {C['n']}³ FP16)")
            nt = {t["tile"]: t for t in (self.j("ncu_tiles.json") or {}).get("tiles", [])}
            rows = []
            for t in C["tiles"]:
                if "error" in t:
                    rows.append([t["tile"], "–", "–", "–", "–", "–", t["error"][:50]])
                    continue
                m = nt.get(t["tile"], {})
                rows.append([t["tile"], f(t["ms"], 3), f(t["tflops"], 1), f(t["model_ai"], 0),
                             f(t["model_traffic_gb"] * 1e3, 0), f(m.get("dram_bytes", 0) / 1e6 if m else None, 0), ""])
            self.md.append(table(["tile BMxBNxBK", "ms", "TFLOP/s", "model AI (FLOP/B)", "model DRAM MB (no L2)",
                                  "ncu DRAM MB", "note"], rows))
            self.md.append(f"\ncuBLAS reference: {f(C['cublas'], 3)} ms "
                           f"({f(2 * C['n'] ** 3 / C['cublas'] / 1e9, 1)} TFLOP/s). Small tiles re-read A and B "
                           "many times: traffic ∝ 1/BM + 1/BN, so intensity — and attainable performance "
                           "under the bandwidth roof — rises with tile size until registers/shared memory run out.")
        D = L.get("D_bandwidth")
        if D:
            s = D["summary"]
            self.h("### 4d. Bandwidth bottlenecks (memory-bound layers)")
            self.md.append(f"{s['memory_bound_layers']} of {s['layers']} layers are below the FP16 ridge point "
                           f"and account for **{f(s['memory_bound_time_pct'], 0)}%** of runtime.\n")
            self.md.append(table(["layer", "type", "ms", "AI FLOP/B", "achieved GB/s", "% of peak BW"],
                                 [[r["layer"], r["type"], f(r["ms"], 4), f(r["ai"], 1), f(r["gbs"], 0),
                                   f(100 * r["gbs"] / D["peak_gbs"] if r["gbs"] else None, 0, "%")]
                                  for r in s["top_memory_bound"]]))
            self.img(f"bandwidth_{arch}.png", "bandwidth")
            if s["top_memory_bound"]:
                types = sorted({r["type"] for r in s["top_memory_bound"][:5]})
                self.findings.append(f"Memory-bound work is {f(s['memory_bound_time_pct'], 0)}% of {arch} FP16 "
                                     f"time, dominated by {', '.join(types)} — these benefit from fusion and "
                                     "lower-precision activations, not from more FLOPs.")

    def surgery(self):
        S = self.j("surgery.json")
        if not S:
            return
        self.h("## 5. Model surgery for the accelerator")
        A = S.get("A_silu_to_relu6")
        if A:
            def ms(r, p, b=1):
                return f(lat((r.get("trt") or {}).get(p), b).get("mean_ms"), 3)
            bmax = max(self.cfg.bench.batch_sizes)
            self.h(f"### 5a. SiLU → ReLU6 ({A['arch']}, {A['replaced']} activations replaced)")
            rows = [["SiLU (baseline)", f(A["baseline"]["acc"]), f(A["baseline"].get("int8_fakequant_acc")),
                     ms(A["baseline"], "fp16"), ms(A["baseline"], "fp16", bmax), ms(A["baseline"], "int8"),
                     ms(A["baseline"], "int8", bmax)],
                    ["ReLU6, no fine-tune", f(A["relu6_no_ft"]["acc"]), "–", "–", "–", "–", "–"],
                    ["ReLU6 + fine-tune", f(A["relu6_ft"]["acc"]), f(A["relu6_ft"].get("int8_fakequant_acc")),
                     ms(A["relu6_ft"], "fp16"), ms(A["relu6_ft"], "fp16", bmax), ms(A["relu6_ft"], "int8"),
                     ms(A["relu6_ft"], "int8", bmax)]]
            self.md.append(table(["variant", "top-1 %", "INT8 PTQ top-1 %", "FP16 bs1 ms", f"FP16 bs{bmax} ms",
                                  "INT8 bs1 ms", f"INT8 bs{bmax} ms"], rows))
        Bs = S.get("B_resolution")
        if Bs:
            self.h(f"### 5b. Input resolution ({Bs['arch']})")
            rows = [[r["resolution"], f(r["acc"]), f(r["gflops"], 2), f(r["min_traffic_mb"], 1),
                     f(lat((r.get("trt") or {}).get("fp16"), 1).get("mean_ms"), 3),
                     f(lat((r.get("trt") or {}).get("fp16"), max(self.cfg.bench.batch_sizes)).get("throughput_ips"), 0)]
                    for r in Bs["rows"]]
            self.md.append(table(["res", "top-1 %", "GFLOPs", "min traffic MB", "FP16 bs1 ms",
                                  f"FP16 bs{max(self.cfg.bench.batch_sizes)} img/s"], rows))
            self._plot_res(Bs)
            self.img("resolution.png", "resolution sweep")
        C = S.get("C_head_pruning")
        if C:
            self.h(f"### 5c. Attention-head pruning ({C['arch']})")
            rows = [[f"{int(r['keep_ratio'] * 100)}%", f(r["params_m"], 1), f(r["gflops"], 2), f(r["acc_no_ft"]),
                     f(r["acc_ft"]), f(lat((r.get("trt") or {}).get("fp16"), 1).get("mean_ms"), 3),
                     f(lat((r.get("trt") or {}).get("fp16"), max(self.cfg.bench.batch_sizes)).get("throughput_ips"), 0)]
                    for r in C["rows"]]
            self.md.append(table(["heads kept", "params M", "GFLOPs", "top-1 no FT", "top-1 after FT",
                                  "FP16 bs1 ms", f"FP16 bs{max(self.cfg.bench.batch_sizes)} img/s"], rows))
            self._plot_heads(C)
            self.img("head_importance.png", "head importance")

    def _plot_res(self, Bs):
        plt = plotstyle.apply()
        rows = [r for r in Bs["rows"] if lat((r.get("trt") or {}).get("fp16"), 1).get("mean_ms")]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = [lat(r["trt"]["fp16"], 1)["mean_ms"] for r in rows]
        ax.plot(x, [r["acc"] for r in rows], marker="o", ms=8, color=plotstyle.SERIES[0])
        for r, xi in zip(rows, x):
            ax.annotate(f"{r['resolution']}px", (xi, r["acc"]), xytext=(5, -12), textcoords="offset points",
                        fontsize=8, color=plotstyle.TEXT2)
        ax.set_xlabel("TensorRT FP16 latency, batch 1 [ms]")
        ax.set_ylabel("top-1 [%]")
        ax.set_title(f"{Bs['arch']}: resolution vs accuracy vs latency", loc="left")
        fig.tight_layout()
        fig.savefig(self.D / "resolution.png")
        plt.close(fig)

    def _plot_heads(self, C):
        plt = plotstyle.apply()
        import numpy as np
        imp = C.get("importance")
        if not imp:
            return
        mat = np.array([imp[k] for k in sorted(imp, key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("encoder layer")
        ax.grid(False)
        ax.set_title("Head importance |∂L/∂gate| (normalised per layer)", loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(self.D / "head_importance.png")
        plt.close(fig)

    def roofline(self):
        archs = list(dict.fromkeys([self.cfg.models.cnn, self.cfg.models.silu, self.cfg.models.vit]))
        any_ = False
        for arch in archs:
            Rf = self.j(f"roofline_{arch}.json")
            if not Rf:
                continue
            if not any_:
                self.h("## 6. Roofline analysis")
                self.md.append(
                    "Each point is one layer: x = arithmetic intensity (FLOPs per byte of DRAM traffic), "
                    "y = attained TFLOP/s, size = share of runtime. The diagonal is the bandwidth roof "
                    "(AI × GB/s), the flat line the compute roof; they meet at the ridge point. Points left of "
                    "the ridge are **memory-bound** (speed them up by moving fewer bytes: fusion, lower "
                    "precision activations, better layout/tiling); points right of it are **compute-bound** "
                    "(speed them up with tensor-core precisions: FP16 → INT8/FP8, or fewer FLOPs: pruning, "
                    "lower resolution). Distance below the roof is inefficiency (launch overhead, small "
                    "grids at batch 1, poor kernels).")
                any_ = True
            self.h(f"### {arch}")
            rows = []
            for prec, P in Rf["precisions"].items():
                s = P["summary"]
                rows.append([prec, f(s["total_gflops"] / Rf["batch"], 2), f(P["ridge"], 0),
                             f"{s['memory_bound_layers']}/{s['layers']}",
                             f(s["memory_bound_time_pct"], 0, "%"), f(s["compute_bound_time_pct"], 0, "%")])
            self.md.append(f"Batch {Rf['batch']}.\n")
            self.md.append(table(["precision", "GFLOP / image", "ridge FLOP/B", "memory-bound layers",
                                  "time memory-bound", "time compute-bound"], rows))
            for prec in Rf["precisions"]:
                self.img(f"roofline_{arch}_{prec}.png", f"roofline {arch} {prec}")
            top = Rf["precisions"].get("fp16", next(iter(Rf["precisions"].values())))["summary"]["top_memory_bound"]
            if top:
                self.md.append("\nSlowest memory-bound layers:\n")
                self.md.append(table(["layer", "type", "ms", "AI FLOP/B", "achieved GB/s"],
                                     [[t["layer"], t["type"], f(t["ms"], 4), f(t["ai"], 1), f(t["gbs"], 0)]
                                      for t in top[:8]]))
            ncu = self.j(f"ncu_{arch}_fp16.json")
            if ncu and ncu.get("layers"):
                self.md.append("\nMeasured DRAM traffic (Nsight Compute) vs compulsory minimum, FP16:\n")
                self.md.append(table(["layer", "type", "compulsory MB", "DRAM MB", "ratio", "L2 MB",
                                      "DRAM % peak", "SM % peak"],
                                     [[l["layer"], l["type"], f((l.get("compulsory_bytes") or 0) / 1e6, 2),
                                       f(l["dram_bytes"] / 1e6, 2), f(l.get("traffic_ratio"), 2, "×"),
                                       f(l["l2_bytes"] / 1e6, 1), f(l["dram_pct_peak"], 0, "%"),
                                       f(l["sm_pct_peak"], 0, "%")] for l in ncu["layers"]]))

    def render(self):
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        head = [f"# Inference profiling & optimisation report — {self.cfg.hardware.name}",
                f"*Generated {now} by `scripts/09_make_report.py` from `results/`.*"]
        self.environment()
        self.precision()
        self.fusion()
        self.layout()
        self.surgery()
        self.roofline()
        summary = ["\n## Key findings\n"] + [f"- {x}" for x in self.findings] if self.findings else []
        text = "\n".join(head + summary + self.md + [
            "\n## Method notes\n",
            "- Latency: CUDA events around each inference after warm-up; p50/p99 over "
            f"{self.cfg.bench.iters} iterations. TensorRT timings exclude host↔device copies "
            "(GB10 memory is unified, input already resident).",
            "- Accuracy: Imagenette validation split (10 ImageNet classes), models fine-tuned from ImageNet weights.",
            "- Power: NVML GPU power sampled at "
            f"{self.cfg.bench.power_sample_hz} Hz during {self.cfg.bench.power_seconds} s of saturated max-batch inference.",
            "- Bytes for the roofline: compulsory (each input, weight and output touched once) unless "
            "Nsight Compute DRAM measurements are available.",
            "- Per-layer eager timings use forward-hook CUDA events; at batch 1 small layers are partly "
            "launch-bound, which shows up as low efficiency, not as memory- or compute-bound behaviour.",
        ])
        out = self.D / "REPORT.md"
        out.write_text(text)
        print(f"wrote {out}")


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    Report(cfg).render()


if __name__ == "__main__":
    main()
