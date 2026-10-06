"""Before/after report: compare experiments against a baseline.

Reads ``<results_dir>/<experiment>/summary.json`` for every experiment and
writes ``REPORT.md`` plus ``plots/*.png``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Categorical palette (fixed order, colour follows the experiment, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SIM_BANNER = (
    "> **SIMULATED RESULTS - NOT GB10 MEASUREMENTS.** Produced by the analytical GB10 simulator "
    "(`servebench.mock`) to validate the harness end-to-end. They show the *direction* of each knob, "
    "not real numbers. Agent `task success` is meaningless here (the simulator does not reason). "
    "Real results live in `results/gb10/` after running on the device.\n"
)


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(results_dir.glob("*/summary.json")):
        out[p.parent.name] = json.loads(p.read_text())
    if not out:
        raise FileNotFoundError(f"no */summary.json under {results_dir}")
    return out


def _g(d: dict[str, Any] | None, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _delta(new: Any, base: Any, lower_is_better: bool | None) -> str:
    if new is None or base in (None, 0):
        return ""
    pct = (new - base) / base * 100
    if abs(pct) < 0.5:
        return " (=)"
    if lower_is_better is None:  # descriptive metric: no better/worse judgement
        return f" ({'+' if pct > 0 else ''}{pct:.0f}%)"
    good = (pct < 0) == lower_is_better
    return f" ({'+' if pct > 0 else ''}{pct:.0f}% {'better' if good else 'worse'})"


# metric label, path in level summary, lower-is-better, decimals
LEVEL_METRICS = [
    ("output tok/s", ("output_tok_per_s",), False, 1),
    ("TTFT p50 ms", ("ttft_ms", "p50"), True, 0),
    ("TTFT p99 ms", ("ttft_ms", "p99"), True, 0),
    ("TPOT p50 ms", ("tpot_ms", "p50"), True, 1),
    ("E2E p99 ms", ("e2e_ms", "p99"), True, 0),
    ("goodput req/s", ("goodput_rps",), False, 2),
    ("errors", ("error_rate",), True, 3),
    ("peak KV usage", ("server", "peak_kv_cache_usage"), True, 2),
    ("preemptions", ("server", "preemptions"), True, 0),
    ("prefix hit rate", ("server", "prefix_cache_hit_rate"), False, 2),
]

AGENT_METRICS = [
    ("task p50 ms", ("task_latency_p50_ms",), True, 0),
    ("task p90 ms", ("task_latency_p90_ms",), True, 0),
    ("episodes/min", ("episodes_per_min",), False, 1),
    ("TTFT share of LLM time", ("ttft_share_of_llm_time",), None, 2),
    ("prefix hit rate", ("server", "prefix_cache_hit_rate"), False, 2),
    ("task success", ("task_success_rate",), False, 2),
]


def _levels(res: dict[str, Any], workload: str) -> dict[float, dict[str, Any]]:
    return {s["level"]: s for s in res.get("loadtest", []) if s["workload"] == workload}


def _agent(res: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(a["mode"], a["sessions"]): a for a in res.get("agent", [])}


def _plot_lines(
    path: Path, title: str, ylabel: str, series: dict[str, list[tuple[float, float]]], xlabel: str, log_x: bool = True
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    for i, (name, pts) in enumerate(series.items()):
        pts = [(x, y) for x, y in pts if y is not None]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts), strict=True)
        ax.plot(
            xs,
            ys,
            color=PALETTE[i % len(PALETTE)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=6,
            linewidth=2,
            label=name,
        )
    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_report(
    results_dir: Path,
    baseline: str | None = None,
    headline_level: float | None = None,
    title: str = "vLLM on DGX Spark (GB10): before / after",
) -> Path:
    results = load_results(results_dir)
    names = list(results)
    base = baseline if baseline in results else names[0]
    others = [n for n in names if n != base]
    simulated = any(r.get("simulated") for r in results.values())
    plots = results_dir / "plots"
    md: list[str] = [f"# {title}", ""]
    if simulated:
        md += [SIM_BANNER, ""]
    md += [
        f"Baseline: **`{base}`**. Each other experiment changes one or two knobs relative to it.",
        "",
        "| experiment | what changed | server startup s |",
        "|---|---|---:|",
    ]
    for n in names:
        ch = "; ".join(results[n].get("changes") or []) or results[n].get("description", "")
        md.append(f"| `{n}` | {ch or '(baseline)'} | {_fmt(results[n].get('startup_s'))} |")
    md.append("")

    workloads: list[str] = []
    for r in results.values():
        for s in r.get("loadtest", []):
            if s["workload"] not in workloads:
                workloads.append(s["workload"])

    for wl in workloads:
        md += [f"## Workload `{wl}`", ""]
        common = sorted(set.intersection(*[set(_levels(results[n], wl)) for n in names if _levels(results[n], wl)]))
        if not common:
            continue
        lvl = headline_level if headline_level in common else common[-1]
        mode = _levels(results[base], wl).get(lvl, {}).get("mode", "concurrency")
        md += [
            f"Headline at {mode} = **{lvl:g}** (deltas vs `{base}`):",
            "",
            "| metric | " + " | ".join(f"`{n}`" for n in names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for label, path, lower, nd in LEVEL_METRICS:
            bv = _g(_levels(results[base], wl).get(lvl), *path)
            cells = []
            for n in names:
                v = _g(_levels(results[n], wl).get(lvl), *path)
                cells.append(_fmt(v, nd) + (_delta(v, bv, lower) if n != base else ""))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
        md += [
            "<details><summary>All levels</summary>",
            "",
            "| experiment | level | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | goodput | peak KV | preempt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n in names:
            for lv, s in sorted(_levels(results[n], wl).items()):
                md.append(
                    f"| `{n}` | {lv:g} | {_fmt(s['output_tok_per_s'])} | {_fmt(s['ttft_ms']['p50'], 0)} | "
                    f"{_fmt(s['ttft_ms']['p99'], 0)} | {_fmt(s['tpot_ms']['p50'])} | "
                    f"{_fmt(s['goodput_rps'], 2)} | {_fmt(_g(s, 'server', 'peak_kv_cache_usage'), 2)} | "
                    f"{_fmt(_g(s, 'server', 'preemptions'), 0)} |"
                )
        md += ["", "</details>", ""]
        xl = f"{mode} (log2)"
        for key, label, path in (
            ("throughput", "output tokens/s", ("output_tok_per_s",)),
            ("ttft_p50", "TTFT p50 (ms)", ("ttft_ms", "p50")),
            ("tpot_p50", "TPOT p50 (ms)", ("tpot_ms", "p50")),
        ):
            series = {n: [(lv, _g(s, *path)) for lv, s in _levels(results[n], wl).items()] for n in names}
            f = plots / f"{wl}_{key}.png"
            _plot_lines(f, f"{wl}: {label}", label, series, xl, log_x=mode == "concurrency")
            md.append(f"![{wl} {key}](plots/{f.name})")
        md.append("")

    agent_keys = sorted({k for r in results.values() for k in _agent(r)})
    if agent_keys:
        md += [
            "## Agent loop",
            "",
            "Multi-step tool-using agent; each step's prompt = previous prompt + output + tool result. "
            "`replay` mode has a fixed shape so latency is comparable across configs; `react` mode uses real "
            "tool calls and also reports task success.",
            "",
        ]
        for key in agent_keys:
            md += [
                f"### `{key[0]}` - {key[1]} concurrent session(s)",
                "",
                "| metric | " + " | ".join(f"`{n}`" for n in names) + " |",
                "|---|" + "---:|" * len(names),
            ]
            for label, path, lower, nd in AGENT_METRICS:
                bv = _g(_agent(results[base]).get(key), *path)
                cells = []
                for n in names:
                    v = _g(_agent(results[n]).get(key), *path)
                    cells.append(_fmt(v, nd) + (_delta(v, bv, lower) if n != base else ""))
                md.append(f"| {label} | " + " | ".join(cells) + " |")
            md.append("")
            series = {
                n: [(st["step"], st["ttft_p50_ms"]) for st in (_agent(results[n]).get(key) or {}).get("per_step", [])]
                for n in names
            }
            f = plots / f"agent_{key[0]}_s{key[1]}_ttft_per_step.png"
            _plot_lines(
                f, f"agent {key[0]} x{key[1]}: TTFT p50 by step", "TTFT p50 (ms)", series, "agent step", log_x=False
            )
            md += [f"![agent ttft per step](plots/{f.name})", ""]

    if any(r.get("profile") for r in results.values()):
        md += [
            "## Profiles",
            "",
            "Open the `.json.gz` traces in https://ui.perfetto.dev (drag & drop).",
            "",
            "| experiment | trace | GPU busy frac | top categories |",
            "|---|---|---:|---|",
        ]
        for n in names:
            p = results[n].get("profile") or {}
            for t, s in zip(p.get("traces", []), p.get("summaries", []), strict=False):
                cats = ", ".join(f"{k} {v:.0f}%" for k, v in list(s["by_category_pct"].items())[:3])
                md.append(f"| `{n}` | [{Path(t).name}]({n}/{t}) | {_fmt(s.get('gpu_busy_frac'), 2)} | {cats} |")
        md.append("")
    md += [
        "---",
        f"Generated by `servebench report` from `{results_dir}`. Raw per-request data: "
        "`<experiment>/loadtest/<workload>/*.requests.jsonl`.",
    ]
    if others and simulated:
        md += ["", SIM_BANNER]
    out = results_dir / "REPORT.md"
    out.write_text("\n".join(md) + "\n")
    return out
