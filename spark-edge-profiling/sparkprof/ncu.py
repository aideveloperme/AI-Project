"""Nsight Compute helpers: run a target under `ncu`, parse raw CSV into per-kernel metrics.

DRAM traffic on GB10 is LPDDR5x traffic ("dram__*" metrics); L2 traffic ("lts__*")
shows how much reuse the cache provides. Requires access to GPU performance counters:
run as root (inside the NGC container with --cap-add SYS_ADMIN) or set the driver
option NVreg_RestrictProfilingToAdminUsers=0.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess

METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]


def available() -> bool:
    return shutil.which("ncu") is not None


def run(cmd: list[str], metrics: list[str] | None = None, extra: list[str] | None = None,
        timeout: int = 1800) -> list[dict]:
    """Profile `cmd` (only between cudaProfilerStart/Stop) and return one dict per kernel."""
    args = ["ncu", "--profile-from-start", "off", "--metrics", ",".join(metrics or METRICS),
            "--csv", "--page", "raw", "--print-units", "base", *(extra or []), *cmd]
    from .config import PROJECT_ROOT
    out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"ncu failed ({out.returncode}):\n{out.stderr[-3000:]}")
    return parse_raw_csv(out.stdout)


def parse_raw_csv(text: str) -> list[dict]:
    # ncu prints its own "==PROF==" lines and the program's stdout before the CSV
    lines = [l for l in text.splitlines() if l.startswith('"')]
    if len(lines) < 3:
        return []
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header, _units, data = rows[0], rows[1], rows[2:]
    kernels = []
    for r in data:
        d = dict(zip(header, r))
        k = {"kernel": d.get("Kernel Name", ""), "id": d.get("ID")}
        for m in METRICS:
            if m in d:
                try:
                    k[m] = float(d[m].replace(",", ""))
                except ValueError:
                    k[m] = None
        kernels.append(k)
    return kernels


def aggregate(kernels: list[dict], calls: int = 1) -> dict:
    """Sum kernel metrics and divide by the number of layer invocations profiled."""
    def s(m):
        return sum(k.get(m) or 0 for k in kernels) / max(calls, 1)
    return {
        "kernels_per_call": len(kernels) / max(calls, 1),
        "time_ms": s("gpu__time_duration.sum") / 1e6,
        "dram_read_bytes": s("dram__bytes_read.sum"),
        "dram_write_bytes": s("dram__bytes_write.sum"),
        "dram_bytes": s("dram__bytes_read.sum") + s("dram__bytes_write.sum"),
        "l2_bytes": s("lts__t_bytes.sum"),
        "dram_pct_peak": max((k.get("dram__throughput.avg.pct_of_peak_sustained_elapsed") or 0)
                             for k in kernels) if kernels else 0,
        "sm_pct_peak": max((k.get("sm__throughput.avg.pct_of_peak_sustained_elapsed") or 0)
                           for k in kernels) if kernels else 0,
    }
