"""Canonical metric catalog.

Each metric has a canonical name, unit, category and the *direction* in which a
deviation is considered bad. Detection thresholds live here so they can be
tuned per deployment (see ``Settings.metric_overrides``) without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Level = Literal["gpu", "node"]
Direction = Literal["low_bad", "high_bad", "both"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    level: Level
    unit: str
    label: str
    category: str
    direction: Direction = "both"
    # Peer comparison: minimum relative deviation from peer median to flag,
    # and an absolute floor on the robust spread so near-constant metrics
    # don't produce huge z-scores on tiny differences.
    peer_min_rel_dev: float = 0.10
    peer_abs_floor: float = 0.0
    # Static thresholds (inclusive), evaluated on the moving average.
    warn: float | None = None
    crit: float | None = None
    # Counters/rates where any non-zero value is abnormal (ECC DBE, XID...).
    nonzero_is_bad: bool = False
    peer_compare: bool = True
    temporal: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)


GPU_METRICS: list[MetricSpec] = [
    MetricSpec("gpu_util", "gpu", "%", "GPU utilization", "compute", "low_bad", 0.15, 3.0),
    MetricSpec("sm_active", "gpu", "%", "SM activity", "compute", "low_bad", 0.15, 3.0),
    MetricSpec("sm_clock_mhz", "gpu", "MHz", "SM clock", "clocks", "low_bad", 0.07, 15.0),
    MetricSpec("mem_clock_mhz", "gpu", "MHz", "Memory clock", "clocks", "low_bad", 0.05, 10.0),
    MetricSpec("temp_c", "gpu", "°C", "GPU temperature", "thermal", "high_bad", 0.10, 2.0, warn=85, crit=90),
    MetricSpec("mem_temp_c", "gpu", "°C", "HBM temperature", "thermal", "high_bad", 0.10, 2.0, warn=95, crit=100),
    MetricSpec("power_w", "gpu", "W", "Power draw", "power", "both", 0.12, 15.0),
    MetricSpec("power_limit_w", "gpu", "W", "Enforced power limit", "power", "low_bad", 0.05, 5.0, temporal=False),
    MetricSpec("mem_util", "gpu", "%", "Memory copy utilization", "memory", "low_bad", 0.20, 3.0),
    MetricSpec("mem_used_pct", "gpu", "%", "Framebuffer used", "memory", "high_bad", 0.25, 5.0, warn=97, peer_compare=False),
    MetricSpec("hbm_bw_gbps", "gpu", "GB/s", "HBM bandwidth", "memory", "low_bad", 0.12, 40.0),
    MetricSpec("pcie_tx_gbps", "gpu", "GB/s", "PCIe TX", "pcie", "low_bad", 0.30, 0.5),
    MetricSpec("pcie_rx_gbps", "gpu", "GB/s", "PCIe RX", "pcie", "low_bad", 0.30, 0.5),
    MetricSpec("pcie_link_gen", "gpu", "gen", "PCIe link generation", "pcie", "low_bad", 0.01, 0.0, temporal=False),
    MetricSpec("pcie_link_width", "gpu", "lanes", "PCIe link width", "pcie", "low_bad", 0.01, 0.0, temporal=False),
    MetricSpec("pcie_replay_rate", "gpu", "/s", "PCIe replay rate", "pcie", "high_bad", 1.0, 0.05, warn=1.0, crit=10.0, peer_compare=False),
    MetricSpec("nvlink_bw_gbps", "gpu", "GB/s", "NVLink bandwidth", "nvlink", "low_bad", 0.15, 5.0),
    MetricSpec("nvlink_crc_rate", "gpu", "/s", "NVLink CRC error rate", "nvlink", "high_bad", 1.0, 0.05, warn=0.5, crit=5.0, peer_compare=False),
    MetricSpec("ecc_sbe_rate", "gpu", "/min", "ECC single-bit error rate", "ecc", "high_bad", 1.0, 0.1, warn=5.0, crit=50.0, peer_compare=False),
    MetricSpec("ecc_dbe_total", "gpu", "count", "ECC double-bit errors (total)", "ecc", "high_bad", 1.0, 0.0, crit=1.0, nonzero_is_bad=True, peer_compare=False, temporal=False),
    MetricSpec("retired_pages", "gpu", "count", "Retired memory pages", "ecc", "high_bad", 1.0, 0.0, warn=1.0, crit=16.0, peer_compare=False, temporal=False),
    MetricSpec("xid_errors", "gpu", "code", "Last XID error", "health", "high_bad", 1.0, 0.0, warn=1.0, nonzero_is_bad=True, peer_compare=False, temporal=False),
    MetricSpec("perf_index", "gpu", "%", "Tensor pipe activity (throughput proxy)", "performance", "low_bad", 0.08, 2.0),
]

NODE_METRICS: list[MetricSpec] = [
    MetricSpec("cpu_util", "node", "%", "CPU utilization", "cpu", "high_bad", 0.35, 5.0, warn=92, crit=98),
    MetricSpec("cpu_iowait", "node", "%", "CPU iowait", "cpu", "high_bad", 1.0, 2.0, warn=20),
    MetricSpec("load1", "node", "", "Load average (1m)", "cpu", "high_bad", 0.50, 4.0),
    MetricSpec("cpu_freq_mhz", "node", "MHz", "CPU frequency", "cpu", "low_bad", 0.10, 50.0),
    MetricSpec("ctx_switches_k", "node", "k/s", "Context switches", "cpu", "high_bad", 0.60, 10.0),
    MetricSpec("mem_used_pct", "node", "%", "System memory used", "memory", "high_bad", 0.30, 5.0, warn=92, crit=97, temporal=False),
    MetricSpec("mem_psi_some", "node", "%", "Memory pressure (PSI some)", "memory", "high_bad", 1.0, 5.0, warn=10, crit=30),
    MetricSpec("net_rx_gbps", "node", "Gb/s", "Ethernet RX", "network", "low_bad", 0.35, 1.0),
    MetricSpec("net_tx_gbps", "node", "Gb/s", "Ethernet TX", "network", "low_bad", 0.35, 1.0),
    MetricSpec("net_err_rate", "node", "/s", "Network error rate", "network", "high_bad", 1.0, 0.1, warn=1.0, crit=20.0, peer_compare=False),
    MetricSpec("net_drop_rate", "node", "/s", "Network drop rate", "network", "high_bad", 1.0, 0.1, warn=5.0, crit=50.0, peer_compare=False),
    MetricSpec("ib_rx_gbps", "node", "Gb/s", "InfiniBand RX", "network", "low_bad", 0.20, 5.0),
    MetricSpec("ib_tx_gbps", "node", "Gb/s", "InfiniBand TX", "network", "low_bad", 0.20, 5.0),
    MetricSpec("ib_symbol_err_rate", "node", "/s", "InfiniBand symbol error rate", "network", "high_bad", 1.0, 0.05, warn=0.5, crit=10.0, peer_compare=False),
    MetricSpec("net_latency_us", "node", "µs", "Fabric latency (p50)", "network", "high_bad", 0.30, 0.5),
    MetricSpec("nccl_busbw_gbps", "node", "GB/s", "NCCL all-reduce bus bandwidth", "communication", "low_bad", 0.15, 5.0),
    MetricSpec("nccl_comm_ratio", "node", "%", "Communication / step time", "communication", "high_bad", 0.35, 3.0),
    MetricSpec("dataloader_wait_pct", "node", "%", "GPU input-wait (data loader)", "workload", "high_bad", 0.80, 3.0),
    MetricSpec("throughput", "node", "units/s", "Workload throughput", "performance", "low_bad", 0.08, 0.0),
    MetricSpec("step_time_ms", "node", "ms", "Step / request latency", "performance", "high_bad", 0.10, 5.0),
]

CATALOG: dict[tuple[str, str], MetricSpec] = {(m.level, m.name): m for m in GPU_METRICS + NODE_METRICS}


def spec(level: str, name: str) -> MetricSpec | None:
    return CATALOG.get((level, name))


# Throttle reason bitmask values from NVML / DCGM (DCGM_FI_DEV_CLOCK_THROTTLE_REASONS).
THROTTLE_REASONS: dict[int, str] = {
    0x0001: "gpu_idle",
    0x0002: "applications_clocks_setting",
    0x0004: "sw_power_cap",
    0x0008: "hw_slowdown",
    0x0010: "sync_boost",
    0x0020: "sw_thermal_slowdown",
    0x0040: "hw_thermal_slowdown",
    0x0080: "hw_power_brake_slowdown",
    0x0100: "display_clock_setting",
}


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_REASONS.items() if mask & bit]


def encode_throttle(reasons: list[str]) -> int:
    rev = {v: k for k, v in THROTTLE_REASONS.items()}
    mask = 0
    for r in reasons:
        mask |= rev.get(r, 0)
    return mask


# Static hardware catalog. Used for nominal values (peak HBM bandwidth,
# boost clocks) when deriving canonical metrics from vendor ratios.
GPU_HARDWARE: dict[str, dict[str, float]] = {
    "NVIDIA H100 80GB HBM3": {"hbm_peak_gbps": 3350, "max_sm_clock": 1980, "mem_clock": 2619, "tdp_w": 700, "nvlink_peak_gbps": 450},
    "NVIDIA A100-SXM4-80GB": {"hbm_peak_gbps": 2039, "max_sm_clock": 1410, "mem_clock": 1593, "tdp_w": 400, "nvlink_peak_gbps": 300},
    "NVIDIA L40S": {"hbm_peak_gbps": 864, "max_sm_clock": 2520, "mem_clock": 9001, "tdp_w": 350, "nvlink_peak_gbps": 0},
    # DGX Spark (GB10 Grace Blackwell): 128 GB unified LPDDR5x at 273 GB/s shared with the CPU.
    # The Prometheus collector only uses hbm_peak_gbps (DRAM_ACTIVE → GB/s); the other
    # values are used by the simulator and are approximate.
    "NVIDIA GB10": {"hbm_peak_gbps": 273, "max_sm_clock": 3000, "mem_clock": 4266, "tdp_w": 140, "nvlink_peak_gbps": 0},
}
DEFAULT_HARDWARE = GPU_HARDWARE["NVIDIA H100 80GB HBM3"]


def hardware(model: str) -> dict[str, float]:
    return GPU_HARDWARE.get(model, DEFAULT_HARDWARE)
