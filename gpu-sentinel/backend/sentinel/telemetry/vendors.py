"""Vendor → canonical metric mappings (PromQL).

To add AMD (ROCm/``amd-smi`` exporter) or Intel (XPU Manager) support, add
another :class:`VendorMapping` with the same canonical names. Nothing
downstream changes.

All queries must return one series per (node, gpu) for GPU metrics and one
series per node for node metrics. The ``node`` label is expected on every
series (added by Prometheus relabeling — see deploy/prometheus/prometheus.yml).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VendorMapping:
    vendor: str
    gpu_queries: dict[str, str]
    # Canonical metrics that need hardware-catalog post-processing.
    gpu_derived: dict[str, str] = field(default_factory=dict)
    gpu_label: str = "gpu"
    uuid_label: str = "UUID"
    model_label: str = "modelName"


NVIDIA_DCGM = VendorMapping(
    vendor="nvidia",
    gpu_queries={
        "gpu_util": "DCGM_FI_DEV_GPU_UTIL",
        "sm_active": "DCGM_FI_PROF_SM_ACTIVE * 100",
        "perf_index": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * 100",
        "sm_clock_mhz": "DCGM_FI_DEV_SM_CLOCK",
        "mem_clock_mhz": "DCGM_FI_DEV_MEM_CLOCK",
        "temp_c": "DCGM_FI_DEV_GPU_TEMP",
        "mem_temp_c": "DCGM_FI_DEV_MEMORY_TEMP",
        "power_w": "DCGM_FI_DEV_POWER_USAGE",
        "power_limit_w": "DCGM_FI_DEV_POWER_MGMT_LIMIT",
        "mem_util": "DCGM_FI_DEV_MEM_COPY_UTIL",
        "mem_used_pct": "100 * DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)",
        "dram_active": "DCGM_FI_PROF_DRAM_ACTIVE",
        "pcie_tx_gbps": "DCGM_FI_PROF_PCIE_TX_BYTES / 1e9",
        "pcie_rx_gbps": "DCGM_FI_PROF_PCIE_RX_BYTES / 1e9",
        "pcie_link_gen": "DCGM_FI_DEV_PCIE_LINK_GEN",
        "pcie_link_width": "DCGM_FI_DEV_PCIE_LINK_WIDTH",
        "pcie_replay_rate": "rate(DCGM_FI_DEV_PCIE_REPLAY_COUNTER[1m])",
        "nvlink_bw_gbps": "(DCGM_FI_PROF_NVLINK_TX_BYTES + DCGM_FI_PROF_NVLINK_RX_BYTES) / 1e9",
        "nvlink_crc_rate": "rate(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL[1m])",
        "ecc_sbe_rate": "rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[2m]) * 60",
        "ecc_dbe_total": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
        "retired_pages": "DCGM_FI_DEV_RETIRED_SBE + DCGM_FI_DEV_RETIRED_DBE",
        "xid_errors": "DCGM_FI_DEV_XID_ERRORS",
        "throttle_mask": "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS",
    },
    gpu_derived={"hbm_bw_gbps": "dram_active"},
)

VENDORS: dict[str, VendorMapping] = {"nvidia": NVIDIA_DCGM}

# Node / OS / fabric / workload metrics (vendor-neutral: node_exporter + Sentinel agent).
NODE_QUERIES: dict[str, str] = {
    "cpu_util": '100 * (1 - avg by (node) (rate(node_cpu_seconds_total{mode="idle"}[1m])))',
    "cpu_iowait": '100 * avg by (node) (rate(node_cpu_seconds_total{mode="iowait"}[1m]))',
    "load1": "max by (node) (node_load1)",
    "cpu_freq_mhz": "avg by (node) (node_cpu_scaling_frequency_hertz) / 1e6",
    "ctx_switches_k": "sum by (node) (rate(node_context_switches_total[1m])) / 1000",
    "mem_used_pct": "100 * (1 - sum by (node) (node_memory_MemAvailable_bytes) / sum by (node) (node_memory_MemTotal_bytes))",
    "mem_psi_some": "100 * sum by (node) (rate(node_pressure_memory_waiting_seconds_total[1m]))",
    "net_rx_gbps": 'sum by (node) (rate(node_network_receive_bytes_total{device!~"lo|docker.*|veth.*"}[1m])) * 8 / 1e9',
    "net_tx_gbps": 'sum by (node) (rate(node_network_transmit_bytes_total{device!~"lo|docker.*|veth.*"}[1m])) * 8 / 1e9',
    "net_err_rate": "sum by (node) (rate(node_network_receive_errs_total[1m])) + sum by (node) (rate(node_network_transmit_errs_total[1m]))",
    "net_drop_rate": "sum by (node) (rate(node_network_receive_drop_total[1m]))",
    "ib_rx_gbps": "sum by (node) (rate(node_infiniband_port_data_received_bytes_total[1m])) * 8 / 1e9",
    "ib_tx_gbps": "sum by (node) (rate(node_infiniband_port_data_transmitted_bytes_total[1m])) * 8 / 1e9",
    "ib_symbol_err_rate": "sum by (node) (rate(node_infiniband_symbol_error_total[1m]))",
    "net_latency_us": "max by (node) (sentinel_fabric_latency_us)",
    "nccl_busbw_gbps": "max by (node) (sentinel_nccl_allreduce_busbw_gbps)",
    "nccl_comm_ratio": "100 * max by (node) (sentinel_nccl_comm_ratio)",
    "dataloader_wait_pct": "100 * max by (node) (sentinel_dataloader_wait_ratio)",
    "throughput": "sum by (node) (sentinel_workload_throughput)",
    "step_time_ms": "max by (node) (sentinel_workload_step_time_ms)",
}

NODE_INFO_QUERY = "sentinel_node_info"
WORKLOAD_INFO_QUERY = "sentinel_workload_info"
# Used when no Sentinel node-info metric exists (plain DCGM deployment).
DCGM_INVENTORY_QUERY = "DCGM_FI_DEV_GPU_TEMP"
