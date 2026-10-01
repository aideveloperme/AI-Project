"""Render simulator state in Prometheus exposition format.

Metric names and label sets mirror the real exporters so the backend's PromQL
mapping (``sentinel.telemetry.vendors``) works unchanged against a real cluster:

* NVIDIA DCGM Exporter  — ``DCGM_FI_*`` with ``gpu``, ``UUID``, ``modelName``, ``Hostname``
* Prometheus node_exporter — ``node_*`` counters/gauges
* Sentinel workload agent — ``sentinel_*`` (PyTorch/NCCL instrumentation, fabric probe)

Every series also carries ``node="<hostname>"`` — in production this label is
added by Prometheus relabeling (see deploy/prometheus/prometheus.yml).
"""
from __future__ import annotations

from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.catalog import hardware


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{str(v)}"' for k, v in d.items()) + "}"


class _Writer:
    def __init__(self) -> None:
        self.families: dict[str, tuple[str, str, list[str]]] = {}

    def add(self, name: str, mtype: str, help_: str, labels: dict, value: float) -> None:
        fam = self.families.setdefault(name, (mtype, help_, []))
        fam[2].append(f"{name}{_labels(labels)} {value:.6g}" if isinstance(value, float) else f"{name}{_labels(labels)} {value}")

    def render(self) -> str:
        out = []
        for name, (mtype, help_, lines) in self.families.items():
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {mtype}")
            out.extend(lines)
        return "\n".join(out) + "\n"


def render_metrics(sim: ClusterSimulator) -> str:
    snap = sim.last_snapshot or sim.tick(0.0)
    w = _Writer()
    state = {n.name: n for n in sim.nodes}
    for node in snap.nodes:
        st = state[node.node]
        hw = hardware(node.gpu_model)
        w.add("sentinel_node_info", "gauge", "Node inventory", {
            "node": node.node, "cluster": node.cluster, "server_type": node.server_type,
            "gpu_model": node.gpu_model, "rack": node.rack or "", "driver_version": node.driver_version or "",
            "cuda_version": node.cuda_version or "", "job_id": node.workload_id or ""}, 1)
        for g in node.gpus:
            gs = st.gpus[g.index]
            base = {"gpu": g.index, "UUID": g.uuid, "device": f"nvidia{g.index}", "modelName": g.model,
                    "Hostname": node.node, "node": node.node}
            m = g.metrics
            if node.workload_id:
                base_job = {**base, "job_id": node.workload_id}
            else:
                base_job = base
            w.add("DCGM_FI_DEV_GPU_UTIL", "gauge", "GPU utilization (in %).", base_job, m["gpu_util"])
            w.add("DCGM_FI_PROF_SM_ACTIVE", "gauge", "Ratio of cycles an SM has at least 1 warp assigned.", base, m["sm_active"] / 100)
            w.add("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "gauge", "Ratio of cycles the tensor pipe is active.", base, m["perf_index"] / 100)
            w.add("DCGM_FI_PROF_DRAM_ACTIVE", "gauge", "Ratio of cycles the device memory interface is active.", base, m["hbm_bw_gbps"] / hw["hbm_peak_gbps"])
            w.add("DCGM_FI_DEV_SM_CLOCK", "gauge", "SM clock frequency (in MHz).", base, m["sm_clock_mhz"])
            w.add("DCGM_FI_DEV_MEM_CLOCK", "gauge", "Memory clock frequency (in MHz).", base, m["mem_clock_mhz"])
            w.add("DCGM_FI_DEV_GPU_TEMP", "gauge", "GPU temperature (in C).", base, m["temp_c"])
            w.add("DCGM_FI_DEV_MEMORY_TEMP", "gauge", "Memory temperature (in C).", base, m["mem_temp_c"])
            w.add("DCGM_FI_DEV_POWER_USAGE", "gauge", "Power draw (in W).", base, m["power_w"])
            w.add("DCGM_FI_DEV_POWER_MGMT_LIMIT", "gauge", "Power management limit (in W).", base, m["power_limit_w"])
            w.add("DCGM_FI_DEV_MEM_COPY_UTIL", "gauge", "Memory utilization (in %).", base, m["mem_util"])
            used = round(81559 * m["mem_used_pct"] / 100)
            w.add("DCGM_FI_DEV_FB_USED", "gauge", "Framebuffer memory used (in MiB).", base, used)
            w.add("DCGM_FI_DEV_FB_FREE", "gauge", "Framebuffer memory free (in MiB).", base, 81559 - used)
            w.add("DCGM_FI_PROF_PCIE_TX_BYTES", "gauge", "PCIe TX bytes per second.", base, m["pcie_tx_gbps"] * 1e9)
            w.add("DCGM_FI_PROF_PCIE_RX_BYTES", "gauge", "PCIe RX bytes per second.", base, m["pcie_rx_gbps"] * 1e9)
            w.add("DCGM_FI_DEV_PCIE_LINK_GEN", "gauge", "PCIe current link generation.", base, m["pcie_link_gen"])
            w.add("DCGM_FI_DEV_PCIE_LINK_WIDTH", "gauge", "PCIe current link width.", base, m["pcie_link_width"])
            w.add("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "counter", "Total PCIe retries.", base, gs.counters["pcie_replay"])
            w.add("DCGM_FI_PROF_NVLINK_TX_BYTES", "gauge", "NVLink TX bytes per second.", base, m["nvlink_bw_gbps"] * 1e9 / 2)
            w.add("DCGM_FI_PROF_NVLINK_RX_BYTES", "gauge", "NVLink RX bytes per second.", base, m["nvlink_bw_gbps"] * 1e9 / 2)
            w.add("DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL", "counter", "NVLink CRC flit errors.", base, gs.counters["nvlink_crc"])
            w.add("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "counter", "Volatile single-bit ECC errors.", base, gs.counters["ecc_sbe"])
            w.add("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "counter", "Volatile double-bit ECC errors.", base, gs.counters["ecc_dbe"])
            w.add("DCGM_FI_DEV_RETIRED_SBE", "counter", "Pages retired due to SBE.", base, gs.counters["retired_sbe"])
            w.add("DCGM_FI_DEV_RETIRED_DBE", "counter", "Pages retired due to DBE.", base, gs.counters["retired_dbe"])
            w.add("DCGM_FI_DEV_XID_ERRORS", "gauge", "Value of the last XID error encountered.", base, m["xid_errors"])
            w.add("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "gauge", "Current clock throttle reasons (bitmask).", base, int(m["throttle_mask"]))

        # ---------------- node_exporter compatible
        n = {"node": node.node, "instance": f"{node.node}:9100"}
        c = st.counters
        cores = sim.cpu_cores
        # Expose per-mode totals on a single synthetic "cpu" to keep cardinality low;
        # PromQL uses avg by (node) so this is equivalent to per-core series.
        for mode, key in (("idle", "cpu_idle"), ("user", "cpu_user"), ("system", "cpu_system"), ("iowait", "cpu_iowait")):
            w.add("node_cpu_seconds_total", "counter", "Seconds the CPUs spent in each mode.", {**n, "cpu": "all", "mode": mode}, c[key] / cores)
        w.add("node_load1", "gauge", "1m load average.", n, node.metrics["load1"])
        w.add("node_cpu_scaling_frequency_hertz", "gauge", "Current scaled CPU frequency.", {**n, "cpu": "all"}, node.metrics["cpu_freq_mhz"] * 1e6)
        w.add("node_context_switches_total", "counter", "Total number of context switches.", n, c["ctx"])
        total = 2 * 1024 ** 4
        w.add("node_memory_MemTotal_bytes", "gauge", "Memory information field MemTotal_bytes.", n, float(total))
        w.add("node_memory_MemAvailable_bytes", "gauge", "Memory information field MemAvailable_bytes.", n, float(total * (1 - node.metrics["mem_used_pct"] / 100)))
        w.add("node_pressure_memory_waiting_seconds_total", "counter", "Total time processes stalled due to memory.", n, c["psi_mem"])
        eth = {**n, "device": "eth0"}
        w.add("node_network_receive_bytes_total", "counter", "Network device statistic receive_bytes.", eth, c["net_rx"])
        w.add("node_network_transmit_bytes_total", "counter", "Network device statistic transmit_bytes.", eth, c["net_tx"])
        w.add("node_network_receive_errs_total", "counter", "Network device statistic receive_errs.", eth, c["net_errs"] / 2)
        w.add("node_network_transmit_errs_total", "counter", "Network device statistic transmit_errs.", eth, c["net_errs"] / 2)
        w.add("node_network_receive_drop_total", "counter", "Network device statistic receive_drop.", eth, c["net_drop"])
        ib = {**n, "device": "mlx5_0", "port": "1"}
        w.add("node_infiniband_port_data_received_bytes_total", "counter", "Data received (bytes).", ib, c["ib_rx"])
        w.add("node_infiniband_port_data_transmitted_bytes_total", "counter", "Data transmitted (bytes).", ib, c["ib_tx"])
        w.add("node_infiniband_symbol_error_total", "counter", "Symbol errors.", ib, c["ib_symbol"])

        # ---------------- sentinel workload / NCCL agent
        wl = {**n, "job_id": node.workload_id or ""}
        w.add("sentinel_workload_throughput", "gauge", "Workload throughput on this node (samples/s or tokens/s).", wl, node.metrics["throughput"])
        w.add("sentinel_workload_step_time_ms", "gauge", "Training step time or request latency (ms).", wl, node.metrics["step_time_ms"])
        w.add("sentinel_dataloader_wait_ratio", "gauge", "Fraction of step time GPUs wait for input.", wl, node.metrics["dataloader_wait_pct"] / 100)
        w.add("sentinel_nccl_allreduce_busbw_gbps", "gauge", "NCCL all-reduce bus bandwidth (GB/s).", wl, node.metrics["nccl_busbw_gbps"])
        w.add("sentinel_nccl_comm_ratio", "gauge", "Fraction of step time spent in collectives.", wl, node.metrics["nccl_comm_ratio"] / 100)
        w.add("sentinel_fabric_latency_us", "gauge", "Fabric probe p50 latency (us).", n, node.metrics["net_latency_us"])

    for wk in snap.workloads:
        w.add("sentinel_workload_info", "gauge", "Workload/job inventory.", {
            "job_id": wk.job_id, "name": wk.name, "scheduler": wk.scheduler, "framework": wk.framework,
            "kind": wk.kind, "user": wk.user or "", "namespace": wk.namespace or "",
            "nodes": ",".join(wk.nodes), "throughput_unit": wk.throughput_unit}, 1)
    w.add("sentinel_simulator_active_faults", "gauge", "Number of injected faults currently active.", {}, len(sim.active_faults()))
    return w.render()
