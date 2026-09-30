"""Physically-plausible GPU cluster telemetry simulator.

Produces canonical telemetry (``FleetSnapshot``) plus cumulative counters used
by the Prometheus exporter (``sentinel.simulator.exporter``) so the whole
pipeline — exporter → Prometheus → PromQL → analytics — can be exercised
without real hardware.

Model:
* each GPU has a steady-state operating point that depends on the workload
  (training vs. inference) plus small per-device manufacturing variance;
* temperatures follow a first-order thermal model (they *ramp*, not jump);
* faults modify per-GPU/per-node factors (clock, HBM, utilization, comm);
* node workload throughput is derived from those factors, so a slow GPU really
  does make the node slower than its peers.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field

from sentinel.simulator.faults import GPU_SCOPED, Fault, FaultType
from sentinel.telemetry.catalog import encode_throttle, hardware
from sentinel.telemetry.models import FleetSnapshot, GPUSample, NodeSample, Workload, utcnow

TRAINING_PROFILE = {
    "gpu_util": 97.0, "sm_active": 88.0, "sm_clock": 1830.0, "temp": 67.0, "power": 615.0,
    "mem_util": 55.0, "mem_used": 78.0, "dram_active": 0.55, "pcie_tx": 4.0, "pcie_rx": 9.0,
    "nvlink": 310.0, "tensor": 0.62,
    "cpu_util": 38.0, "load1": 30.0, "ctx_k": 180.0, "mem_used_node": 55.0, "net_gbps": 12.0,
    "ib_gbps": 1500.0, "latency_us": 2.1, "nccl_busbw": 360.0, "comm_ratio": 18.0,
    "dl_wait": 2.0, "throughput": 1450.0, "step_ms": 705.0,
}
INFERENCE_PROFILE = {
    "gpu_util": 74.0, "sm_active": 56.0, "sm_clock": 1905.0, "temp": 58.0, "power": 430.0,
    "mem_util": 48.0, "mem_used": 86.0, "dram_active": 0.45, "pcie_tx": 6.0, "pcie_rx": 5.0,
    "nvlink": 45.0, "tensor": 0.38,
    "cpu_util": 30.0, "load1": 22.0, "ctx_k": 240.0, "mem_used_node": 48.0, "net_gbps": 38.0,
    "ib_gbps": 120.0, "latency_us": 2.3, "nccl_busbw": 180.0, "comm_ratio": 8.0,
    "dl_wait": 1.0, "throughput": 9800.0, "step_ms": 45.0,
}


@dataclass
class GPUState:
    index: int
    uuid: str
    variance: dict[str, float]
    temp: float = 40.0
    mem_temp: float = 48.0
    counters: dict[str, float] = field(default_factory=lambda: {
        "ecc_sbe": 0.0, "ecc_dbe": 0.0, "pcie_replay": 0.0, "nvlink_crc": 0.0,
        "retired_sbe": 0.0, "retired_dbe": 0.0,
    })
    last_xid: int = 0


@dataclass
class NodeState:
    name: str
    cluster: str
    server_type: str
    gpu_model: str
    rack: str
    workload_id: str
    gpus: list[GPUState]
    counters: dict[str, float] = field(default_factory=lambda: {
        "cpu_idle": 0.0, "cpu_user": 0.0, "cpu_system": 0.0, "cpu_iowait": 0.0,
        "ctx": 0.0, "net_rx": 0.0, "net_tx": 0.0, "net_errs": 0.0, "net_drop": 0.0,
        "ib_rx": 0.0, "ib_tx": 0.0, "ib_symbol": 0.0, "psi_mem": 0.0,
    })


class ClusterSimulator:
    def __init__(
        self,
        n_nodes: int = 12,
        gpus_per_node: int = 8,
        gpu_model: str = "NVIDIA H100 80GB HBM3",
        cluster: str = "dxb-ai-01",
        seed: int | None = None,
        random_faults: bool = False,
        random_fault_rate_per_hour: float = 2.0,
        cpu_cores: int = 112,
    ):
        self.rng = random.Random(seed)
        self.gpu_model = gpu_model
        self.cluster = cluster
        self.cpu_cores = cpu_cores
        self.random_faults = random_faults
        self.random_fault_rate_per_hour = random_fault_rate_per_hour
        self.faults: list[Fault] = []
        self.lock = threading.RLock()
        self.sim_time = 0.0
        self.last_snapshot: FleetSnapshot | None = None
        self.last_derived: dict[str, dict] = {}

        n_train = max(1, round(n_nodes * 2 / 3))
        train_nodes = [f"gpu-{i:02d}" for i in range(1, n_train + 1)]
        infer_nodes = [f"gpu-{i:02d}" for i in range(n_train + 1, n_nodes + 1)]
        self.workloads = [
            Workload(job_id="78421", name="llama3-70b-pretrain", scheduler="slurm", framework="pytorch",
                     kind="training", user="ml-research", nodes=train_nodes, throughput_unit="samples/s"),
        ]
        if infer_nodes:
            self.workloads.append(
                Workload(job_id="k8s-llm-serving", name="llm-serving", scheduler="kubernetes",
                         framework="pytorch", kind="inference", namespace="inference",
                         nodes=infer_nodes, throughput_unit="tokens/s"))
        node_to_job = {n: w.job_id for w in self.workloads for n in w.nodes}

        self.nodes: list[NodeState] = []
        for i in range(1, n_nodes + 1):
            name = f"gpu-{i:02d}"
            gpus = []
            for g in range(gpus_per_node):
                # Small, persistent per-device variance (silicon lottery, slot airflow).
                variance = {
                    "clock": self.rng.gauss(0, 0.006),
                    "temp": self.rng.gauss(0, 1.4) + (1.5 if g in (3, 4) else 0.0),
                    "power": self.rng.gauss(0, 0.015),
                    "hbm": self.rng.gauss(0, 0.012),
                }
                gpus.append(GPUState(index=g, uuid=f"GPU-{i:03d}{g:02d}-5e7a-4c1d-9b2f-{self.rng.randrange(16**12):012x}",
                                     variance=variance))
            self.nodes.append(NodeState(
                name=name, cluster=cluster, server_type="DGX-H100" if "H100" in gpu_model else "HGX",
                gpu_model=gpu_model, rack=f"R{(i - 1) // 4 + 1:02d}", workload_id=node_to_job[name], gpus=gpus))
        # Start GPUs at their steady-state temperature so startup isn't an "anomaly".
        kinds = {w.job_id: w.kind for w in self.workloads}
        for node in self.nodes:
            prof = TRAINING_PROFILE if kinds[node.workload_id] == "training" else INFERENCE_PROFILE
            for g in node.gpus:
                g.temp = prof["temp"] + g.variance["temp"]
                g.mem_temp = g.temp + 8

    # ------------------------------------------------------------------ faults
    def inject(self, fault: Fault) -> Fault:
        if fault.node not in {n.name for n in self.nodes}:
            raise ValueError(f"unknown node {fault.node}")
        with self.lock:
            fault.severity = max(0.05, min(1.0, fault.severity))
            fault.started_at = time.time()
            self.faults.append(fault)
        return fault

    def clear(self, fault_id: str | None = None, node: str | None = None) -> int:
        """Remove faults. Clearing simulates remediation (e.g. GPU reset), which
        also resets volatile ECC counters / last XID on the affected GPUs."""
        with self.lock:
            removed = [f for f in self.faults
                       if (fault_id is None or f.id == fault_id) and (node is None or f.node == node)]
            self.faults = [f for f in self.faults if f not in removed]
            nodes = {n.name: n for n in self.nodes}
            for f in removed:
                for g in nodes[f.node].gpus:
                    if f.gpu is None or f.gpu == g.index:
                        g.last_xid = 0
                        for k in g.counters:
                            if k in ("ecc_dbe", "retired_dbe", "retired_sbe"):
                                g.counters[k] = 0.0
            return len(removed)

    def active_faults(self) -> list[Fault]:
        now = time.time()
        with self.lock:
            self.faults = [f for f in self.faults if f.active(now)]
            return list(self.faults)

    def _maybe_random_fault(self, dt: float) -> None:
        if not self.random_faults:
            return
        p = self.random_fault_rate_per_hour * dt / 3600.0
        if self.rng.random() < p:
            ftype = self.rng.choice(list(FaultType))
            node = self.rng.choice(self.nodes)
            gpu = self.rng.randrange(len(node.gpus)) if ftype in GPU_SCOPED else None
            self.inject(Fault(type=ftype, node=node.name, gpu=gpu, severity=self.rng.uniform(0.5, 1.0),
                              duration_s=self.rng.uniform(300, 1200), source="random"))

    # -------------------------------------------------------------------- tick
    def tick(self, dt: float = 5.0) -> FleetSnapshot:
        with self.lock:
            self.sim_time += dt
            self._maybe_random_fault(dt)
            faults = self.active_faults()
            wl = {w.job_id: w for w in self.workloads}
            snapshot_nodes = []
            for node in self.nodes:
                snapshot_nodes.append(self._tick_node(node, wl[node.workload_id], faults, dt))
            snap = FleetSnapshot(timestamp=utcnow(), source="simulator", nodes=snapshot_nodes,
                                 workloads=[w.model_copy() for w in self.workloads])
            self.last_snapshot = snap
            return snap

    def _n(self, sigma: float) -> float:
        return self.rng.gauss(0.0, sigma)

    def _tick_node(self, node: NodeState, workload: Workload, faults: list[Fault], dt: float) -> NodeSample:
        hw = hardware(node.gpu_model)
        prof = TRAINING_PROFILE if workload.kind == "training" else INFERENCE_PROFILE
        # Inference load follows a slow wave so the fleet isn't unrealistically flat.
        load_wave = 1.0 + (0.06 * math.sin(self.sim_time / 900.0) if workload.kind == "inference" else 0.0)

        node_faults = [f for f in faults if f.node == node.name]
        nf = {f.type: f.severity for f in node_faults if f.type not in GPU_SCOPED}

        # ---- node-level factors
        cpu_s = nf.get(FaultType.CPU_BOTTLENECK, 0.0)
        mem_s = nf.get(FaultType.HOST_MEMORY_PRESSURE, 0.0)
        net_s = nf.get(FaultType.NETWORK_DEGRADATION, 0.0)
        comm_s = nf.get(FaultType.COMMUNICATION_BOTTLENECK, 0.0)
        app_s = nf.get(FaultType.APP_REGRESSION, 0.0)

        input_factor = (1 - 0.45 * cpu_s) * (1 - 0.30 * mem_s)
        comm_factor = (1 - 0.30 * net_s) * (1 - 0.28 * comm_s)
        gpu_feed = input_factor * comm_factor  # how busy the GPUs can be kept

        gpus_out: list[GPUSample] = []
        gpu_perf: list[float] = []
        nvlink_penalty = 0.0
        for g in node.gpus:
            gf = {f.type: f.severity for f in node_faults if f.type in GPU_SCOPED and f.applies_to_gpu(node.name, g.index)}
            clock_f, hbm_f, util_f = 1.0, 1.0, gpu_feed
            temp_delta, power_f, mem_temp_delta = 0.0, 1.0, 0.0
            power_limit = hw["tdp_w"]
            throttle: list[str] = []
            mem_clock = hw["mem_clock"]
            link_gen, link_width, pcie_f = 5.0, 16.0, 1.0
            nvlink_f = 1.0
            sbe_per_min, crc_rate, replay_rate = 0.0, 0.0, 0.0
            xid = 0

            if (s := gf.get(FaultType.THERMAL)) is not None:
                temp_delta += 10 + 12 * s
                mem_temp_delta += 8 + 10 * s
                # Clock reduction kicks in as the GPU actually heats (thermal inertia).
                heat = max(0.0, min(1.0, (g.temp - (prof["temp"] + 6)) / 12.0))
                clock_f *= 1 - 0.25 * s * heat
                power_f *= 1 - 0.12 * s * heat
                if heat > 0.2:
                    throttle.append("sw_thermal_slowdown")
                if heat > 0.2 and s > 0.6:
                    throttle.append("hw_thermal_slowdown")
            if (s := gf.get(FaultType.POWER_CAP)) is not None:
                power_limit = hw["tdp_w"] * (1 - 0.40 * s)
                clock_f *= 1 - 0.22 * s
                temp_delta -= 4 * s
                throttle.append("sw_power_cap")
            if (s := gf.get(FaultType.MEMORY_BW)) is not None:
                hbm_f *= 1 - 0.35 * s
                mem_temp_delta += 6 * s
                sbe_per_min += 2 * s
                mem_clock *= 1 - 0.10 * s
            if (s := gf.get(FaultType.CLOCK_REDUCTION)) is not None:
                clock_f *= 1 - 0.30 * s
                temp_delta -= 3 * s
                power_f *= 1 - 0.18 * s
                throttle.append("applications_clocks_setting")
            if (s := gf.get(FaultType.ECC_ERRORS)) is not None:
                sbe_per_min += 20 + 80 * s
                hbm_f *= 0.97
                if s > 0.7:
                    xid = 48
            if (s := gf.get(FaultType.NVLINK_DEGRADATION)) is not None:
                nvlink_f *= 1 - 0.6 * s
                crc_rate += 2 + 20 * s
                nvlink_penalty = max(nvlink_penalty, s)
            if (s := gf.get(FaultType.PCIE_DEGRADATION)) is not None:
                link_width = 8.0
                if s > 0.5:
                    link_gen = 3.0
                pcie_f *= 0.45
                replay_rate += 2 + 20 * s
                util_f *= 0.94

            # Thermal model: first-order approach to target temperature.
            load = util_f * load_wave
            target = prof["temp"] + g.variance["temp"] + temp_delta - (1 - load) * 18
            alpha = 1 - math.exp(-dt / 30.0)
            g.temp += (target - g.temp) * alpha + self._n(0.25)
            g.mem_temp += (target + 8 + mem_temp_delta - g.mem_temp) * alpha + self._n(0.25)

            sm_clock = min(hw["max_sm_clock"], prof["sm_clock"] * (1 + g.variance["clock"]) * clock_f + self._n(8))
            power = prof["power"] * (1 + g.variance["power"]) * power_f * (0.55 + 0.45 * load) + self._n(6)
            power = min(power, power_limit - self.rng.uniform(0, 4)) if "sw_power_cap" in throttle else power
            dram_active = prof["dram_active"] * (1 + g.variance["hbm"]) * hbm_f * (0.5 + 0.5 * load)
            dram_active = max(0.0, dram_active + self._n(0.006))
            # Tensor pipe activity ≈ achieved compute throughput.
            perf = (clock_f ** 0.8) * (hbm_f ** 0.5) * util_f * (1 - 0.15 * nvlink_penalty)
            tensor = max(0.0, prof["tensor"] * perf * load_wave + self._n(0.006))
            gpu_perf.append((clock_f ** 0.8) * (hbm_f ** 0.5) * (1 - 0.15 * nvlink_penalty) * (0.94 if pcie_f < 1 else 1.0))

            # Counters
            g.counters["ecc_sbe"] += sbe_per_min * dt / 60.0 + (1 if self.rng.random() < 0.0005 else 0)
            if xid == 48 and self.rng.random() < 0.02 * dt:
                g.counters["ecc_dbe"] += 1
                g.counters["retired_dbe"] += 1
            if sbe_per_min > 30 and self.rng.random() < 0.01 * dt:
                g.counters["retired_sbe"] += 1
            g.counters["pcie_replay"] += replay_rate * dt
            g.counters["nvlink_crc"] += crc_rate * dt
            if xid:
                g.last_xid = xid
            elif FaultType.ECC_ERRORS not in gf and g.last_xid and self.rng.random() < 0.05 * dt:
                g.last_xid = 0  # XID gauge clears once the condition stops recurring

            util = max(0.0, min(100.0, prof["gpu_util"] * load + self._n(1.2)))
            metrics = {
                "gpu_util": round(util, 2),
                "sm_active": round(max(0.0, min(100.0, prof["sm_active"] * load * clock_f ** 0.1 + self._n(1.0))), 2),
                "sm_clock_mhz": round(sm_clock, 1),
                "mem_clock_mhz": round(mem_clock, 1),
                "temp_c": round(g.temp, 2),
                "mem_temp_c": round(g.mem_temp, 2),
                "power_w": round(max(60.0, power), 1),
                "power_limit_w": round(power_limit, 1),
                "mem_util": round(max(0.0, prof["mem_util"] * load * hbm_f + self._n(1.2)), 2),
                "mem_used_pct": round(prof["mem_used"] + self._n(0.4), 2),
                "hbm_bw_gbps": round(dram_active * hw["hbm_peak_gbps"], 1),
                "pcie_tx_gbps": round(max(0.0, prof["pcie_tx"] * pcie_f * load + self._n(0.3)), 3),
                "pcie_rx_gbps": round(max(0.0, prof["pcie_rx"] * pcie_f * load + self._n(0.5)), 3),
                "pcie_link_gen": link_gen,
                "pcie_link_width": link_width,
                "pcie_replay_rate": round(replay_rate, 3),
                "nvlink_bw_gbps": round(max(0.0, prof["nvlink"] * nvlink_f * comm_factor * load + self._n(4)), 1),
                "nvlink_crc_rate": round(crc_rate, 3),
                "ecc_sbe_rate": round(sbe_per_min, 3),
                "ecc_dbe_total": g.counters["ecc_dbe"],
                "retired_pages": g.counters["retired_sbe"] + g.counters["retired_dbe"],
                "xid_errors": float(g.last_xid),
                "perf_index": round(tensor * 100, 2),
                "throttle_mask": float(encode_throttle(throttle)),
            }
            procs = [{"pid": 41000 + node.gpus.index(g) * 7 + 3, "name": "python3" if workload.framework == "pytorch" else workload.framework,
                      "job_id": workload.job_id, "used_memory_mib": round(81559 * metrics["mem_used_pct"] / 100)}]
            gpus_out.append(GPUSample(node=node.name, index=g.index, uuid=g.uuid, model=node.gpu_model,
                                      metrics=metrics, throttle_reasons=throttle, processes=procs))

        # ---- node-level telemetry
        # Distributed training is synchronous: the slowest GPU gates the node.
        gpu_factor = min(gpu_perf) if workload.kind == "training" else sum(gpu_perf) / len(gpu_perf)
        app_factor = 1 - 0.30 * app_s
        node_factor = gpu_factor * input_factor * comm_factor * app_factor

        cpu_util = prof["cpu_util"] * load_wave + 58 * cpu_s + 10 * mem_s + 5 * app_s + self._n(1.5)
        cpu_util = max(1.0, min(99.5, cpu_util))
        iowait = max(0.0, 1.0 + 18 * mem_s + self._n(0.3))
        load1 = max(0.1, prof["load1"] * (1 + 2.5 * cpu_s + 0.8 * mem_s) + self._n(1.0))
        ctx_k = max(1.0, prof["ctx_k"] * (1 + 1.5 * cpu_s + 0.5 * mem_s) + self._n(5))
        mem_used = min(99.5, prof["mem_used_node"] + 42 * mem_s + self._n(0.3))
        psi = max(0.0, 0.3 + 30 * mem_s + 1.5 * cpu_s + self._n(0.15))
        net_gbps = max(0.0, prof["net_gbps"] * load_wave * (1 - 0.3 * net_s) + self._n(0.6))
        ib_gbps = max(0.0, prof["ib_gbps"] * (1 - 0.55 * net_s) * (1 - 0.35 * comm_s) * node_factor ** 0.3 + self._n(prof["ib_gbps"] * 0.015))
        net_err = 5 + 40 * net_s if net_s else 0.0
        net_drop = 20 + 100 * net_s if net_s else 0.0
        ib_symbol = 1 + 10 * net_s if net_s else 0.0
        latency = prof["latency_us"] * (1 + 3 * net_s + 0.3 * comm_s) + abs(self._n(0.05))
        busbw = prof["nccl_busbw"] * (1 - 0.5 * net_s) * (1 - 0.45 * comm_s) * (1 - 0.35 * nvlink_penalty) + self._n(prof["nccl_busbw"] * 0.012)
        comm_ratio = prof["comm_ratio"] + 25 * net_s + 22 * comm_s + 12 * nvlink_penalty + self._n(0.6)
        dl_wait = max(0.0, prof["dl_wait"] + 30 * cpu_s + 18 * mem_s + self._n(0.4))
        throughput = prof["throughput"] * load_wave * node_factor * (1 + self._n(0.008))
        step_ms = prof["step_ms"] / max(0.05, node_factor) * (1 + self._n(0.006))

        c = node.counters
        cores = self.cpu_cores
        busy = cpu_util / 100.0
        c["cpu_idle"] += cores * dt * (1 - busy)
        c["cpu_user"] += cores * dt * busy * 0.8
        c["cpu_system"] += cores * dt * busy * 0.2
        c["cpu_iowait"] += cores * dt * iowait / 100.0
        c["ctx"] += ctx_k * 1000 * dt
        c["net_rx"] += net_gbps * 1e9 / 8 * dt * 0.55
        c["net_tx"] += net_gbps * 1e9 / 8 * dt * 0.45
        c["net_errs"] += net_err * dt
        c["net_drop"] += net_drop * dt
        c["ib_rx"] += ib_gbps * 1e9 / 8 * dt / 2
        c["ib_tx"] += ib_gbps * 1e9 / 8 * dt / 2
        c["ib_symbol"] += ib_symbol * dt
        c["psi_mem"] += psi / 100.0 * dt

        self.last_derived[node.name] = {"cpu_freq_mhz": 3000 - 400 * cpu_s * 0.2 + self._n(12)}
        node_metrics = {
            "cpu_util": round(cpu_util, 2),
            "cpu_iowait": round(iowait, 2),
            "load1": round(load1, 2),
            "cpu_freq_mhz": round(self.last_derived[node.name]["cpu_freq_mhz"], 1),
            "ctx_switches_k": round(ctx_k, 1),
            "mem_used_pct": round(mem_used, 2),
            "mem_psi_some": round(psi, 2),
            "net_rx_gbps": round(net_gbps * 0.55, 3),
            "net_tx_gbps": round(net_gbps * 0.45, 3),
            "net_err_rate": round(net_err, 3),
            "net_drop_rate": round(net_drop, 3),
            "ib_rx_gbps": round(ib_gbps / 2, 2),
            "ib_tx_gbps": round(ib_gbps / 2, 2),
            "ib_symbol_err_rate": round(ib_symbol, 3),
            "net_latency_us": round(latency, 3),
            "nccl_busbw_gbps": round(max(0.0, busbw), 2),
            "nccl_comm_ratio": round(max(0.0, min(95.0, comm_ratio)), 2),
            "dataloader_wait_pct": round(min(95.0, dl_wait), 2),
            "throughput": round(throughput, 1),
            "step_time_ms": round(step_ms, 1),
        }
        return NodeSample(node=node.name, cluster=node.cluster, server_type=node.server_type,
                          gpu_model=node.gpu_model, rack=node.rack, driver_version="550.90.07",
                          cuda_version="12.4", workload_id=workload.job_id, metrics=node_metrics, gpus=gpus_out)

    def warmup(self, seconds: float = 300.0, dt: float = 5.0) -> FleetSnapshot:
        snap = None
        for _ in range(int(seconds / dt)):
            snap = self.tick(dt)
        return snap  # type: ignore[return-value]
