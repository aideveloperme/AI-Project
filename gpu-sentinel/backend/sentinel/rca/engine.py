"""Rule-based multi-metric correlation & root-cause engine.

Turns many per-metric anomaly signals on one node into a small set of ranked
*hypotheses*. Every hypothesis keeps the evidence it was derived from
(``evidence_for`` / ``evidence_against``) so the UI and the AI layer can show
exactly why a cause was suggested — nothing is asserted without evidence.

Confidence model (deliberately simple and explainable):
    confidence = base + (matched supporting / total supporting) × (ceiling − base)
                 − 0.15 × matched contradicting
Labels: high ≥ 0.75, medium ≥ 0.5, else low. Only hardware-counter evidence
(e.g. ECC DBE, XID, thermal throttle flags) can push a rule to "high".
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from sentinel.analytics.detectors import SEV_ORDER, AnomalySignal
from sentinel.analytics.peers import PeerComparison
from sentinel.telemetry.models import FleetSnapshot


# ----------------------------------------------------------------- evidence
@dataclass
class NodeEvidence:
    node: str
    gpu_count: int
    signals: list[AnomalySignal]
    throttle: dict[str, list[int]]          # reason -> gpu indices
    perf: PeerComparison | None             # node throughput vs peers
    peer: dict[str, PeerComparison]         # node-level comparisons
    workload_kind: str | None = None

    def sigs(self, level: str, metric: str, direction: str | None = None) -> list[AnomalySignal]:
        return [s for s in self.signals if s.level == level and s.metric == metric
                and (direction is None or s.direction == direction)]

    def categories(self, level: str | None = None) -> set[str]:
        return {s.category for s in self.signals if level is None or s.level == level}

    @property
    def perf_deviation_pct(self) -> float | None:
        return None if self.perf is None else self.perf.deviation_pct


CondResult = tuple[bool, list[str]]


@dataclass
class Cond:
    description: str
    fn: Callable[[NodeEvidence], CondResult]

    def __call__(self, ev: NodeEvidence) -> CondResult:
        return self.fn(ev)


def _fmt(s: AnomalySignal) -> str:
    where = f"GPU {s.gpu_index}" if s.gpu_index is not None else "node"
    return f"{where}: {s.description}"


def sig(level: str, metric: str, direction: str, label: str, min_count: int = 1) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        found = ev.sigs(level, metric, direction)
        return len(found) >= min_count, [_fmt(s) for s in found[:4]] + (
            [f"... and {len(found) - 4} more GPUs"] if len(found) > 4 else [])
    return Cond(label, fn)


def localized(level: str, metric: str, direction: str, label: str) -> Cond:
    """Signal present on some — but not all — GPUs of the node, and those GPUs are otherwise busy."""
    def fn(ev: NodeEvidence) -> CondResult:
        found = ev.sigs(level, metric, direction)
        starved = {s.gpu_index for s in ev.sigs("gpu", "gpu_util", "low")}
        found = [s for s in found if s.gpu_index not in starved]
        ok = bool(found) and len(found) < max(2, ev.gpu_count)
        return ok, [_fmt(s) for s in found[:4]]
    return Cond(label, fn)


def throttle(reasons: tuple[str, ...], label: str) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        hits = [(r, ev.throttle[r]) for r in reasons if ev.throttle.get(r)]
        return bool(hits), [f"Clock throttle reason '{r}' active on GPU(s) {', '.join(map(str, g))}" for r, g in hits]
    return Cond(label, fn)


def perf_degraded(threshold_pct: float = 8.0) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        d = ev.perf_deviation_pct
        if d is None or d > -threshold_pct:
            return False, []
        p = ev.perf
        return True, [f"Workload throughput {p.value:.0f} is {abs(d):.1f}% below peer median {p.peer.median:.0f} (n={p.peer.n})"]
    return Cond(f"workload throughput ≥{threshold_pct:.0f}% below peers", fn)


def no_signals(categories: set[str], label: str, level: str | None = None) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        hit = ev.categories(level) & categories
        return not hit, ([f"No anomalies in: {', '.join(sorted(categories))}"] if not hit else [])
    return Cond(label, fn)


def any_of(label: str, *conds: Cond) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        ok, evid = False, []
        for c in conds:
            m, e = c(ev)
            if m:
                ok = True
                evid += e
        return ok, evid
    return Cond(label, fn)


def not_(c: Cond, label: str) -> Cond:
    def fn(ev: NodeEvidence) -> CondResult:
        m, _ = c(ev)
        return not m, []
    return Cond(label, fn)


# -------------------------------------------------------------------- rules
@dataclass
class Rule:
    id: str
    title: str
    category: str
    required: list[Cond]
    supporting: list[Cond]
    contradicting: list[Cond]
    explanation: str
    actions: list[str]
    base: float = 0.45
    ceiling: float = 0.9
    severity_floor: str = "warning"
    # If any of these match, the rule may reach "high" confidence.
    hard_evidence: list[Cond] = field(default_factory=list)


@dataclass
class Hypothesis:
    rule_id: str
    title: str
    category: str
    confidence: float
    confidence_label: str
    evidence_for: list[str]
    evidence_against: list[str]
    explanation: str
    recommended_actions: list[str]
    affected_gpus: list[int]
    severity_floor: str = "warning"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 2)
        return d


def _label(c: float) -> str:
    return "high" if c >= 0.75 else "medium" if c >= 0.5 else "low"


GPU_HW = {"thermal", "clocks", "power", "memory", "ecc", "nvlink", "pcie", "health"}

RULES: list[Rule] = [
    Rule(
        id="thermal_throttling", title="Thermal throttling", category="thermal",
        required=[
            any_of("GPU temperature above peers/thresholds",
                   sig("gpu", "temp_c", "high", "GPU temperature high"),
                   throttle(("sw_thermal_slowdown", "hw_thermal_slowdown"), "thermal throttle flag")),
            sig("gpu", "sm_clock_mhz", "low", "SM clock below peers"),
        ],
        supporting=[
            throttle(("sw_thermal_slowdown", "hw_thermal_slowdown"), "NVML thermal throttle reason set"),
            sig("gpu", "mem_temp_c", "high", "HBM temperature high"),
            sig("gpu", "perf_index", "low", "Tensor activity below peers"),
            perf_degraded(),
        ],
        contradicting=[throttle(("sw_power_cap",), "power-cap throttle (points to power, not cooling)")],
        hard_evidence=[throttle(("hw_thermal_slowdown", "sw_thermal_slowdown"), "thermal throttle flag")],
        explanation="Elevated GPU temperature coincides with reduced SM clocks. This pattern is consistent with the "
                    "GPU protecting itself by lowering clocks (thermal slowdown), commonly caused by restricted "
                    "airflow, a failed fan, high inlet temperature or degraded thermal interface material.",
        actions=[
            "Check rack/chassis cooling: inlet temperature, fan speeds and airflow obstructions (BMC/IPMI sensors).",
            "Compare this GPU's thermal history with neighbouring GPUs in the same chassis and rack.",
            "Inspect throttle reasons: `nvidia-smi -q -d PERFORMANCE,TEMPERATURE -i <gpu>`.",
            "If isolated to one GPU, schedule a maintenance window to run `dcgmi diag -r 3` and inspect the heatsink/TIM.",
        ],
        base=0.5, severity_floor="warning"),
    Rule(
        id="power_throttling", title="Power capping / power-related throttling", category="power",
        required=[
            sig("gpu", "sm_clock_mhz", "low", "SM clock below peers"),
            any_of("power cap indicators",
                   throttle(("sw_power_cap", "hw_power_brake_slowdown"), "power throttle flag"),
                   sig("gpu", "power_limit_w", "low", "enforced power limit below peers")),
        ],
        supporting=[
            sig("gpu", "power_limit_w", "low", "Enforced power limit lower than peers"),
            throttle(("sw_power_cap",), "NVML power-cap throttle reason set"),
            sig("gpu", "perf_index", "low", "Tensor activity below peers"),
            perf_degraded(),
            not_(sig("gpu", "temp_c", "high", "temp"), "temperature not elevated"),
        ],
        contradicting=[sig("gpu", "temp_c", "high", "GPU temperature elevated (may be thermal)")],
        hard_evidence=[throttle(("sw_power_cap", "hw_power_brake_slowdown"), "power throttle flag")],
        explanation="Clocks are reduced while the GPU is held at a lower power limit than comparable GPUs. This is "
                    "consistent with a power cap (administrative limit, BMC/PSU policy or power-brake event) rather "
                    "than a thermal problem.",
        actions=[
            "Compare the enforced power limit with peers: `nvidia-smi -q -d POWER -i <gpu>`.",
            "Check for administrative power limits (`nvidia-smi -pl`), datacenter power-capping policies and BMC events.",
            "Review PSU health and redundancy status on the chassis.",
            "If the limit was changed unintentionally, restore the default through your change-management process.",
        ]),
    Rule(
        id="clock_misconfiguration", title="GPU clocks reduced without thermal/power cause", category="clocks",
        required=[
            sig("gpu", "sm_clock_mhz", "low", "SM clock below peers"),
            not_(throttle(("sw_thermal_slowdown", "hw_thermal_slowdown", "sw_power_cap", "hw_power_brake_slowdown"),
                          "x"), "no thermal/power throttle reason"),
            not_(sig("gpu", "temp_c", "high", "x"), "temperature normal"),
        ],
        supporting=[
            throttle(("applications_clocks_setting",), "Application-clocks setting limits clocks"),
            sig("gpu", "power_w", "low", "Power draw lower than peers (consistent with lower clocks)"),
            perf_degraded(),
        ],
        contradicting=[],
        hard_evidence=[throttle(("applications_clocks_setting",), "application clocks")],
        explanation="SM clocks are lower than peers but temperature and power limits are normal. This points to a "
                    "configuration cause such as locked application clocks, a non-default clock policy or a "
                    "profile applied by a job prolog.",
        actions=[
            "Inspect clock settings: `nvidia-smi -q -d CLOCK -i <gpu>` (applications vs. max clocks).",
            "Check job prolog/epilog scripts and node configuration management for clock locking (`nvidia-smi -lgc`).",
            "Compare driver version and persistence mode with healthy peers.",
            "Reset clocks to defaults through change management if the lock is unintended.",
        ]),
    Rule(
        id="hbm_degradation", title="HBM / memory-bandwidth degradation", category="memory",
        required=[sig("gpu", "hbm_bw_gbps", "low", "HBM bandwidth below peers"),
                  not_(sig("gpu", "gpu_util", "low", "x"), "GPU utilization normal (GPU is busy)")],
        supporting=[
            sig("gpu", "ecc_sbe_rate", "high", "Correctable ECC errors elevated"),
            sig("gpu", "mem_temp_c", "high", "HBM temperature elevated"),
            sig("gpu", "mem_clock_mhz", "low", "Memory clock below peers"),
            sig("gpu", "perf_index", "low", "Tensor activity below peers"),
            perf_degraded(),
        ],
        contradicting=[
            sig("gpu", "gpu_util", "low", "GPU utilization low (GPU may be starved rather than memory-bound)"),
            sig("gpu", "sm_clock_mhz", "low", "SM clocks also low (may be throttling)"),
        ],
        hard_evidence=[sig("gpu", "ecc_sbe_rate", "high", "sbe")],
        explanation="Device-memory bandwidth is below comparable GPUs while the GPU is otherwise busy. Possible "
                    "contributors are HBM degradation (often accompanied by rising correctable ECC errors), a "
                    "reduced memory clock, or a workload-specific access pattern on this device.",
        actions=[
            "Check ECC counters and retired/remapped rows: `nvidia-smi -q -d ECC,ROW_REMAPPER -i <gpu>`.",
            "Compare memory clock and HBM temperature with peers.",
            "Run a memory bandwidth diagnostic (`dcgmi diag -r 3` includes memory tests) during a maintenance window.",
            "If ECC errors keep increasing, open an RMA case with the hardware vendor.",
        ]),
    Rule(
        id="ecc_hardware_fault", title="GPU memory errors (ECC / XID)", category="ecc",
        required=[any_of("ECC or XID errors",
                         sig("gpu", "ecc_sbe_rate", "high", "Correctable ECC error rate high"),
                         sig("gpu", "ecc_dbe_total", "high", "Uncorrectable (DBE) ECC errors"),
                         sig("gpu", "xid_errors", "high", "XID error reported"),
                         sig("gpu", "retired_pages", "high", "Retired memory pages"))],
        supporting=[
            sig("gpu", "ecc_dbe_total", "high", "Uncorrectable (double-bit) ECC errors present"),
            sig("gpu", "xid_errors", "high", "XID error reported by driver"),
            sig("gpu", "retired_pages", "high", "Memory pages retired"),
        ],
        contradicting=[],
        hard_evidence=[sig("gpu", "ecc_dbe_total", "high", "dbe"), sig("gpu", "xid_errors", "high", "xid")],
        explanation="The GPU reports memory error counters above normal. Uncorrectable errors or XID 48/63/64 "
                    "events indicate a hardware memory fault that can crash jobs or corrupt results; correctable "
                    "errors alone are an early-warning signal.",
        actions=[
            "Drain the node from the scheduler if uncorrectable errors are present (`scontrol update nodename=<n> state=drain` / `kubectl cordon`).",
            "Collect `nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER` and kernel logs for XID events (`dmesg | grep -i xid`).",
            "Run `dcgmi diag -r 3` during maintenance; follow NVIDIA XID guidance for the reported code.",
            "Track recurrence; repeated errors on the same GPU warrant an RMA.",
        ],
        base=0.55, severity_floor="critical"),
    Rule(
        id="cpu_bottleneck", title="Host CPU / input-pipeline bottleneck", category="cpu",
        required=[
            sig("gpu", "gpu_util", "low", "GPU utilization below peers"),
            sig("node", "cpu_util", "high", "Host CPU utilization high"),
        ],
        supporting=[
            sig("node", "dataloader_wait_pct", "high", "GPUs waiting on input data"),
            sig("node", "load1", "high", "Load average above peers"),
            sig("node", "ctx_switches_k", "high", "Context switches elevated"),
            no_signals({"clocks", "thermal"}, "GPU clocks/thermals normal", level="gpu"),
            perf_degraded(),
        ],
        contradicting=[sig("node", "nccl_comm_ratio", "high", "Communication time elevated (may be network-bound)"),
                       sig("node", "mem_used_pct", "high", "System memory nearly full (may be memory pressure)")],
        explanation="GPUs are under-utilised while the host CPU is saturated and GPUs report waiting for input. "
                    "This is consistent with the data loader / preprocessing path not keeping up, rather than a GPU fault.",
        actions=[
            "Identify top CPU consumers on the host (`top`/`pidstat`) and check for stray processes outside the job.",
            "Check data-loader worker count, CPU pinning/NUMA affinity and container CPU limits for this job.",
            "Compare storage read throughput and iowait with peer nodes.",
            "Verify CPU frequency governor and BIOS power profile match healthy peers.",
        ]),
    Rule(
        id="host_memory_pressure", title="Host memory pressure", category="memory",
        required=[any_of("host memory pressure", sig("node", "mem_psi_some", "high", "Memory PSI high"),
                         sig("node", "mem_used_pct", "high", "System memory usage high"))],
        supporting=[
            sig("node", "cpu_iowait", "high", "iowait elevated"),
            sig("node", "dataloader_wait_pct", "high", "GPUs waiting on input data"),
            sig("node", "mem_used_pct", "high", "System memory usage high"),
            sig("gpu", "gpu_util", "low", "GPU utilization below peers"),
            perf_degraded(),
        ],
        contradicting=[sig("node", "cpu_util", "high", "CPU saturated (CPU bottleneck may be primary)")],
        explanation="The host is under memory pressure (stalls on memory reclaim / swapping). This can slow the "
                    "input pipeline and pinned-memory transfers, leaving GPUs idle.",
        actions=[
            "Check memory usage and PSI on the host (`free -g`, `/proc/pressure/memory`), and for swapping.",
            "Look for memory leaks in data-loader workers or co-located processes.",
            "Verify container/cgroup memory limits for the job.",
        ]),
    Rule(
        id="network_degradation", title="Network / InfiniBand degradation", category="network",
        required=[
            any_of("fabric anomalies",
                   sig("node", "ib_rx_gbps", "low", "IB receive bandwidth low"),
                   sig("node", "net_err_rate", "high", "Network errors"),
                   sig("node", "ib_symbol_err_rate", "high", "IB symbol errors"),
                   sig("node", "net_drop_rate", "high", "Packet drops"),
                   sig("node", "net_latency_us", "high", "Fabric latency high")),
            any_of("communication impact",
                   sig("node", "nccl_busbw_gbps", "low", "NCCL bandwidth low"),
                   sig("node", "nccl_comm_ratio", "high", "Communication time high"),
                   sig("gpu", "gpu_util", "low", "GPU utilization low")),
        ],
        supporting=[
            sig("node", "ib_symbol_err_rate", "high", "InfiniBand symbol errors"),
            sig("node", "net_err_rate", "high", "Link errors"),
            sig("node", "net_drop_rate", "high", "Packet drops"),
            sig("node", "net_latency_us", "high", "Fabric latency above peers"),
            sig("node", "nccl_busbw_gbps", "low", "NCCL all-reduce bandwidth below peers"),
            perf_degraded(),
        ],
        contradicting=[not_(any_of("errors", sig("node", "net_err_rate", "high", "e"), sig("node", "ib_symbol_err_rate", "high", "s"),
                                   sig("node", "net_drop_rate", "high", "d")), "No link errors or drops observed")],
        hard_evidence=[sig("node", "ib_symbol_err_rate", "high", "sym"), sig("node", "net_err_rate", "high", "err")],
        explanation="Network counters on this node show reduced bandwidth and/or link errors while collective "
                    "communication slowed. In synchronous distributed training, one degraded link slows every rank.",
        actions=[
            "Check the HCA port state and counters: `ibstat`, `perfquery -x`, `ibdiagnet` on the fabric.",
            "Inspect the cable/transceiver and the switch port; compare error counters with peer ports.",
            "Verify link speed/width negotiated at the expected rate (e.g. NDR 400G).",
            "Consider draining the node from multi-node jobs until the link is repaired.",
        ]),
    Rule(
        id="communication_bottleneck", title="Collective-communication (NCCL) bottleneck", category="communication",
        required=[any_of("NCCL degradation", sig("node", "nccl_busbw_gbps", "low", "NCCL bus bandwidth low"),
                         sig("node", "nccl_comm_ratio", "high", "Communication share of step time high"))],
        supporting=[
            sig("node", "nccl_comm_ratio", "high", "Communication share of step time above peers"),
            sig("gpu", "gpu_util", "low", "GPU utilization below peers (waiting on collectives)"),
            perf_degraded(),
        ],
        contradicting=[
            sig("node", "net_err_rate", "high", "Link errors (points to network hardware)"),
            sig("node", "ib_symbol_err_rate", "high", "IB symbol errors (points to network hardware)"),
            sig("gpu", "nvlink_crc_rate", "high", "NVLink errors (points to NVLink)"),
        ],
        explanation="Collective operations on this node are slower than on peers without clear link-level errors. "
                    "Possible contributors: NCCL configuration/topology (e.g. wrong NIC/GPU affinity, disabled "
                    "GPUDirect RDMA), congestion, or an imbalanced job placement.",
        actions=[
            "Compare NCCL environment (NCCL_IB_HCA, NCCL_SOCKET_IFNAME, NCCL_NET_GDR_LEVEL) with healthy nodes.",
            "Run `nccl-tests` (all_reduce_perf) between this node and a healthy peer during a maintenance window.",
            "Check GPU↔NIC topology/affinity (`nvidia-smi topo -m`) and PCIe ACS settings.",
            "Review switch congestion counters on the fabric for this node's uplinks.",
        ], base=0.4, ceiling=0.75),
    Rule(
        id="nvlink_degradation", title="NVLink degradation", category="nvlink",
        required=[any_of("NVLink anomalies", sig("gpu", "nvlink_crc_rate", "high", "NVLink CRC errors"),
                         localized("gpu", "nvlink_bw_gbps", "low", "NVLink bandwidth low on specific GPUs"))],
        supporting=[
            sig("gpu", "nvlink_crc_rate", "high", "NVLink CRC error rate elevated"),
            sig("node", "nccl_busbw_gbps", "low", "NCCL bandwidth below peers"),
            sig("node", "nccl_comm_ratio", "high", "Communication share of step time elevated"),
            perf_degraded(),
        ],
        contradicting=[sig("gpu", "gpu_util", "low", "GPU utilization low on all GPUs (feed problem more likely)", min_count=8)],
        hard_evidence=[sig("gpu", "nvlink_crc_rate", "high", "crc")],
        explanation="NVLink bandwidth on specific GPU(s) is below peers and/or NVLink CRC errors are rising. "
                    "Degraded NVLink slows intra-node collectives and tensor-parallel communication.",
        actions=[
            "Check NVLink status and error counters: `nvidia-smi nvlink -s` and `nvidia-smi nvlink -e -i <gpu>`.",
            "Check NVSwitch/fabric manager logs for link training failures.",
            "Run `dcgmi diag -r 2` (includes NVLink tests) during a maintenance window.",
        ]),
    Rule(
        id="pcie_degradation", title="PCIe link degradation", category="pcie",
        required=[any_of("PCIe anomalies", sig("gpu", "pcie_link_width", "low", "PCIe link width reduced"),
                         sig("gpu", "pcie_link_gen", "low", "PCIe link generation reduced"),
                         sig("gpu", "pcie_replay_rate", "high", "PCIe replay errors"))],
        supporting=[
            sig("gpu", "pcie_replay_rate", "high", "PCIe replay counter increasing"),
            sig("gpu", "pcie_rx_gbps", "low", "PCIe RX throughput below peers"),
            sig("gpu", "pcie_tx_gbps", "low", "PCIe TX throughput below peers"),
            perf_degraded(),
        ],
        contradicting=[],
        hard_evidence=[sig("gpu", "pcie_link_width", "low", "w"), sig("gpu", "pcie_link_gen", "low", "g")],
        explanation="The GPU's PCIe link trained at a lower width/generation than peers and/or shows replay "
                    "errors. This limits host↔GPU transfers (data loading, checkpointing, GPUDirect).",
        actions=[
            "Check link status: `nvidia-smi -q -d PCIE -i <gpu>` and `lspci -vv -s <bdf> | grep -i lnksta`.",
            "Reseat the GPU / riser and inspect for physical damage during maintenance.",
            "Check BIOS PCIe settings and firmware versions against healthy peers.",
        ]),
    Rule(
        id="software_regression", title="Application / software-layer slowdown", category="software",
        required=[
            perf_degraded(),
            no_signals(GPU_HW | {"compute", "performance"}, "GPU hardware telemetry normal", level="gpu"),
            no_signals({"cpu", "network", "communication"}, "host CPU and network normal", level="node"),
        ],
        supporting=[sig("node", "step_time_ms", "high", "Step time above peers")],
        contradicting=[],
        explanation="The workload on this node is slower than peers, but GPU, CPU and network telemetry are all "
                    "within normal ranges. The evidence points away from hardware; investigate the application "
                    "or software layer (code/config/version change, data skew, different batch size).",
        actions=[
            "Compare job configuration, container image and library versions (CUDA, cuDNN, NCCL, PyTorch) with peer nodes.",
            "Check whether this rank receives more/different data (data skew) or a different batch size.",
            "Profile a short window with PyTorch profiler / Nsight Systems on this node and a healthy peer.",
            "Review recent deployments or config changes for this workload.",
        ], base=0.4, ceiling=0.65),
]


class RCAEngine:
    def __init__(self, rules: list[Rule] | None = None, min_confidence: float = 0.3):
        self.rules = rules or RULES
        self.min_confidence = min_confidence

    def build_evidence(self, node: str, snap: FleetSnapshot, signals: list[AnomalySignal],
                       peers: dict[str, dict[str, PeerComparison]]) -> NodeEvidence:
        ns = snap.node(node)
        thr: dict[str, list[int]] = {}
        if ns:
            for g in ns.gpus:
                for r in g.throttle_reasons:
                    thr.setdefault(r, []).append(g.index)
        kind = None
        if ns and ns.workload_id:
            w = snap.workload(ns.workload_id)
            kind = w.kind if w else None
        node_peers = peers.get(node, {})
        return NodeEvidence(node=node, gpu_count=len(ns.gpus) if ns else 0,
                            signals=[s for s in signals if s.node == node], throttle=thr,
                            perf=node_peers.get("throughput"), peer=node_peers, workload_kind=kind)

    def evaluate(self, ev: NodeEvidence) -> list[Hypothesis]:
        out = []
        for r in self.rules:
            req = [c(ev) for c in r.required]
            if not all(m for m, _ in req):
                continue
            ev_for = [e for _, es in req for e in es]
            sup = [(c, *c(ev)) for c in r.supporting]
            matched_sup = [(c, e) for c, m, e in sup if m]
            for c, e in matched_sup:
                ev_for += e or [c.description]
            con = [(c, *c(ev)) for c in r.contradicting]
            matched_con = [(c, e) for c, m, e in con if m]
            ev_against = [c.description for c, _ in matched_con]
            ev_against += [f"Not observed: {c.description}" for c, m, _ in sup if not m]
            frac = len(matched_sup) / len(r.supporting) if r.supporting else 1.0
            conf = r.base + frac * (r.ceiling - r.base) - 0.15 * len(matched_con)
            hard = any(c(ev)[0] for c in r.hard_evidence)
            if not hard:
                conf = min(conf, 0.74)
            conf = max(0.05, min(r.ceiling, conf))
            if conf < self.min_confidence:
                continue
            gpus = sorted({s.gpu_index for s in ev.signals
                           if s.gpu_index is not None and s.category in _rule_categories(r)})
            out.append(Hypothesis(rule_id=r.id, title=r.title, category=r.category, confidence=conf,
                                  confidence_label=_label(conf), evidence_for=_dedupe(ev_for),
                                  evidence_against=_dedupe(ev_against), explanation=r.explanation,
                                  recommended_actions=r.actions, affected_gpus=gpus, severity_floor=r.severity_floor))
        out.sort(key=lambda h: h.confidence, reverse=True)
        if not out and ev.perf_deviation_pct is not None and ev.perf_deviation_pct <= -8:
            out.append(Hypothesis(
                rule_id="unexplained_degradation", title="Unexplained performance degradation", category="performance",
                confidence=0.25, confidence_label="low", evidence_for=perf_degraded()(ev)[1] + [_fmt(s) for s in ev.signals[:6]],
                evidence_against=[], explanation="Throughput is below peers but the observed signals do not match a known "
                "pattern. Treat as an open investigation.", recommended_actions=[
                    "Compare the full metric set of this node with a healthy peer on the Node Details page.",
                    "Check recent changes (firmware, driver, job configuration) on this node.",
                    "Run controlled diagnostics (`dcgmi diag -r 2`) during a maintenance window."],
                affected_gpus=sorted({s.gpu_index for s in ev.signals if s.gpu_index is not None})))
        return out


_RULE_CATS = {
    "thermal_throttling": {"thermal", "clocks", "power", "performance"},
    "power_throttling": {"power", "clocks", "performance"},
    "clock_misconfiguration": {"clocks", "power", "performance"},
    "hbm_degradation": {"memory", "ecc", "performance"},
    "ecc_hardware_fault": {"ecc", "health"},
    "nvlink_degradation": {"nvlink"},
    "pcie_degradation": {"pcie"},
}


def _rule_categories(r: Rule) -> set[str]:
    return _RULE_CATS.get(r.id, {"compute", "performance", "memory", "nvlink", "pcie"})


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def node_severity(ev: NodeEvidence, hyps: list[Hypothesis]) -> str:
    """Critical only on hard thresholds, critical-floor rules or ≥20% throughput loss.
    Statistical (peer/temporal) deviations alone cap at warning."""
    sev = max((SEV_ORDER[s.severity] if s.method == "static_threshold" else min(1, SEV_ORDER[s.severity])
               for s in ev.signals), default=0)
    if hyps:
        sev = max(sev, SEV_ORDER[hyps[0].severity_floor])
    d = ev.perf_deviation_pct
    if d is not None and d <= -20:
        sev = 2
    return {0: "info", 1: "warning", 2: "critical"}[sev]
