"""Fault definitions for the GPU cluster simulator (demo / test mode)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class FaultType(str, Enum):
    THERMAL = "thermal"
    POWER_CAP = "power_cap"
    MEMORY_BW = "memory_bw"
    CLOCK_REDUCTION = "clock_reduction"
    ECC_ERRORS = "ecc_errors"
    NVLINK_DEGRADATION = "nvlink_degradation"
    PCIE_DEGRADATION = "pcie_degradation"
    CPU_BOTTLENECK = "cpu_bottleneck"
    HOST_MEMORY_PRESSURE = "host_memory_pressure"
    NETWORK_DEGRADATION = "network_degradation"
    COMMUNICATION_BOTTLENECK = "communication_bottleneck"
    APP_REGRESSION = "app_regression"


# Faults that act on individual GPUs (a gpu index may be given) vs. the node.
GPU_SCOPED = {
    FaultType.THERMAL,
    FaultType.POWER_CAP,
    FaultType.MEMORY_BW,
    FaultType.CLOCK_REDUCTION,
    FaultType.ECC_ERRORS,
    FaultType.NVLINK_DEGRADATION,
    FaultType.PCIE_DEGRADATION,
}

FAULT_DESCRIPTIONS: dict[FaultType, str] = {
    FaultType.THERMAL: "Cooling/airflow problem: GPU heats up and thermally throttles clocks.",
    FaultType.POWER_CAP: "Power limit reduced (PSU/BMC policy): GPU is power-capped and clocks drop.",
    FaultType.MEMORY_BW: "HBM degradation: memory bandwidth drops, correctable ECC errors rise.",
    FaultType.CLOCK_REDUCTION: "Application clocks locked low (misconfiguration) without thermal/power cause.",
    FaultType.ECC_ERRORS: "Failing HBM: correctable and (at high severity) uncorrectable ECC errors, XID 48.",
    FaultType.NVLINK_DEGRADATION: "NVLink lane/link errors: NVLink bandwidth drops, CRC errors rise.",
    FaultType.PCIE_DEGRADATION: "PCIe link downtrained (width/gen reduced) with replay errors.",
    FaultType.CPU_BOTTLENECK: "Host CPU saturated (data loader / preprocessing) starving the GPUs.",
    FaultType.HOST_MEMORY_PRESSURE: "Host memory pressure / swapping slowing input pipeline.",
    FaultType.NETWORK_DEGRADATION: "InfiniBand port degradation: errors, drops, lower bandwidth, higher latency.",
    FaultType.COMMUNICATION_BOTTLENECK: "NCCL collective slowdown (topology/config) without link errors.",
    FaultType.APP_REGRESSION: "Software regression: workload throughput drops while GPU telemetry looks normal.",
}


@dataclass
class Fault:
    type: FaultType
    node: str
    gpu: int | None = None
    severity: float = 0.8
    duration_s: float | None = 900.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    started_at: float = field(default_factory=time.time)
    source: str = "operator"

    def active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.duration_s is None or now < self.started_at + self.duration_s

    def applies_to_gpu(self, node: str, index: int) -> bool:
        return self.node == node and (self.gpu is None or self.gpu == index)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "node": self.node,
            "gpu": self.gpu,
            "severity": self.severity,
            "duration_s": self.duration_s,
            "started_at": self.started_at,
            "source": self.source,
            "description": FAULT_DESCRIPTIONS[self.type],
        }
