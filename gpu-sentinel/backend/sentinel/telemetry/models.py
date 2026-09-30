"""Vendor-neutral, canonical telemetry model.

Every collector (Prometheus/DCGM, in-process simulator, future AMD/Intel
sources) produces a :class:`FleetSnapshot`. Everything downstream (peer
benchmarking, anomaly detection, RCA) only ever sees canonical metric names
defined in :mod:`sentinel.telemetry.catalog`, never vendor field names.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GPUSample(BaseModel):
    node: str
    index: int
    uuid: str
    model: str
    vendor: str = "nvidia"
    metrics: dict[str, float] = Field(default_factory=dict)
    throttle_reasons: list[str] = Field(default_factory=list)
    processes: list[dict] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.node}/gpu{self.index}"


class Workload(BaseModel):
    job_id: str
    name: str
    scheduler: str  # slurm | kubernetes | standalone
    framework: str = "pytorch"
    kind: str = "training"  # training | inference
    user: str | None = None
    namespace: str | None = None
    nodes: list[str] = Field(default_factory=list)
    throughput_unit: str = "samples/s"


class NodeSample(BaseModel):
    node: str
    cluster: str
    server_type: str
    gpu_model: str
    rack: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    workload_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    gpus: list[GPUSample] = Field(default_factory=list)


class FleetSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=utcnow)
    source: str
    nodes: list[NodeSample] = Field(default_factory=list)
    workloads: list[Workload] = Field(default_factory=list)

    def node(self, name: str) -> NodeSample | None:
        return next((n for n in self.nodes if n.node == name), None)

    def all_gpus(self) -> list[GPUSample]:
        return [g for n in self.nodes for g in n.gpus]

    def workload(self, job_id: str | None) -> Workload | None:
        if job_id is None:
            return None
        return next((w for w in self.workloads if w.job_id == job_id), None)
