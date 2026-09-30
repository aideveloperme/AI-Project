"""Telemetry sources: produce canonical :class:`FleetSnapshot` objects.

* ``SimulatorSource``  — in-process simulator (single-container demo, tests)
* ``PrometheusSource`` — production path: Prometheus HTTP API over DCGM
  exporter + node_exporter + Sentinel workload agent metrics.
"""
from __future__ import annotations

import abc
import asyncio
import logging
from collections import defaultdict
from datetime import timedelta

import httpx

from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.catalog import decode_throttle, hardware
from sentinel.telemetry.models import FleetSnapshot, GPUSample, NodeSample, Workload, utcnow
from sentinel.telemetry.vendors import NODE_INFO_QUERY, NODE_QUERIES, VENDORS, WORKLOAD_INFO_QUERY, DCGM_INVENTORY_QUERY

log = logging.getLogger(__name__)


class TelemetrySource(abc.ABC):
    name: str

    @abc.abstractmethod
    async def collect(self) -> FleetSnapshot: ...

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None


class SimulatorSource(TelemetrySource):
    name = "simulator"

    def __init__(self, sim: ClusterSimulator, dt: float = 5.0, warmup_s: float = 120.0, virtual_time: bool = False):
        """``virtual_time``: timestamps advance by exactly ``dt`` per collect
        (tests / accelerated demos) instead of following the wall clock."""
        self.sim = sim
        self.dt = dt
        self.virtual_time = virtual_time
        self._t = utcnow()
        if warmup_s:
            sim.warmup(warmup_s, dt)

    async def collect(self) -> FleetSnapshot:
        snap = self.sim.tick(self.dt)
        if self.virtual_time:
            self._t = self._t + timedelta(seconds=self.dt)
            snap.timestamp = self._t
        return snap


class PrometheusSource(TelemetrySource):
    name = "prometheus"

    def __init__(self, url: str, vendor: str = "nvidia", timeout: float = 10.0, client: httpx.AsyncClient | None = None,
                 node_label: str = "node"):
        self.url = url.rstrip("/")
        self.mapping = VENDORS[vendor]
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.node_label = node_label

    async def _query(self, promql: str) -> list[dict]:
        r = await self.client.get(f"{self.url}/api/v1/query", params={"query": promql})
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            raise RuntimeError(f"prometheus query failed: {body}")
        return body["data"]["result"]

    async def _safe(self, promql: str) -> list[dict]:
        try:
            return await self._query(promql)
        except Exception as e:  # one broken query must not break the whole snapshot
            log.warning("query failed %s: %s", promql, e)
            return []

    async def collect(self) -> FleetSnapshot:
        m = self.mapping
        gpu_names = list(m.gpu_queries)
        node_names = list(NODE_QUERIES)
        results = await asyncio.gather(
            self._safe(NODE_INFO_QUERY), self._safe(WORKLOAD_INFO_QUERY), self._safe(DCGM_INVENTORY_QUERY),
            *[self._safe(m.gpu_queries[k]) for k in gpu_names],
            *[self._safe(NODE_QUERIES[k]) for k in node_names],
        )
        node_info, wl_info, inventory = results[0], results[1], results[2]
        gpu_res = dict(zip(gpu_names, results[3:3 + len(gpu_names)]))
        node_res = dict(zip(node_names, results[3 + len(gpu_names):]))
        nl = self.node_label

        nodes: dict[str, NodeSample] = {}
        for s in node_info:
            lb = s["metric"]
            nodes[lb[nl]] = NodeSample(node=lb[nl], cluster=lb.get("cluster", "default"),
                                       server_type=lb.get("server_type", "unknown"), gpu_model=lb.get("gpu_model", "unknown"),
                                       rack=lb.get("rack") or None, driver_version=lb.get("driver_version") or None,
                                       cuda_version=lb.get("cuda_version") or None, workload_id=lb.get("job_id") or None)

        gpus: dict[tuple[str, int], GPUSample] = {}
        for s in inventory:
            lb = s["metric"]
            if nl not in lb:
                continue
            node = lb[nl]
            idx = int(lb.get(m.gpu_label, 0))
            model = lb.get(m.model_label, "unknown")
            gpus[(node, idx)] = GPUSample(node=node, index=idx, uuid=lb.get(m.uuid_label, f"{node}-{idx}"),
                                          model=model, vendor=m.vendor)
            if node not in nodes:
                nodes[node] = NodeSample(node=node, cluster=lb.get("cluster", "default"), server_type="unknown",
                                         gpu_model=model)

        for metric, series in gpu_res.items():
            for s in series:
                lb = s["metric"]
                key = (lb.get(nl), int(lb.get(m.gpu_label, -1)))
                if key in gpus:
                    gpus[key].metrics[metric] = float(s["value"][1])
        for g in gpus.values():
            for canonical, source in m.gpu_derived.items():
                if source in g.metrics:
                    g.metrics[canonical] = round(g.metrics.pop(source) * hardware(g.model)["hbm_peak_gbps"], 1)
            g.throttle_reasons = decode_throttle(int(g.metrics.get("throttle_mask", 0)))

        for metric, series in node_res.items():
            for s in series:
                node = s["metric"].get(nl)
                if node in nodes:
                    nodes[node].metrics[metric] = float(s["value"][1])

        by_node: dict[str, list[GPUSample]] = defaultdict(list)
        for g in gpus.values():
            by_node[g.node].append(g)
        for name, n in nodes.items():
            n.gpus = sorted(by_node.get(name, []), key=lambda g: g.index)

        workloads = []
        for s in wl_info:
            lb = s["metric"]
            workloads.append(Workload(job_id=lb["job_id"], name=lb.get("name", lb["job_id"]),
                                      scheduler=lb.get("scheduler", "unknown"), framework=lb.get("framework", "unknown"),
                                      kind=lb.get("kind", "training"), user=lb.get("user") or None,
                                      namespace=lb.get("namespace") or None,
                                      nodes=[x for x in lb.get("nodes", "").split(",") if x],
                                      throughput_unit=lb.get("throughput_unit", "units/s")))
        for g in gpus.values():
            wid = nodes[g.node].workload_id
            if wid:
                g.processes = [{"job_id": wid}]
        return FleetSnapshot(timestamp=utcnow(), source="prometheus", nodes=sorted(nodes.values(), key=lambda n: n.node),
                             workloads=workloads)

    async def close(self) -> None:
        await self.client.aclose()
