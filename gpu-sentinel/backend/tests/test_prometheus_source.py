"""PrometheusSource parses DCGM/node_exporter PromQL results into canonical telemetry."""
import asyncio

import httpx

from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.catalog import encode_throttle, hardware
from sentinel.telemetry.sources import PrometheusSource
from sentinel.telemetry.vendors import NODE_QUERIES, NVIDIA_DCGM


def fake_prometheus(snap):
    """Answers each mapped PromQL query with the simulator's values (inverse-mapped)."""
    gq = {v: k for k, v in NVIDIA_DCGM.gpu_queries.items()}
    nq = {v: k for k, v in NODE_QUERIES.items()}

    def vec(items):
        return {"status": "success", "data": {"resultType": "vector",
                                              "result": [{"metric": m, "value": [0, str(v)]} for m, v in items]}}

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params["query"]
        if q == "sentinel_node_info":
            return httpx.Response(200, json=vec([({"node": n.node, "cluster": n.cluster, "server_type": n.server_type,
                                                   "gpu_model": n.gpu_model, "rack": n.rack, "job_id": n.workload_id}, 1)
                                                 for n in snap.nodes]))
        if q == "sentinel_workload_info":
            return httpx.Response(200, json=vec([({"job_id": w.job_id, "name": w.name, "scheduler": w.scheduler,
                                                   "kind": w.kind, "nodes": ",".join(w.nodes)}, 1) for w in snap.workloads]))
        gl = lambda g: {"node": g.node, "gpu": str(g.index), "UUID": g.uuid, "modelName": g.model}  # noqa: E731
        if q == "DCGM_FI_DEV_GPU_TEMP":
            return httpx.Response(200, json=vec([(gl(g), g.metrics["temp_c"]) for g in snap.all_gpus()]))
        if q in gq:
            k = gq[q]
            if k == "dram_active":
                vals = [(gl(g), g.metrics["hbm_bw_gbps"] / hardware(g.model)["hbm_peak_gbps"]) for g in snap.all_gpus()]
            elif k == "throttle_mask":
                vals = [(gl(g), encode_throttle(g.throttle_reasons)) for g in snap.all_gpus()]
            else:
                vals = [(gl(g), g.metrics[k]) for g in snap.all_gpus()]
            return httpx.Response(200, json=vec(vals))
        if q in nq:
            return httpx.Response(200, json=vec([({"node": n.node}, n.metrics[nq[q]]) for n in snap.nodes]))
        return httpx.Response(400, json={"status": "error", "error": "unknown query"})

    return handler


def test_prometheus_source_builds_canonical_snapshot():
    from sentinel.simulator.faults import Fault, FaultType
    sim = ClusterSimulator(seed=9)
    sim.inject(Fault(FaultType.THERMAL, "gpu-02", gpu=1, severity=1.0))
    snap = sim.warmup(200, 10)
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_prometheus(snap)))
    src = PrometheusSource("http://prom:9090", client=client)
    out = asyncio.run(src.collect())
    assert out.source == "prometheus"
    assert len(out.nodes) == 12 and len(out.all_gpus()) == 96
    g = out.node("gpu-02").gpus[1]
    ref = snap.node("gpu-02").gpus[1]
    assert abs(g.metrics["hbm_bw_gbps"] - ref.metrics["hbm_bw_gbps"]) < 1.0   # derived from DRAM_ACTIVE ratio
    assert g.metrics["temp_c"] == ref.metrics["temp_c"]
    assert set(g.throttle_reasons) == set(ref.throttle_reasons) and "sw_thermal_slowdown" in g.throttle_reasons
    assert out.node("gpu-02").metrics["throughput"] == snap.node("gpu-02").metrics["throughput"]
    assert out.node("gpu-02").workload_id == "78421"
    assert {w.scheduler for w in out.workloads} == {"slurm", "kubernetes"}


def test_prometheus_source_tolerates_failing_queries():
    def handler(request):
        if "DCGM_FI_DEV_SM_CLOCK" in request.url.params["query"]:
            return httpx.Response(500)
        return httpx.Response(200, json={"status": "success", "data": {"resultType": "vector", "result": []}})
    src = PrometheusSource("http://prom:9090", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = asyncio.run(src.collect())
    assert out.nodes == []
