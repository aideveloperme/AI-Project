"""Workload / job-level analytics (Slurm jobs, Kubernetes deployments)."""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from sentinel.db import Incident
from sentinel.incidents.service import ACTIVE_STATUSES


def job_analysis(engine, job_id: str) -> dict:
    st = engine.state
    snap = st.snapshot
    wl = snap.workload(job_id)
    if wl is None:
        return {"job_id": job_id, "found": False}
    nodes = [snap.node(n) for n in wl.nodes]
    nodes = [n for n in nodes if n is not None]
    tp = {n.node: n.metrics.get("throughput", 0.0) for n in nodes}
    total = float(sum(tp.values()))
    med = float(np.median(list(tp.values()))) if tp else 0.0

    # Aggregate baseline from history (median of the summed series, excluding the recent window).
    series = [engine.history.values(n.node, "throughput") for n in nodes]
    baseline = None
    if series and min(s.size for s in series) > 30:
        k = min(s.size for s in series)
        summed = np.sum([s[-k:] for s in series], axis=0)
        baseline = float(np.median(summed[:-6]))
    change = (total - baseline) / baseline * 100 if baseline else None

    stragglers = []
    for n in nodes:
        dev = (tp[n.node] - med) / med * 100 if med else 0.0
        if dev <= -8:
            res = st.results.get(n.node)
            top = res[1][0] if res and res[1] else None
            stragglers.append({"node": n.node, "throughput": round(tp[n.node], 1), "deviation_pct": round(dev, 1),
                               "top_hypothesis": top.title if top else None,
                               "confidence": top.confidence_label if top else None})
    stragglers.sort(key=lambda s: s["deviation_pct"])
    with engine.db.session() as db:
        incs = db.scalars(select(Incident).where(Incident.node.in_(wl.nodes), Incident.status.in_(ACTIVE_STATUSES))).all()
        inc_ids = [i.id for i in incs]
    per_node = [{"node": n.node, "throughput": round(tp[n.node], 1),
                 "deviation_pct": round((tp[n.node] - med) / med * 100, 1) if med else 0.0,
                 "status": st.node_status.get(n.node, "healthy"),
                 "gpu_util": round(float(np.mean([g.metrics.get("gpu_util", 0) for g in n.gpus])), 1) if n.gpus else None,
                 "nccl_busbw_gbps": n.metrics.get("nccl_busbw_gbps"), "step_time_ms": n.metrics.get("step_time_ms")}
                for n in nodes]
    # Synchronous training: effective job speed is gated by the slowest rank.
    effective = min(tp.values()) * len(tp) if wl.kind == "training" and tp else total
    return {"job_id": wl.job_id, "found": True, "name": wl.name, "scheduler": wl.scheduler, "kind": wl.kind,
            "framework": wl.framework, "user": wl.user, "namespace": wl.namespace, "nodes": wl.nodes,
            "gpus": sum(len(n.gpus) for n in nodes), "unit": wl.throughput_unit,
            "throughput": round(total, 1), "effective_throughput": round(effective, 1),
            "baseline_throughput": round(baseline, 1) if baseline else None,
            "change_pct": round(change, 1) if change is not None else None,
            "median_node_throughput": round(med, 1), "stragglers": stragglers, "incidents": inc_ids,
            "per_node": per_node}
