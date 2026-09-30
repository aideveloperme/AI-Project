"""Fleet, node, GPU, peer, performance, network, workload and trend endpoints."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from sentinel.analytics.peers import compare, peer_groups
from sentinel.analytics.workloads import job_analysis
from sentinel.api.deps import Principal, ctx, require
from sentinel.db import AnomalyRow, FleetSampleRow, GPUSampleRow, Incident, NodeSampleRow
from sentinel.incidents.service import ACTIVE_STATUSES
from sentinel.telemetry.catalog import CATALOG, GPU_METRICS, NODE_METRICS
from sentinel.telemetry.models import utcnow

router = APIRouter(prefix="/api/v1", tags=["fleet"])
READ = Depends(require("read"))

SEV_RANK = {"critical": 2, "warning": 1, "healthy": 0}


def _state(c):
    st = c.engine.state
    if st.snapshot is None:
        raise HTTPException(503, "telemetry not yet available — first analysis cycle pending")
    return st


def _r(v, n=2):
    return None if v is None else round(float(v), n)


@router.get("/overview")
def overview(p: Principal = READ, c=Depends(ctx)) -> dict:
    st = _state(c)
    snap = st.snapshot
    gpus = snap.all_gpus()
    counts = Counter(st.gpu_status.values())
    with c.db.session() as db:
        active = db.scalars(select(Incident).where(Incident.tenant_id == p.tenant, Incident.status.in_(ACTIVE_STATUSES))
                            .order_by(Incident.created_at.desc())).all()
        trend = db.scalars(select(FleetSampleRow).where(FleetSampleRow.tenant_id == p.tenant)
                           .order_by(FleetSampleRow.ts.desc()).limit(120)).all()
    inc_by_node = {i.node: i for i in active}
    problems = []
    for node, (ev, hyps, sev) in st.results.items():
        if not ev.signals:
            continue
        problems.append({"node": node, "status": st.node_status[node], "signals": len(ev.signals),
                         "perf_deviation_pct": _r(ev.perf_deviation_pct, 1),
                         "top_hypothesis": hyps[0].title if hyps else None,
                         "confidence": hyps[0].confidence_label if hyps else None,
                         "incident_id": inc_by_node[node].id if node in inc_by_node else None})
    problems.sort(key=lambda x: (-SEV_RANK.get(x["status"], 0), x["perf_deviation_pct"] or 0))
    perf = [ev.perf_deviation_pct for ev, _, _ in st.results.values() if ev.perf is not None]
    return {
        "timestamp": snap.timestamp.isoformat(),
        "source": snap.source,
        "clusters": sorted({n.cluster for n in snap.nodes}),
        "total_nodes": len(snap.nodes),
        "total_gpus": len(gpus),
        "healthy_gpus": counts.get("healthy", 0),
        "warning_gpus": counts.get("warning", 0),
        "critical_gpus": counts.get("critical", 0),
        "healthy_nodes": sum(1 for s in st.node_status.values() if s == "healthy"),
        "avg_gpu_util": _r(np.mean([g.metrics.get("gpu_util", 0) for g in gpus]), 1),
        "avg_temp_c": _r(np.mean([g.metrics.get("temp_c", 0) for g in gpus]), 1),
        "total_power_kw": _r(sum(g.metrics.get("power_w", 0) for g in gpus) / 1000, 1),
        "performance_anomalies": sum(1 for d in perf if d is not None and d <= -8),
        "active_anomalies": len(st.signals),
        "active_incidents": len(active),
        "critical_incidents": sum(1 for i in active if i.severity == "critical"),
        "top_problem_nodes": problems[:8],
        "recent_incidents": [{"id": i.id, "title": i.title, "severity": i.severity, "status": i.status, "node": i.node,
                              "created_at": i.created_at.isoformat()} for i in active[:6]],
        "trend": [{"ts": t.ts.isoformat(), **t.metrics} for t in reversed(trend)],
        "engine": {"cycle": st.cycle, "last_cycle_ms": _r(st.last_cycle_ms, 1), "last_error": st.last_error},
    }


@router.get("/clusters")
def clusters(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = _state(c)
    out: dict[str, dict] = {}
    groups = peer_groups(st.snapshot, c.engine.peers.group_keys, c.engine.peers.min_group)
    group_of = {n: g for g, ns in groups.items() for n in ns}
    for n in st.snapshot.nodes:
        cl = out.setdefault(n.cluster, {"cluster": n.cluster, "racks": {}, "nodes": 0, "gpus": 0,
                                        "status": Counter(), "peer_groups": sorted(groups)})
        cl["nodes"] += 1
        cl["gpus"] += len(n.gpus)
        cl["status"][st.node_status.get(n.node, "healthy")] += 1
        cl["racks"].setdefault(n.rack or "unassigned", []).append({
            "node": n.node, "status": st.node_status.get(n.node, "healthy"), "workload": n.workload_id,
            "peer_group": group_of.get(n.node),
            "gpus": [{"index": g.index, "status": st.gpu_status.get(g.key, "healthy"),
                      "temp_c": _r(g.metrics.get("temp_c"), 1), "util": _r(g.metrics.get("gpu_util"), 0)} for g in n.gpus]})
    for cl in out.values():
        cl["status"] = dict(cl["status"])
        cl["racks"] = [{"rack": r, "nodes": ns} for r, ns in sorted(cl["racks"].items())]
    return list(out.values())


def _node_row(st, n) -> dict:
    ev, hyps, sev = st.results.get(n.node, (None, [], "info"))
    gs = Counter(st.gpu_status.get(g.key, "healthy") for g in n.gpus)
    avg = lambda k: _r(np.mean([g.metrics.get(k, 0) for g in n.gpus]), 1) if n.gpus else None  # noqa: E731
    return {"node": n.node, "cluster": n.cluster, "rack": n.rack, "server_type": n.server_type, "gpu_model": n.gpu_model,
            "gpu_count": len(n.gpus), "status": st.node_status.get(n.node, "healthy"), "gpu_status": dict(gs),
            "workload_id": n.workload_id, "avg_gpu_util": avg("gpu_util"), "avg_temp_c": avg("temp_c"),
            "max_temp_c": _r(max((g.metrics.get("temp_c", 0) for g in n.gpus), default=0), 1),
            "avg_sm_clock_mhz": avg("sm_clock_mhz"), "power_kw": _r(sum(g.metrics.get("power_w", 0) for g in n.gpus) / 1000, 2),
            "cpu_util": _r(n.metrics.get("cpu_util"), 1), "throughput": _r(n.metrics.get("throughput"), 1),
            "perf_deviation_pct": _r(ev.perf_deviation_pct, 1) if ev else None,
            "signals": len(ev.signals) if ev else 0, "top_hypothesis": hyps[0].title if hyps else None}


@router.get("/nodes")
def nodes(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = _state(c)
    return [_node_row(st, n) for n in st.snapshot.nodes]


def _pc_dict(pc):
    return None if pc is None else pc.to_dict()


@router.get("/nodes/{node}")
def node_detail(node: str, p: Principal = READ, c=Depends(ctx)) -> dict:
    st = _state(c)
    n = st.snapshot.node(node)
    if n is None:
        raise HTTPException(404, f"node {node} not found")
    ev, hyps, sev = st.results[node]
    # Peer comparison with historical baseline for this node only (cheap).
    peers = c.engine.peers.run(st.snapshot, c.engine.history, with_history_baseline=True, only_nodes={node})
    node_peer = [peers[node][m.name].to_dict() for m in NODE_METRICS if m.name in peers.get(node, {})]
    gpu_rows = []
    for g in n.gpus:
        gp = peers.get(g.key, {})
        gpu_rows.append({"index": g.index, "uuid": g.uuid, "model": g.model, "status": st.gpu_status.get(g.key, "healthy"),
                         "metrics": g.metrics, "throttle_reasons": g.throttle_reasons, "processes": g.processes,
                         "peer": {k: {"median": _r(v.peer.median), "deviation_pct": _r(v.deviation_pct, 1),
                                      "robust_z": _r(v.robust_z, 1)} for k, v in gp.items()}})
    wl = st.snapshot.workload(n.workload_id)
    with c.db.session() as db:
        incs = db.scalars(select(Incident).where(Incident.tenant_id == p.tenant, Incident.node == node)
                          .order_by(Incident.created_at.desc()).limit(25)).all()
    return {
        **_node_row(st, n),
        "driver_version": n.driver_version, "cuda_version": n.cuda_version,
        "metrics": n.metrics,
        "gpus": gpu_rows,
        "workload": wl.model_dump() if wl else None,
        "peer_group": next(iter(peers.get(node, {}).values())).group if peers.get(node) else None,
        "peer_comparison": node_peer,
        "gpu_peer_summary": _gpu_peer_summary(n, peers),
        "anomalies": [s.to_dict() for s in ev.signals],
        "hypotheses": [h.to_dict() for h in hyps],
        "severity": sev,
        "incidents": [{"id": i.id, "title": i.title, "severity": i.severity, "status": i.status,
                       "created_at": i.created_at.isoformat(), "resolved_at": i.resolved_at.isoformat() if i.resolved_at else None,
                       "category": i.category} for i in incs],
    }


def _gpu_peer_summary(n, peers) -> list[dict]:
    rows = []
    for spec in GPU_METRICS:
        if not spec.peer_compare:
            continue
        pcs = [peers.get(g.key, {}).get(spec.name) for g in n.gpus]
        pcs = [x for x in pcs if x is not None]
        if not pcs:
            continue
        worst = min(pcs, key=lambda x: x.deviation_pct) if spec.direction == "low_bad" else max(pcs, key=lambda x: x.deviation_pct)
        rows.append({"metric": spec.name, "label": spec.label, "unit": spec.unit, "category": spec.category,
                     "node_avg": _r(np.mean([x.value for x in pcs])), "peer_median": _r(pcs[0].peer.median),
                     "p10": _r(pcs[0].peer.p10), "p90": _r(pcs[0].peer.p90),
                     "worst_gpu": int(worst.entity.split("/gpu")[1]), "worst_value": _r(worst.value),
                     "worst_deviation_pct": _r(worst.deviation_pct, 1), "worst_z": _r(worst.robust_z, 1)})
    return rows


@router.get("/nodes/{node}/history")
def node_history(node: str, metrics: str = Query("throughput,cpu_util"), gpu: int | None = None,
                 minutes: int = Query(30, ge=1, le=60 * 24 * 30), p: Principal = READ, c=Depends(ctx)) -> dict:
    """Time series for a node or one of its GPUs. Recent windows come from the
    in-memory ring buffer; longer windows from the database (downsampled)."""
    entity = f"{node}/gpu{gpu}" if gpu is not None else node
    names = [m.strip() for m in metrics.split(",") if m.strip()]
    hist = c.engine.history
    span_s = len(hist.times(entity, names[0])) * c.settings.analysis_interval_s if names else 0
    out: dict[str, list] = {}
    if minutes * 60 <= span_s + 1 or minutes <= 120:
        since = (utcnow() - timedelta(minutes=minutes)).timestamp()
        for m in names:
            t, v = hist.times(entity, m), hist.values(entity, m)
            mask = t >= since
            step = max(1, int(mask.sum() / 400))
            out[m] = [[float(a) * 1000, round(float(b), 3)] for a, b in list(zip(t[mask], v[mask]))[::step]]
        return {"entity": entity, "source": "memory", "series": out}
    since = utcnow() - timedelta(minutes=minutes)
    with c.db.session() as db:
        if gpu is None:
            rows = db.scalars(select(NodeSampleRow).where(NodeSampleRow.node == node, NodeSampleRow.ts >= since)
                              .order_by(NodeSampleRow.ts)).all()
        else:
            rows = db.scalars(select(GPUSampleRow).where(GPUSampleRow.node == node, GPUSampleRow.gpu_index == gpu,
                                                         GPUSampleRow.ts >= since).order_by(GPUSampleRow.ts)).all()
    step = max(1, len(rows) // 400)
    for m in names:
        out[m] = [[r.ts.timestamp() * 1000, r.metrics.get(m)] for r in rows[::step] if m in r.metrics]
    return {"entity": entity, "source": "database", "series": out}


@router.get("/gpus")
def gpus(status: str | None = None, node: str | None = None, p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = _state(c)
    out = []
    for n in st.snapshot.nodes:
        if node and n.node != node:
            continue
        for g in n.gpus:
            s = st.gpu_status.get(g.key, "healthy")
            if status and s != status:
                continue
            pp = st.peers.get(g.key, {}).get("perf_index")
            m = g.metrics
            out.append({"key": g.key, "node": n.node, "index": g.index, "model": g.model, "status": s,
                        "gpu_util": _r(m.get("gpu_util"), 1), "sm_clock_mhz": _r(m.get("sm_clock_mhz"), 0),
                        "temp_c": _r(m.get("temp_c"), 1), "power_w": _r(m.get("power_w"), 0),
                        "hbm_bw_gbps": _r(m.get("hbm_bw_gbps"), 0), "mem_used_pct": _r(m.get("mem_used_pct"), 1),
                        "ecc_sbe_rate": _r(m.get("ecc_sbe_rate"), 2), "nvlink_bw_gbps": _r(m.get("nvlink_bw_gbps"), 0),
                        "perf_index": _r(m.get("perf_index"), 1),
                        "perf_deviation_pct": _r(pp.deviation_pct, 1) if pp else None,
                        "throttle_reasons": g.throttle_reasons, "workload_id": n.workload_id})
    return out


@router.get("/gpus/{node}/{index}")
def gpu_detail(node: str, index: int, p: Principal = READ, c=Depends(ctx)) -> dict:
    st = _state(c)
    n = st.snapshot.node(node)
    g = next((x for x in (n.gpus if n else []) if x.index == index), None)
    if g is None:
        raise HTTPException(404, "gpu not found")
    peers = c.engine.peers.run(st.snapshot, c.engine.history, with_history_baseline=True, only_nodes={node})
    ev, hyps, _ = st.results[node]
    return {"key": g.key, "node": node, "index": index, "uuid": g.uuid, "model": g.model, "vendor": g.vendor,
            "status": st.gpu_status.get(g.key, "healthy"), "metrics": g.metrics, "throttle_reasons": g.throttle_reasons,
            "processes": g.processes, "workload_id": n.workload_id,
            "peer_comparison": [v.to_dict() for v in peers.get(g.key, {}).values()],
            "anomalies": [s.to_dict() for s in ev.signals if s.gpu_index == index],
            "hypotheses": [h.to_dict() for h in hyps if index in h.affected_gpus]}


@router.get("/peers")
def peer_distribution(metric: str = "perf_index", level: str = "gpu", p: Principal = READ, c=Depends(ctx)) -> dict:
    """Peer benchmarking: every entity's value vs. its peer-group median."""
    st = _state(c)
    spec = CATALOG.get((level, metric))
    if spec is None:
        raise HTTPException(404, f"unknown metric {level}/{metric}")
    rows = []
    for ent, ms in st.peers.items():
        pc = ms.get(metric)
        if pc is None or pc.level != level:
            continue
        rows.append({"entity": ent, "value": _r(pc.value), "peer_median": _r(pc.peer.median),
                     "deviation_pct": _r(pc.deviation_pct, 1), "robust_z": _r(pc.robust_z, 1),
                     "percentile": _r(pc.percentile_rank, 0), "group": pc.group,
                     "status": st.gpu_status.get(ent) if level == "gpu" else st.node_status.get(ent)})
    rows.sort(key=lambda r: r["deviation_pct"])
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["value"])
    return {"metric": metric, "level": level, "label": spec.label, "unit": spec.unit, "direction": spec.direction,
            "groups": {g: {"n": len(v), "median": _r(np.median(v)), "p10": _r(np.percentile(v, 10)),
                           "p90": _r(np.percentile(v, 90)), "min": _r(min(v)), "max": _r(max(v))} for g, v in groups.items()},
            "rows": rows}


@router.get("/metrics/catalog")
def metric_catalog(p: Principal = READ) -> list[dict]:
    return [{"name": m.name, "level": m.level, "unit": m.unit, "label": m.label, "category": m.category,
             "direction": m.direction, "warn": m.warn, "crit": m.crit, "peer_min_rel_dev": m.peer_min_rel_dev}
            for m in GPU_METRICS + NODE_METRICS]


@router.get("/performance")
def performance(p: Principal = READ, c=Depends(ctx)) -> dict:
    st = _state(c)
    nodes_out = []
    for n in st.snapshot.nodes:
        pc = st.peers.get(n.node, {}).get("throughput")
        wl = st.snapshot.workload(n.workload_id)
        ev, hyps, _ = st.results[n.node]
        nodes_out.append({"node": n.node, "workload": wl.name if wl else None, "unit": wl.throughput_unit if wl else "",
                          "throughput": _r(n.metrics.get("throughput"), 1),
                          "peer_median": _r(pc.peer.median, 1) if pc else None,
                          "deviation_pct": _r(pc.deviation_pct, 1) if pc else None,
                          "step_time_ms": _r(n.metrics.get("step_time_ms"), 1), "group": pc.group if pc else None,
                          "status": st.node_status.get(n.node), "top_hypothesis": hyps[0].title if hyps else None})
    nodes_out.sort(key=lambda r: r["deviation_pct"] if r["deviation_pct"] is not None else 0)
    return {"nodes": nodes_out, "gpu_perf": peer_distribution("perf_index", "gpu", p, c)}


@router.get("/network")
def network(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = _state(c)
    keys = ["net_rx_gbps", "net_tx_gbps", "net_err_rate", "net_drop_rate", "ib_rx_gbps", "ib_tx_gbps",
            "ib_symbol_err_rate", "net_latency_us", "nccl_busbw_gbps", "nccl_comm_ratio"]
    out = []
    for n in st.snapshot.nodes:
        ev, hyps, _ = st.results[n.node]
        flagged = {s.metric for s in ev.signals if s.gpu_index is None}
        nv = [g.metrics.get("nvlink_bw_gbps", 0) for g in n.gpus]
        out.append({"node": n.node, "workload_id": n.workload_id, "status": st.node_status.get(n.node),
                    **{k: _r(n.metrics.get(k), 2) for k in keys},
                    "nvlink_avg_gbps": _r(np.mean(nv), 1) if nv else None,
                    "nvlink_crc_rate": _r(sum(g.metrics.get("nvlink_crc_rate", 0) for g in n.gpus), 2),
                    "flagged": sorted(flagged & set(keys)),
                    "peer": {k: {"median": _r(v.peer.median, 2), "deviation_pct": _r(v.deviation_pct, 1)}
                             for k, v in st.peers.get(n.node, {}).items() if k in keys}})
    return out


@router.get("/workloads")
def workloads(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = _state(c)
    return [job_analysis(c.engine, w.job_id) for w in st.snapshot.workloads]


@router.get("/workloads/{job_id}")
def workload(job_id: str, p: Principal = READ, c=Depends(ctx)) -> dict:
    st = _state(c)
    if st.snapshot.workload(job_id) is None:
        raise HTTPException(404, "workload not found")
    return job_analysis(c.engine, job_id)


@router.get("/trends")
def trends(minutes: int = Query(180, ge=5, le=60 * 24 * 30), node: str | None = None,
           metrics: str = "gpu_util,temp_c,power_kw,throughput", p: Principal = READ, c=Depends(ctx)) -> dict:
    since = utcnow() - timedelta(minutes=minutes)
    names = [m.strip() for m in metrics.split(",") if m.strip()]
    with c.db.session() as db:
        if node:
            rows = db.scalars(select(NodeSampleRow).where(NodeSampleRow.tenant_id == p.tenant, NodeSampleRow.node == node,
                                                          NodeSampleRow.ts >= since).order_by(NodeSampleRow.ts)).all()
        else:
            rows = db.scalars(select(FleetSampleRow).where(FleetSampleRow.tenant_id == p.tenant, FleetSampleRow.ts >= since)
                              .order_by(FleetSampleRow.ts)).all()
        incs = db.scalars(select(Incident).where(Incident.tenant_id == p.tenant, Incident.created_at >= since)
                          .order_by(Incident.created_at)).all()
        anoms = db.scalars(select(AnomalyRow).where(AnomalyRow.tenant_id == p.tenant, AnomalyRow.first_seen >= since)).all()
    step = max(1, len(rows) // 500)
    series = {m: [[r.ts.timestamp() * 1000, r.metrics.get(m)] for r in rows[::step] if m in r.metrics] for m in names}
    by_cat = Counter(a.category for a in anoms)
    return {"minutes": minutes, "node": node, "series": series,
            "incidents": [{"id": i.id, "ts": i.created_at.timestamp() * 1000, "title": i.title, "severity": i.severity,
                           "node": i.node, "category": i.category} for i in incs],
            "anomalies_by_category": dict(by_cat), "incident_count": len(incs)}
