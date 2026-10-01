"""Anomalies, incidents, RCA, Ask Sentinel, alerts and demo (fault injection) endpoints."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from sentinel.api.deps import Principal, ctx, require
from sentinel.db import AnomalyRow, Incident, NotificationLog
from sentinel.incidents.service import ACTIVE_STATUSES, incident_to_dict
from sentinel.rca.engine import RULES
from sentinel.simulator.faults import FAULT_DESCRIPTIONS, GPU_SCOPED, Fault, FaultType
from sentinel.telemetry.sources import SimulatorSource

router = APIRouter(prefix="/api/v1", tags=["operations"])
READ = Depends(require("read"))


# ---------------------------------------------------------------- anomalies
@router.get("/anomalies")
def anomalies(active: bool = True, node: str | None = None, category: str | None = None,
              limit: int = Query(500, le=5000), p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        q = select(AnomalyRow).where(AnomalyRow.tenant_id == p.tenant)
        if active:
            q = q.where(AnomalyRow.active.is_(True))
        if node:
            q = q.where(AnomalyRow.node == node)
        if category:
            q = q.where(AnomalyRow.category == category)
        rows = db.scalars(q.order_by(AnomalyRow.last_seen.desc()).limit(limit)).all()
    return [{"id": r.id, "entity": r.entity, "node": r.node, "gpu_index": r.gpu_index, "level": r.level,
             "metric": r.metric, "category": r.category, "direction": r.direction, "severity": r.severity,
             "methods": r.methods, "value": r.value, "expected": r.expected, "deviation_pct": r.deviation_pct,
             "zscore": r.zscore, "description": r.description, "incident_id": r.incident_id, "active": r.active,
             "first_seen": r.first_seen.isoformat(), "last_seen": r.last_seen.isoformat()} for r in rows]


# ---------------------------------------------------------------- incidents
@router.get("/incidents")
def incidents(status: str | None = Query(None, description="comma-separated, or 'active'"), severity: str | None = None,
              node: str | None = None, limit: int = Query(200, le=2000), p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        q = select(Incident).where(Incident.tenant_id == p.tenant)
        if status == "active":
            q = q.where(Incident.status.in_(ACTIVE_STATUSES))
        elif status:
            q = q.where(Incident.status.in_([s.strip().upper() for s in status.split(",")]))
        if severity:
            q = q.where(Incident.severity == severity)
        if node:
            q = q.where(Incident.node == node)
        rows = db.scalars(q.order_by(Incident.created_at.desc()).limit(limit)).all()
        out = []
        for i in rows:
            d = incident_to_dict(i)
            for k in ("signals", "peer_comparison", "observed", "ai_explanation"):
                d.pop(k, None)
            d["top_hypothesis"] = i.hypotheses[0]["title"] if i.hypotheses else None
            out.append(d)
        return out


def _get_incident(db, p: Principal, incident_id: str) -> Incident:
    inc = db.get(Incident, incident_id)
    if inc is None or inc.tenant_id != p.tenant:
        raise HTTPException(404, "incident not found")
    return inc


@router.get("/incidents/{incident_id}")
def incident(incident_id: str, p: Principal = READ, c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        return incident_to_dict(_get_incident(db, p, incident_id), with_events=True)


class StatusChange(BaseModel):
    status: str
    note: str | None = Field(None, max_length=4000)


@router.post("/incidents/{incident_id}/status")
def set_status(incident_id: str, body: StatusChange, p: Principal = Depends(require("incident.update")),
               c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        inc = _get_incident(db, p, incident_id)
        try:
            c.engine.incidents.transition(db, inc, body.status, p.username, body.note)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        d = incident_to_dict(inc, with_events=True)
    if body.status.upper() in ("RESOLVED", "FALSE_POSITIVE"):
        c.engine.notify(d, "resolved")
    return d


class Comment(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@router.post("/incidents/{incident_id}/comments")
def comment(incident_id: str, body: Comment, p: Principal = Depends(require("incident.update")), c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        inc = _get_incident(db, p, incident_id)
        c.engine.incidents.comment(db, inc, p.username, body.text)
        return incident_to_dict(inc, with_events=True)


@router.post("/incidents/{incident_id}/explain")
def explain(incident_id: str, force: bool = False, p: Principal = READ, c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        _get_incident(db, p, incident_id)
    exp = c.engine.explain_incident(incident_id, force=force)
    return exp or {}


# --------------------------------------------------------------------- RCA
@router.get("/rca/rules")
def rca_rules(p: Principal = READ) -> list[dict]:
    return [{"id": r.id, "title": r.title, "category": r.category,
             "required": [x.description for x in r.required], "supporting": [x.description for x in r.supporting],
             "contradicting": [x.description for x in r.contradicting], "explanation": r.explanation,
             "actions": r.actions, "base_confidence": r.base, "max_confidence": r.ceiling} for r in RULES]


@router.get("/rca/nodes")
def rca_nodes(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    st = c.engine.state
    out = []
    for node, (ev, hyps, sev) in st.results.items():
        if not ev.signals:
            continue
        out.append({"node": node, "severity": sev, "perf_deviation_pct": ev.perf_deviation_pct,
                    "signal_count": len(ev.signals),
                    "signals_by_category": _count(s.category for s in ev.signals),
                    "throttle": ev.throttle, "hypotheses": [h.to_dict() for h in hyps]})
    out.sort(key=lambda x: (x["severity"] != "critical", x["perf_deviation_pct"] or 0))
    return out


def _count(it) -> dict:
    d: dict = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d


# ------------------------------------------------------------ Ask Sentinel
class Ask(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    use_llm: bool = True


@router.post("/ask")
def ask(body: Ask, p: Principal = READ, c=Depends(ctx)) -> dict:
    return c.assistant.ask(body.question, use_llm=body.use_llm)


@router.get("/ask/suggestions")
def suggestions(p: Principal = READ, c=Depends(ctx)) -> list[str]:
    st = c.engine.state
    bad = [n for n, s in st.node_status.items() if s != "healthy"]
    node = bad[0] if bad else (st.snapshot.nodes[0].node if st.snapshot else "gpu-01")
    job = st.snapshot.workloads[0].job_id if st.snapshot and st.snapshot.workloads else "78421"
    return [f"Why is {node} slow?", "Which GPUs are abnormal?", "Show me nodes with thermal issues.",
            f"Why did training job {job} slow down?", f"Compare {node} with healthy nodes.",
            "What changed during the last two hours?", "Which nodes have repeated ECC errors?",
            "Show me GPUs with abnormal clocks."]


# ------------------------------------------------------------------ alerts
@router.get("/alerts")
def alerts(unread_only: bool = False, limit: int = 50, p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        q = select(NotificationLog).where(NotificationLog.tenant_id == p.tenant, NotificationLog.channel_type == "dashboard")
        if unread_only:
            q = q.where(NotificationLog.read.is_(False))
        rows = db.scalars(q.order_by(NotificationLog.ts.desc()).limit(limit)).all()
    return [{"id": r.id, "ts": r.ts.isoformat(), "incident_id": r.incident_id, "event": r.event, "message": r.message,
             "read": r.read} for r in rows]


@router.post("/alerts/read")
def alerts_read(p: Principal = READ, c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        n = db.execute(update(NotificationLog).where(NotificationLog.tenant_id == p.tenant,
                                                     NotificationLog.read.is_(False)).values(read=True)).rowcount
        db.commit()
    return {"marked": n}


# -------------------------------------------------------------------- demo
class Inject(BaseModel):
    type: FaultType
    node: str
    gpu: int | None = None
    severity: float = Field(0.8, ge=0.05, le=1.0)
    duration_s: float | None = Field(900, ge=10, le=86400)


SCENARIOS = {
    "thermal-gpu04": [Inject(type=FaultType.THERMAL, node="gpu-04", gpu=3, severity=0.9)],
    "cpu-bottleneck": [Inject(type=FaultType.CPU_BOTTLENECK, node="gpu-06", severity=0.8)],
    "network-degradation": [Inject(type=FaultType.NETWORK_DEGRADATION, node="gpu-02", severity=0.8)],
    "ecc-failing-hbm": [Inject(type=FaultType.ECC_ERRORS, node="gpu-07", gpu=5, severity=0.9)],
    "software-regression": [Inject(type=FaultType.APP_REGRESSION, node="gpu-05", severity=0.8)],
    "mixed-incidents": [Inject(type=FaultType.THERMAL, node="gpu-04", gpu=3, severity=0.9),
                        Inject(type=FaultType.POWER_CAP, node="gpu-10", gpu=1, severity=0.8),
                        Inject(type=FaultType.COMMUNICATION_BOTTLENECK, node="gpu-02", severity=0.8)],
}


def _sim(c):
    src = c.engine.source
    return src.sim if isinstance(src, SimulatorSource) else None


def _remote(c, method: str, path: str, json: dict | None = None):
    if not c.settings.demo_mode:
        raise HTTPException(403, "demo mode disabled")
    try:
        r = httpx.request(method, f"{c.settings.simulator_url.rstrip('/')}{path}", json=json, timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"simulator unreachable: {e}") from e


@router.get("/demo/fault-types")
def fault_types(p: Principal = READ) -> list[dict]:
    return [{"type": t.value, "description": d, "gpu_scoped": t in GPU_SCOPED} for t, d in FAULT_DESCRIPTIONS.items()]


@router.get("/demo/scenarios")
def scenarios(p: Principal = READ) -> dict:
    return {k: [i.model_dump(mode="json") for i in v] for k, v in SCENARIOS.items()}


@router.get("/demo/faults")
def faults(p: Principal = READ, c=Depends(ctx)) -> list[dict]:
    sim = _sim(c)
    if sim is not None:
        return [f.to_dict() for f in sim.active_faults()]
    return _remote(c, "GET", "/api/faults")


def _inject(c, body: Inject) -> dict:
    sim = _sim(c)
    if sim is not None:
        if not c.settings.demo_mode:
            raise HTTPException(403, "demo mode disabled")
        try:
            return sim.inject(Fault(type=body.type, node=body.node, gpu=body.gpu, severity=body.severity,
                                    duration_s=body.duration_s)).to_dict()
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
    return _remote(c, "POST", "/api/faults", body.model_dump(mode="json"))


@router.post("/demo/faults")
def inject(body: Inject, p: Principal = Depends(require("simulator.inject")), c=Depends(ctx)) -> dict:
    return _inject(c, body)


@router.post("/demo/scenarios/{name}")
def run_scenario(name: str, p: Principal = Depends(require("simulator.inject")), c=Depends(ctx)) -> list[dict]:
    if name not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    return [_inject(c, i) for i in SCENARIOS[name]]


@router.delete("/demo/faults")
def clear_faults(fault_id: str | None = None, p: Principal = Depends(require("simulator.inject")), c=Depends(ctx)) -> dict:
    sim = _sim(c)
    if sim is not None:
        return {"cleared": sim.clear(fault_id=fault_id)}
    return _remote(c, "DELETE", f"/api/faults/{fault_id}" if fault_id else "/api/faults")
