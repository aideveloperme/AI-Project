"""Incident lifecycle: correlation of node signals into incidents, de-duplication,
escalation, recurrence tracking, auto-resolution and operator workflow."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinel.analytics.detectors import SEV_ORDER
from sentinel.analytics.peers import PeerComparison
from sentinel.db import Incident, IncidentEvent
from sentinel.rca.engine import Hypothesis, NodeEvidence
from sentinel.telemetry.models import FleetSnapshot, utcnow

ACTIVE_STATUSES = ("OPEN", "ACKNOWLEDGED", "INVESTIGATING")
ALL_STATUSES = ("OPEN", "ACKNOWLEDGED", "INVESTIGATING", "RESOLVED", "FALSE_POSITIVE")
TRANSITIONS = {
    "OPEN": {"ACKNOWLEDGED", "INVESTIGATING", "RESOLVED", "FALSE_POSITIVE"},
    "ACKNOWLEDGED": {"INVESTIGATING", "RESOLVED", "FALSE_POSITIVE", "OPEN"},
    "INVESTIGATING": {"RESOLVED", "FALSE_POSITIVE", "ACKNOWLEDGED"},
    "RESOLVED": {"OPEN"},
    "FALSE_POSITIVE": {"OPEN"},
}

KEY_GPU_METRICS = ["gpu_util", "sm_clock_mhz", "temp_c", "power_w", "hbm_bw_gbps", "mem_util", "nvlink_bw_gbps", "perf_index"]
KEY_NODE_METRICS = ["throughput", "cpu_util", "dataloader_wait_pct", "nccl_busbw_gbps", "nccl_comm_ratio", "ib_rx_gbps", "net_latency_us"]


def incident_to_dict(i: Incident, with_events: bool = False) -> dict:
    d = {c.name: getattr(i, c.name) for c in Incident.__table__.columns}
    for k in ("created_at", "updated_at", "last_signal_at", "acknowledged_at", "resolved_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    if with_events:
        d["events"] = [{"ts": e.ts.isoformat(), "actor": e.actor, "type": e.type, "message": e.message, "data": e.data}
                       for e in i.events]
    return d


def _pc_row(pc: PeerComparison, scope: str, abnormal: bool) -> dict:
    return {"metric": pc.metric, "label": pc.label, "unit": pc.unit, "scope": scope, "value": round(pc.value, 2),
            "peer_median": round(pc.peer.median, 2), "peer_mean": round(pc.peer.mean, 2), "p10": round(pc.peer.p10, 2),
            "p90": round(pc.peer.p90, 2), "n": pc.peer.n, "deviation_pct": round(pc.deviation_pct, 2),
            "robust_z": round(pc.robust_z, 2), "status": "abnormal" if abnormal else "normal"}


def build_observed(ev: NodeEvidence, snap: FleetSnapshot, peers: dict[str, dict[str, PeerComparison]]) -> tuple[dict, list[dict], list[int]]:
    ns = snap.node(ev.node)
    wl = snap.workload(ns.workload_id) if ns else None
    flagged = {(s.entity, s.metric) for s in ev.signals}
    gpu_hits: dict[int, int] = {}
    for s in ev.signals:
        if s.gpu_index is not None:
            gpu_hits[s.gpu_index] = gpu_hits.get(s.gpu_index, 0) + 1
    affected = sorted(gpu_hits)
    worst = max(gpu_hits, key=lambda g: gpu_hits[g]) if gpu_hits else 0
    worst_key = f"{ev.node}/gpu{worst}"

    key_metrics, rows = [], []
    for m in KEY_GPU_METRICS:
        pc = peers.get(worst_key, {}).get(m)
        if pc:
            ab = (worst_key, m) in flagged
            key_metrics.append({"metric": m, "label": pc.label, "unit": pc.unit, "scope": f"GPU {worst}",
                                "value": round(pc.value, 2), "peer_median": round(pc.peer.median, 2),
                                "deviation_pct": round(pc.deviation_pct, 2), "status": "abnormal" if ab else "normal"})
            rows.append(_pc_row(pc, f"GPU {worst}", ab))
    for m in KEY_NODE_METRICS:
        pc = peers.get(ev.node, {}).get(m)
        if pc:
            ab = (ev.node, m) in flagged
            key_metrics.append({"metric": m, "label": pc.label, "unit": pc.unit, "scope": "node", "value": round(pc.value, 2),
                                "peer_median": round(pc.peer.median, 2), "deviation_pct": round(pc.deviation_pct, 2),
                                "status": "abnormal" if ab else "normal"})
            rows.append(_pc_row(pc, "node", ab))
    # Additional flagged metrics not already in the table
    seen = {(r["scope"], r["metric"]) for r in rows}
    for s in ev.signals:
        scope = f"GPU {s.gpu_index}" if s.gpu_index is not None else "node"
        pc = peers.get(s.entity, {}).get(s.metric)
        if pc and (scope, s.metric) not in seen and len(rows) < 30:
            rows.append(_pc_row(pc, scope, True))
            seen.add((scope, s.metric))

    thr = ev.perf
    observed = {
        "node": ev.node,
        "gpu_model": ns.gpu_model if ns else None,
        "server_type": ns.server_type if ns else None,
        "cluster": ns.cluster if ns else None,
        "workload": {"job_id": wl.job_id, "name": wl.name, "scheduler": wl.scheduler, "kind": wl.kind} if wl else None,
        "throughput": ({"value": round(thr.value, 1), "peer_median": round(thr.peer.median, 1),
                        "deviation_pct": round(thr.deviation_pct, 1), "n_peers": thr.peer.n,
                        "unit": wl.throughput_unit if wl else "units/s"} if thr else None),
        "throttle_reasons": {r: g for r, g in ev.throttle.items() if r not in ("gpu_idle",)},
        "key_metrics": key_metrics,
        "focus_gpu": worst if gpu_hits else None,
    }
    return observed, rows, affected


class IncidentService:
    def __init__(self, tenant: str, min_cycles: int = 2, resolve_cycles: int = 6):
        self.tenant = tenant
        self.min_cycles = min_cycles
        self.resolve_cycles = resolve_cycles
        self.pending: dict[str, int] = {}
        self.quiet: dict[str, int] = {}

    # ----------------------------------------------------------- helpers
    def _next_id(self, db: Session) -> str:
        n = db.scalar(select(func.count()).select_from(Incident)) or 0
        while True:
            n += 1
            iid = f"INC-{n:06d}"
            if db.get(Incident, iid) is None:
                return iid

    def active_for_node(self, db: Session, node: str) -> Incident | None:
        return db.scalars(select(Incident).where(Incident.tenant_id == self.tenant, Incident.node == node,
                                                 Incident.status.in_(ACTIVE_STATUSES))
                          .order_by(Incident.created_at.desc())).first()

    @staticmethod
    def _event(db: Session, inc: Incident, actor: str, type_: str, msg: str, data: dict | None = None) -> None:
        db.add(IncidentEvent(incident_id=inc.id, actor=actor, type=type_, message=msg, data=data))

    @staticmethod
    def is_incident_worthy(ev: NodeEvidence, hyps: list[Hypothesis]) -> bool:
        if not ev.signals:
            return False
        if any(s.method == "static_threshold" for s in ev.signals):
            return True
        if hyps:
            return True
        d = ev.perf_deviation_pct
        return d is not None and d <= -8

    # --------------------------------------------------------- main step
    def process(self, db: Session, snap: FleetSnapshot, peers: dict[str, dict[str, PeerComparison]],
                results: dict[str, tuple[NodeEvidence, list[Hypothesis], str]]) -> list[tuple[str, Incident]]:
        """Returns list of (event, incident) for notification: opened|updated|escalated|resolved."""
        changes: list[tuple[str, Incident]] = []
        now = utcnow()
        for node, (ev, hyps, severity) in results.items():
            worthy = self.is_incident_worthy(ev, hyps)
            self.pending[node] = self.pending.get(node, 0) + 1 if worthy else 0
            inc = self.active_for_node(db, node)
            if worthy:
                self.quiet[node] = 0
            if worthy and (inc is not None or self.pending[node] >= self.min_cycles):
                observed, rows, affected = build_observed(ev, snap, peers)
                top = hyps[0] if hyps else None
                title = f"{top.title} on {node}" if top else f"Performance anomaly on {node}"
                actions = top.recommended_actions if top else []
                summary = self._summary(node, observed, top)
                sig_dicts = [s.to_dict() for s in sorted(
                    ev.signals, key=lambda s: (-SEV_ORDER[s.severity], -abs(s.zscore or 0), -abs(s.deviation_pct or 0)))][:40]
                if inc is None and self._suppressed(db, node, top.category if top else "performance", now):
                    continue
                if inc is None:
                    inc = self._open(db, node, ev, severity, title, summary, observed, rows, affected, hyps,
                                     actions, sig_dicts, top, snap)
                    changes.append(("opened", inc))
                else:
                    ev_type = self._update(db, inc, severity, title, summary, observed, rows, affected, hyps, actions,
                                           sig_dicts, top, now)
                    if ev_type:
                        changes.append((ev_type, inc))
            elif inc is not None:
                self.quiet[node] = self.quiet.get(node, 0) + 1
                if self.quiet[node] >= self.resolve_cycles:
                    inc.status = "RESOLVED"
                    inc.resolved_at = now
                    inc.updated_at = now
                    inc.resolution = inc.resolution or "Auto-resolved: telemetry returned to peer baseline."
                    self._event(db, inc, "system", "auto_resolved",
                                f"No abnormal signals for {self.quiet[node]} consecutive analysis cycles.")
                    changes.append(("resolved", inc))
        db.commit()
        return changes

    def _suppressed(self, db: Session, node: str, category: str, now) -> bool:
        """Operator marked a same-category incident on this node FALSE_POSITIVE within the hour."""
        return db.scalars(select(Incident.id).where(
            Incident.tenant_id == self.tenant, Incident.node == node, Incident.status == "FALSE_POSITIVE",
            Incident.category == category, Incident.resolved_at >= now - timedelta(hours=1))).first() is not None

    @staticmethod
    def _summary(node: str, observed: dict, top: Hypothesis | None) -> str:
        thr = observed.get("throughput")
        s = (f"{node} is performing {abs(thr['deviation_pct']):.1f}% below its peer baseline."
             if thr and thr["deviation_pct"] <= -3 else f"{node} shows abnormal telemetry versus peers.")
        if top:
            s += f" Possible contributing factor: {top.title.lower()} (confidence {top.confidence_label})."
        return s

    def _open(self, db, node, ev, severity, title, summary, observed, rows, affected, hyps, actions, sig_dicts, top, snap):
        since = utcnow() - timedelta(days=30)
        prior = db.scalars(select(Incident).where(Incident.tenant_id == self.tenant, Incident.node == node,
                                                  Incident.created_at >= since)
                           .order_by(Incident.created_at.desc())).all()
        same_cat = [p for p in prior if top and p.category == top.category]
        ns = snap.node(node)
        inc = Incident(
            id=self._next_id(db), tenant_id=self.tenant, node=node, gpus=affected,
            workload_id=ns.workload_id if ns else None, severity=severity, status="OPEN",
            category=top.category if top else "performance", title=title, summary=summary,
            perf_deviation_pct=ev.perf_deviation_pct, confidence=top.confidence if top else 0.2,
            confidence_label=top.confidence_label if top else "low", observed=observed, signals=sig_dicts,
            peer_comparison=rows, hypotheses=[h.to_dict() for h in hyps], recommended_actions=actions,
            recurrence_count=len(same_cat), related_incidents=[p.id for p in prior[:10]],
        )
        db.add(inc)
        db.flush()
        self._event(db, inc, "system", "opened", summary,
                    {"signals": len(ev.signals), "top_hypothesis": top.rule_id if top else None})
        if same_cat:
            self._event(db, inc, "system", "recurrence",
                        f"Recurring issue: {len(same_cat)} prior '{top.category}' incident(s) on {node} in 30 days "
                        f"({', '.join(p.id for p in same_cat[:5])}).")
        return inc

    def _update(self, db, inc, severity, title, summary, observed, rows, affected, hyps, actions, sig_dicts, top, now):
        ev_type = None
        if SEV_ORDER[severity] > SEV_ORDER[inc.severity]:
            self._event(db, inc, "system", "escalated", f"Severity escalated {inc.severity} → {severity}.")
            inc.severity = severity
            ev_type = "escalated"
        old_top = inc.hypotheses[0]["rule_id"] if inc.hypotheses else None
        if top and top.rule_id != old_top:
            self._event(db, inc, "system", "diagnosis_changed",
                        f"Top hypothesis changed: {old_top or 'none'} → {top.rule_id} ({top.confidence_label}).")
            inc.title, inc.category = title, top.category
            inc.recommended_actions = actions
            inc.ai_explanation = None  # regenerate
            ev_type = ev_type or "updated"
        inc.summary, inc.observed, inc.peer_comparison = summary, observed, rows
        inc.gpus = sorted(set(inc.gpus or []) | set(affected))
        inc.hypotheses = [h.to_dict() for h in hyps] or inc.hypotheses
        inc.signals = sig_dicts
        if top:
            inc.confidence, inc.confidence_label = top.confidence, top.confidence_label
        new_perf = observed["throughput"]["deviation_pct"] if observed.get("throughput") else inc.perf_deviation_pct
        # Explanations quote numbers: regenerate when the evidence changed materially
        # (escalation, new diagnosis, or throughput moved ≥3 points since it was written).
        exp = inc.ai_explanation or {}
        if exp and (ev_type is not None or (new_perf is not None and exp.get("perf_deviation_pct") is not None
                                            and abs(new_perf - exp["perf_deviation_pct"]) >= 3)):
            inc.ai_explanation = None
        inc.perf_deviation_pct = new_perf
        inc.last_signal_at = now
        inc.updated_at = now
        return ev_type

    # ---------------------------------------------------- operator workflow
    def transition(self, db: Session, inc: Incident, status: str, actor: str, note: str | None = None) -> Incident:
        status = status.upper()
        if status not in ALL_STATUSES:
            raise ValueError(f"invalid status {status}")
        if status != inc.status and status not in TRANSITIONS[inc.status]:
            raise ValueError(f"cannot transition {inc.status} → {status}")
        now = utcnow()
        prev = inc.status
        inc.status = status
        inc.updated_at = now
        if status == "ACKNOWLEDGED":
            inc.acknowledged_by, inc.acknowledged_at = actor, now
        if status in ("RESOLVED", "FALSE_POSITIVE"):
            inc.resolved_at = now
            inc.resolution = note or inc.resolution or ("Marked false positive" if status == "FALSE_POSITIVE" else "Resolved by operator")
        if status == "OPEN":
            inc.resolved_at = None
        self._event(db, inc, actor, "status", f"{prev} → {status}" + (f": {note}" if note else ""))
        db.commit()
        return inc

    def comment(self, db: Session, inc: Incident, actor: str, text: str) -> None:
        self._event(db, inc, actor, "comment", text)
        inc.updated_at = utcnow()
        db.commit()
