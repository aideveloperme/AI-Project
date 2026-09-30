"""Analysis engine — the heart of GPU Sentinel.

    collect → history → peer benchmark → detectors → RCA correlation
            → incidents → AI explanation → notifications → persistence

One cycle runs every ``analysis_interval_s``. All DB work happens in a worker
thread so the API event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import delete, insert, select, update

from sentinel.ai.explainer import Explainer
from sentinel.analytics.detectors import AnomalySignal, DetectionContext, default_detectors, run_detectors
from sentinel.analytics.history import MetricHistory
from sentinel.analytics.peers import PeerBenchmark, PeerComparison
from sentinel.auth.security import SecretBox
from sentinel.config import Settings
from sentinel.db import (
    GPU,
    AnomalyRow,
    AuditLog,
    Database,
    FleetSampleRow,
    GPUSampleRow,
    Incident,
    Node,
    NodeSampleRow,
    NotificationChannel,
    NotificationLog,
)
from sentinel.incidents.service import ACTIVE_STATUSES, IncidentService, incident_to_dict
from sentinel.notifications.channels import EXTERNAL_SAAS, NOTIFIERS, SEVERITY_RANK, incident_payload
from sentinel.rca.engine import Hypothesis, NodeEvidence, RCAEngine, node_severity
from sentinel.telemetry.models import FleetSnapshot, utcnow
from sentinel.telemetry.sources import TelemetrySource

log = logging.getLogger(__name__)


@dataclass
class EngineState:
    snapshot: FleetSnapshot | None = None
    peers: dict[str, dict[str, PeerComparison]] = field(default_factory=dict)
    signals: list[AnomalySignal] = field(default_factory=list)
    results: dict[str, tuple[NodeEvidence, list[Hypothesis], str]] = field(default_factory=dict)
    node_status: dict[str, str] = field(default_factory=dict)
    gpu_status: dict[str, str] = field(default_factory=dict)
    cycle: int = 0
    last_cycle_ms: float = 0.0
    last_error: str | None = None
    started_at: float = field(default_factory=time.time)


class AnalysisEngine:
    def __init__(self, settings: Settings, db: Database, source: TelemetrySource, explainer: Explainer):
        self.s = settings
        self.db = db
        self.source = source
        self.explainer = explainer
        self.history = MetricHistory(settings.history_samples)
        self.peers = PeerBenchmark(settings.peer_keys, settings.peer_min_group)
        self.detectors = default_detectors()
        self.rca = RCAEngine()
        self.incidents = IncidentService(settings.tenant, settings.incident_min_cycles, settings.incident_resolve_cycles)
        self.secretbox = SecretBox(settings.secret_key, settings.jwt_secret)
        self.state = EngineState()
        self._anomaly_ids: dict[tuple[str, str, str], int] = {}
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._explain_tasks: set[asyncio.Task] = set()
        self._explaining: set[str] = set()

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self.source.close()

    async def _loop(self) -> None:
        while True:
            t0 = time.time()
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep monitoring alive
                log.exception("analysis cycle failed")
                self.state.last_error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(max(0.5, self.s.analysis_interval_s - (time.time() - t0)))

    # -------------------------------------------------------------- a cycle
    async def run_cycle(self, snapshot: FleetSnapshot | None = None, explain: bool = True) -> EngineState:
        async with self._lock:
            t0 = time.perf_counter()
            snap = snapshot or await self.source.collect()
            self.history.ingest(snap)
            peers = self.peers.run(snap, self.history)
            signals = run_detectors(DetectionContext(snap, self.history, peers), self.detectors)

            results: dict[str, tuple[NodeEvidence, list[Hypothesis], str]] = {}
            node_status, gpu_status = {}, {}
            for n in snap.nodes:
                ev = self.rca.build_evidence(n.node, snap, signals, peers)
                hyps = self.rca.evaluate(ev) if ev.signals else []
                sev = node_severity(ev, hyps) if ev.signals else "info"
                results[n.node] = (ev, hyps, sev)
                node_status[n.node] = "healthy" if not ev.signals else ("critical" if sev == "critical" else "warning")
                for g in n.gpus:
                    gs = [s for s in ev.signals if s.gpu_index == g.index]
                    gpu_status[g.key] = ("critical" if any(s.severity == "critical" and s.method == "static_threshold" for s in gs)
                                         else "warning" if gs else "healthy")
            self.state.cycle += 1
            changes = await asyncio.to_thread(self._persist_cycle, snap, peers, signals, results, node_status, gpu_status)
            self.state.snapshot, self.state.peers, self.state.signals = snap, peers, signals
            self.state.results, self.state.node_status, self.state.gpu_status = results, node_status, gpu_status
            self.state.last_cycle_ms = (time.perf_counter() - t0) * 1000
            self.state.last_error = None
        for event, inc in changes:
            await asyncio.to_thread(self.notify, inc, event)
        if explain:
            for iid in await asyncio.to_thread(self._needs_explanation):
                self._schedule_explain(iid)
        return self.state

    def _persist_cycle(self, snap, peers, signals, results, node_status, gpu_status) -> list[tuple[str, dict]]:
        with self.db.session() as db:
            changes = self.incidents.process(db, snap, peers, results)
            out = [(ev, incident_to_dict(inc)) for ev, inc in changes]
            self._upsert_inventory(db, snap, node_status, gpu_status)
            self._update_anomalies(db, signals)
            if self.state.cycle % max(1, self.s.persist_every_cycles) == 1 or self.s.persist_every_cycles == 1:
                self._store_samples(db, snap)
            if self.state.cycle % 360 == 0:
                self.apply_retention(db)
            db.commit()
            return out

    def _upsert_inventory(self, db, snap: FleetSnapshot, node_status, gpu_status) -> None:
        t = self.s.tenant
        nodes = {n.name: n for n in db.scalars(select(Node).where(Node.tenant_id == t))}
        gpus = {(g.node, g.index): g for g in db.scalars(select(GPU).where(GPU.tenant_id == t))}
        for n in snap.nodes:
            row = nodes.get(n.node)
            if row is None:
                row = Node(tenant_id=t, name=n.node, cluster=n.cluster, server_type=n.server_type, gpu_model=n.gpu_model)
                db.add(row)
            row.cluster, row.server_type, row.gpu_model, row.rack = n.cluster, n.server_type, n.gpu_model, n.rack
            row.driver_version, row.cuda_version, row.gpu_count = n.driver_version, n.cuda_version, len(n.gpus)
            row.last_seen, row.status = snap.timestamp, node_status.get(n.node, "healthy")
            for g in n.gpus:
                gr = gpus.get((n.node, g.index))
                if gr is None:
                    gr = GPU(tenant_id=t, node=n.node, index=g.index, uuid=g.uuid, model=g.model, vendor=g.vendor)
                    db.add(gr)
                gr.last_seen, gr.status, gr.uuid = snap.timestamp, gpu_status.get(g.key, "healthy"), g.uuid

    def _update_anomalies(self, db, signals: list[AnomalySignal]) -> None:
        now = utcnow()
        seen = set()
        active_inc = {r.node: r.id for r in db.scalars(select(Incident).where(
            Incident.tenant_id == self.s.tenant, Incident.status.in_(ACTIVE_STATUSES)))}
        for s in signals:
            key = (s.entity, s.metric, s.direction)
            seen.add(key)
            rid = self._anomaly_ids.get(key)
            vals = dict(last_seen=now, severity=s.severity, methods=s.methods, value=s.value, expected=s.expected,
                        deviation_pct=s.deviation_pct, zscore=s.zscore, description=s.description,
                        incident_id=active_inc.get(s.node))
            if rid is not None:
                db.execute(update(AnomalyRow).where(AnomalyRow.id == rid).values(**vals))
            else:
                row = AnomalyRow(tenant_id=self.s.tenant, first_seen=now, active=True, entity=s.entity, node=s.node,
                                 gpu_index=s.gpu_index, level=s.level, metric=s.metric, category=s.category,
                                 direction=s.direction, **vals)
                db.add(row)
                db.flush()
                self._anomaly_ids[key] = row.id
        gone = [k for k in self._anomaly_ids if k not in seen]
        if gone:
            db.execute(update(AnomalyRow).where(AnomalyRow.id.in_([self._anomaly_ids[k] for k in gone]))
                       .values(active=False))
            for k in gone:
                del self._anomaly_ids[k]

    def _store_samples(self, db, snap: FleetSnapshot) -> None:
        t, ts = self.s.tenant, snap.timestamp
        gpu_rows = [{"ts": ts, "tenant_id": t, "node": g.node, "gpu_index": g.index, "metrics": g.metrics}
                    for g in snap.all_gpus()]
        node_rows = [{"ts": ts, "tenant_id": t, "node": n.node, "metrics": n.metrics} for n in snap.nodes]
        if gpu_rows:
            db.execute(insert(GPUSampleRow), gpu_rows)
        if node_rows:
            db.execute(insert(NodeSampleRow), node_rows)
        db.execute(insert(FleetSampleRow), [{"ts": ts, "tenant_id": t, "metrics": self.fleet_rollup(snap)}])

    @staticmethod
    def fleet_rollup(snap: FleetSnapshot) -> dict:
        gpus = snap.all_gpus()
        if not gpus:
            return {}
        avg = lambda k: round(sum(g.metrics.get(k, 0) for g in gpus) / len(gpus), 2)  # noqa: E731
        return {"gpu_util": avg("gpu_util"), "temp_c": avg("temp_c"), "power_kw": round(sum(g.metrics.get("power_w", 0) for g in gpus) / 1000, 2),
                "sm_clock_mhz": avg("sm_clock_mhz"), "hbm_bw_gbps": avg("hbm_bw_gbps"),
                "throughput": round(sum(n.metrics.get("throughput", 0) for n in snap.nodes), 1)}

    def apply_retention(self, db) -> dict:
        now = utcnow()
        t = self.s.tenant
        res = {}
        cut = now - timedelta(days=self.s.retention_samples_days)
        for tbl in (GPUSampleRow, NodeSampleRow, FleetSampleRow):
            res[tbl.__tablename__] = db.execute(delete(tbl).where(tbl.tenant_id == t, tbl.ts < cut)).rowcount
        res["anomalies"] = db.execute(delete(AnomalyRow).where(AnomalyRow.tenant_id == t, AnomalyRow.active.is_(False),
                                                               AnomalyRow.last_seen < cut)).rowcount
        icut = now - timedelta(days=self.s.retention_incidents_days)
        old = db.scalars(select(Incident).where(Incident.tenant_id == t, Incident.status.in_(("RESOLVED", "FALSE_POSITIVE")),
                                                Incident.updated_at < icut)).all()
        for i in old:
            db.delete(i)
        res["incidents"] = len(old)
        res["audit_logs"] = db.execute(delete(AuditLog).where(
            AuditLog.tenant_id == t, AuditLog.ts < now - timedelta(days=self.s.retention_audit_days))).rowcount
        return res

    # ----------------------------------------------------- AI explanations
    def _needs_explanation(self) -> list[str]:
        with self.db.session() as db:
            return list(db.scalars(select(Incident.id).where(
                Incident.tenant_id == self.s.tenant, Incident.status.in_(ACTIVE_STATUSES),
                Incident.ai_explanation.is_(None))))

    def _schedule_explain(self, incident_id: str) -> None:
        if incident_id in self._explaining:
            return
        self._explaining.add(incident_id)
        task = asyncio.create_task(asyncio.to_thread(self.explain_incident, incident_id))
        self._explain_tasks.add(task)
        task.add_done_callback(lambda t: (self._explain_tasks.discard(t), self._explaining.discard(incident_id)))

    def explain_incident(self, incident_id: str, force: bool = False) -> dict | None:
        with self.db.session() as db:
            inc = db.get(Incident, incident_id)
            if inc is None:
                return None
            if inc.ai_explanation and not force:
                return inc.ai_explanation
            d = incident_to_dict(inc)
        # Template first (instant), then LLM enrichment if available.
        exp = self.explainer.explain(d, use_llm=True)
        exp["generated_at"] = utcnow().isoformat()
        exp["perf_deviation_pct"] = d.get("perf_deviation_pct")
        with self.db.session() as db:
            inc = db.get(Incident, incident_id)
            if inc is not None:
                inc.ai_explanation = exp
                db.commit()
        return exp

    # -------------------------------------------------------- notifications
    def notify(self, inc: dict, event: str) -> None:
        payload = incident_payload(inc, event)
        with self.db.session() as db:
            db.add(NotificationLog(tenant_id=self.s.tenant, channel_type="dashboard", incident_id=inc["id"], event=event,
                                   status="delivered", message=f"[{inc['severity'].upper()}] {inc['title']}"))
            channels = db.scalars(select(NotificationChannel).where(NotificationChannel.tenant_id == self.s.tenant,
                                                                    NotificationChannel.enabled.is_(True))).all()
            for ch in channels:
                if SEVERITY_RANK.get(inc["severity"], 0) < SEVERITY_RANK.get(ch.min_severity, 1) and event != "resolved":
                    continue
                if ch.type in EXTERNAL_SAAS and not self.s.allow_external_notifications:
                    status, err = "blocked", "external SaaS notifications disabled (on-prem policy)"
                else:
                    try:
                        cfg = json.loads(self.secretbox.decrypt(ch.config_encrypted))
                        NOTIFIERS[ch.type](cfg).send(payload)
                        status, err = "sent", None
                    except Exception as e:
                        status, err = "failed", str(e)[:500]
                        log.warning("notification via %s failed: %s", ch.name, e)
                db.add(NotificationLog(tenant_id=self.s.tenant, channel_id=ch.id, channel_type=ch.type,
                                       incident_id=inc["id"], event=event, status=status, error=err))
            db.commit()
