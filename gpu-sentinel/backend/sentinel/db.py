"""Database schema (SQLAlchemy 2.0).

Runs on SQLite for development/tests and on PostgreSQL + TimescaleDB in
production. On PostgreSQL the sample tables are converted to Timescale
hypertables (with compression + retention policies) when the extension is
available; see :func:`init_db`.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine,
                        event, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from sentinel.telemetry.models import utcnow

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    plan: Mapped[str] = mapped_column(String(32), default="enterprise")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenants.id"), index=True)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    email: Mapped[str | None] = mapped_column(String(256))
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))  # admin | operator | viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16))
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Node(Base):
    __tablename__ = "nodes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    cluster: Mapped[str] = mapped_column(String(128))
    server_type: Mapped[str] = mapped_column(String(128))
    gpu_model: Mapped[str] = mapped_column(String(128))
    gpu_count: Mapped[int] = mapped_column(Integer, default=0)
    rack: Mapped[str | None] = mapped_column(String(64))
    driver_version: Mapped[str | None] = mapped_column(String(64))
    cuda_version: Mapped[str | None] = mapped_column(String(32))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="healthy")
    __table_args__ = (Index("ux_nodes_tenant_name", "tenant_id", "name", unique=True),)


class GPU(Base):
    __tablename__ = "gpus"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    node: Mapped[str] = mapped_column(String(128), index=True)
    index: Mapped[int] = mapped_column(Integer)
    uuid: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))
    vendor: Mapped[str] = mapped_column(String(32), default="nvidia")
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="healthy")
    __table_args__ = (Index("ux_gpus_tenant_node_idx", "tenant_id", "node", "index", unique=True),)


class GPUSampleRow(Base):
    """Downsampled GPU telemetry (TimescaleDB hypertable on ``ts``)."""
    __tablename__ = "gpu_samples"
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    node: Mapped[str] = mapped_column(String(128), primary_key=True)
    gpu_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (Index("ix_gpu_samples_node_ts", "node", "ts"),)


class NodeSampleRow(Base):
    __tablename__ = "node_samples"
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    node: Mapped[str] = mapped_column(String(128), primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (Index("ix_node_samples_node_ts", "node", "ts"),)


class FleetSampleRow(Base):
    """Fleet-level rollup per cycle (drives overview trends cheaply)."""
    __tablename__ = "fleet_samples"
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON)


class AnomalyRow(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    entity: Mapped[str] = mapped_column(String(160), index=True)
    node: Mapped[str] = mapped_column(String(128), index=True)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    level: Mapped[str] = mapped_column(String(8))
    metric: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(32))
    direction: Mapped[str] = mapped_column(String(8))
    severity: Mapped[str] = mapped_column(String(16))
    methods: Mapped[list] = mapped_column(JSON)
    value: Mapped[float] = mapped_column(Float)
    expected: Mapped[float | None] = mapped_column(Float)
    deviation_pct: Mapped[float | None] = mapped_column(Float)
    zscore: Mapped[float | None] = mapped_column(Float)
    description: Mapped[str] = mapped_column(Text)
    incident_id: Mapped[str | None] = mapped_column(String(32), index=True)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # INC-000123
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_signal_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    node: Mapped[str] = mapped_column(String(128), index=True)
    gpus: Mapped[list] = mapped_column(JSON, default=list)
    workload_id: Mapped[str | None] = mapped_column(String(128))
    severity: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # OPEN|ACKNOWLEDGED|INVESTIGATING|RESOLVED|FALSE_POSITIVE
    category: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str] = mapped_column(Text)
    perf_deviation_pct: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_label: Mapped[str] = mapped_column(String(8), default="low")
    observed: Mapped[dict] = mapped_column(JSON, default=dict)
    signals: Mapped[list] = mapped_column(JSON, default=list)
    peer_comparison: Mapped[list] = mapped_column(JSON, default=list)
    hypotheses: Mapped[list] = mapped_column(JSON, default=list)
    recommended_actions: Mapped[list] = mapped_column(JSON, default=list)
    ai_explanation: Mapped[dict | None] = mapped_column(JSON)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    assigned_to: Mapped[str | None] = mapped_column(String(128))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution: Mapped[str | None] = mapped_column(Text)
    recurrence_count: Mapped[int] = mapped_column(Integer, default=0)
    related_incidents: Mapped[list] = mapped_column(JSON, default=list)
    events: Mapped[list["IncidentEvent"]] = relationship(back_populates="incident", order_by="IncidentEvent.ts",
                                                        cascade="all, delete-orphan")


class IncidentEvent(Base):
    __tablename__ = "incident_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(32), ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(128))
    type: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSON)
    incident: Mapped[Incident] = relationship(back_populates="events")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(128))
    role: Mapped[str | None] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(128))
    resource: Mapped[str | None] = mapped_column(String(256))
    method: Mapped[str | None] = mapped_column(String(8))
    path: Mapped[str | None] = mapped_column(String(512))
    status_code: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict | None] = mapped_column(JSON)


class NotificationChannel(Base):
    __tablename__ = "notification_channels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    type: Mapped[str] = mapped_column(String(32))  # webhook | email | slack | teams | pagerduty
    config_encrypted: Mapped[str] = mapped_column(Text)
    min_severity: Mapped[str] = mapped_column(String(16), default="warning")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NotificationLog(Base):
    __tablename__ = "notification_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    channel_id: Mapped[int | None] = mapped_column(Integer)
    channel_type: Mapped[str] = mapped_column(String(32))
    incident_id: Mapped[str | None] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    event: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    message: Mapped[str | None] = mapped_column(Text)


class SettingRow(Base):
    __tablename__ = "settings"
    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LicenseRow(Base):
    __tablename__ = "licenses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    customer: Mapped[str] = mapped_column(String(200))
    plan: Mapped[str] = mapped_column(String(32))
    max_gpus: Mapped[int | None] = mapped_column(Integer)
    max_nodes: Mapped[int | None] = mapped_column(Integer)
    features: Mapped[list] = mapped_column(JSON, default=list)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    support_tier: Mapped[str | None] = mapped_column(String(32))
    raw: Mapped[str] = mapped_column(Text)
    installed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------- engine
class Database:
    def __init__(self, url: str):
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.url = url
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragma(conn, _):  # pragma: no cover - trivial
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    def session(self) -> Session:
        return self.SessionLocal()

    def init(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.engine.dialect.name == "postgresql":
            self._timescale()

    def _timescale(self) -> None:
        """Best effort: hypertables + compression if TimescaleDB is installed."""
        try:
            with self.engine.begin() as c:
                c.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                for tbl in ("gpu_samples", "node_samples", "fleet_samples"):
                    c.execute(text(f"SELECT create_hypertable('{tbl}', 'ts', if_not_exists => TRUE, migrate_data => TRUE)"))
                c.execute(text("ALTER TABLE gpu_samples SET (timescaledb.compress, timescaledb.compress_segmentby = 'node')"))
                c.execute(text("SELECT add_compression_policy('gpu_samples', INTERVAL '1 day', if_not_exists => TRUE)"))
            log.info("TimescaleDB hypertables enabled")
        except Exception as e:  # plain PostgreSQL works too; retention handled by the app
            log.warning("TimescaleDB not available, using plain PostgreSQL tables: %s", e)
