"""FastAPI application factory (``uvicorn sentinel.app:app``)."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import select

from sentinel import __version__
from sentinel.ai.explainer import Explainer
from sentinel.ai.providers import build_provider
from sentinel.api import admin, fleet, ops
from sentinel.assistant.service import Assistant
from sentinel.auth.security import decode_token, hash_password
from sentinel.config import Settings, get_settings
from sentinel.db import AuditLog, Database, Tenant, User
from sentinel.engine import AnalysisEngine
from sentinel.licensing.license import DEMO_LICENSE, License, verify_license
from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.sources import PrometheusSource, SimulatorSource, TelemetrySource

log = logging.getLogger("sentinel")


@dataclass
class AppContext:
    settings: Settings
    db: Database
    engine: AnalysisEngine
    explainer: Explainer
    assistant: Assistant
    license: License
    login_attempts: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))


def build_source(s: Settings) -> TelemetrySource:
    if s.telemetry_source == "prometheus":
        return PrometheusSource(s.prometheus_url, vendor=s.gpu_vendor)
    sim = ClusterSimulator(n_nodes=s.sim_nodes, gpus_per_node=s.sim_gpus_per_node, seed=s.sim_seed)
    return SimulatorSource(sim, dt=s.analysis_interval_s, warmup_s=120)


def bootstrap(s: Settings, db: Database) -> None:
    with db.session() as ses:
        if ses.get(Tenant, s.tenant) is None:
            ses.add(Tenant(id=s.tenant, name=s.tenant.title()))
            ses.flush()
        if ses.scalars(select(User).where(User.role == "admin")).first() is None:
            ses.add(User(tenant_id=s.tenant, username=s.bootstrap_admin_user, role="admin",
                         password_hash=hash_password(s.bootstrap_admin_password)))
            if s.demo_mode:
                ses.add(User(tenant_id=s.tenant, username="operator", role="operator",
                             password_hash=hash_password("sentinel-operator")))
                ses.add(User(tenant_id=s.tenant, username="viewer", role="viewer",
                             password_hash=hash_password("sentinel-viewer")))
            log.warning("Bootstrapped admin user '%s' — change the password immediately.", s.bootstrap_admin_user)
        ses.commit()


def load_license(s: Settings) -> License:
    if s.license_file and s.license_public_key:
        try:
            with open(s.license_file) as f:
                return verify_license(f.read(), s.license_public_key.encode())
        except Exception as e:
            log.error("license invalid: %s", e)
    return DEMO_LICENSE


def create_app(settings: Settings | None = None, autostart: bool = True, source: TelemetrySource | None = None) -> FastAPI:
    s = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per PromQL query is noise

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(s.database_url)
        db.init()
        bootstrap(s, db)
        provider = build_provider(s)
        explainer = Explainer(provider)
        engine = AnalysisEngine(s, db, source or build_source(s), explainer)
        app.state.ctx = AppContext(settings=s, db=db, engine=engine, explainer=explainer,
                                   assistant=Assistant(engine, provider), license=load_license(s))
        log.info("GPU Sentinel %s: source=%s llm=%s db=%s", __version__, engine.source.name, explainer.provider_name,
                 db.engine.dialect.name)
        if autostart:
            await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="GPU Sentinel AI", version=__version__, lifespan=lifespan,
                  description="Data-center performance & health intelligence for GPU clusters.")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def security_and_audit(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        # Generic audit trail for every state-changing request on operational endpoints.
        path = request.url.path
        if (request.method in ("POST", "PATCH", "PUT", "DELETE") and path.startswith("/api/v1/")
                and not path.startswith(("/api/v1/auth/", "/api/v1/ask", "/api/v1/users", "/api/v1/api-keys",
                                         "/api/v1/notification-channels", "/api/v1/settings", "/api/v1/license",
                                         "/api/v1/retention"))):
            actor, role = "anonymous", None
            p = getattr(request.state, "principal", None)
            if p is not None:
                actor, role = p.username, p.role
            else:
                auth = request.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    try:
                        claims = decode_token(auth[7:], s.jwt_secret)
                        actor, role = claims["sub"], claims["role"]
                    except Exception:
                        pass
            c = request.app.state.ctx
            with c.db.session() as db:
                db.add(AuditLog(tenant_id=s.tenant, actor=actor, role=role, action=f"{request.method} {path}",
                                resource=path.removeprefix("/api/v1/"), method=request.method, path=path,
                                status_code=response.status_code,
                                ip=request.client.host if request.client else None))
                db.commit()
        return response

    app.include_router(admin.router)
    app.include_router(fleet.router)
    app.include_router(ops.router)

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["system"])
    def readyz(request: Request) -> dict:
        st = request.app.state.ctx.engine.state
        return {"ready": st.snapshot is not None, "cycle": st.cycle, "last_error": st.last_error}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["system"])
    def self_metrics(request: Request) -> str:
        """Sentinel's own health metrics (scraped by Prometheus)."""
        st = request.app.state.ctx.engine.state
        lines = [
            "# TYPE sentinel_analysis_cycles_total counter", f"sentinel_analysis_cycles_total {st.cycle}",
            "# TYPE sentinel_analysis_cycle_ms gauge", f"sentinel_analysis_cycle_ms {st.last_cycle_ms:.2f}",
            "# TYPE sentinel_active_signals gauge", f"sentinel_active_signals {len(st.signals)}",
        ]
        for status in ("healthy", "warning", "critical"):
            lines.append(f'sentinel_gpus{{status="{status}"}} {sum(1 for v in st.gpu_status.values() if v == status)}')
        return "\n".join(lines) + "\n"

    return app


app = create_app()
