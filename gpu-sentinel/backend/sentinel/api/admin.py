"""Authentication, users, API keys, notification channels, settings, audit, license, system."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from sentinel import __version__
from sentinel.api.deps import Principal, ctx, current_user, require
from sentinel.auth.security import Role, create_token, generate_api_key, hash_password, verify_password
from sentinel.db import ApiKey, AuditLog, LicenseRow, NotificationChannel, NotificationLog, User
from sentinel.licensing.license import usage_status, verify_license
from sentinel.notifications.channels import EXTERNAL_SAAS, NOTIFIERS, incident_payload
from sentinel.telemetry.models import utcnow

router = APIRouter(prefix="/api/v1", tags=["admin"])

# ------------------------------------------------------------------- auth
MAX_ATTEMPTS, WINDOW_S = 10, 300


class Login(BaseModel):
    username: str = Field(..., max_length=128)
    password: str = Field(..., max_length=256)


def audit(c, actor: str, action: str, request: Request | None = None, role: str | None = None,
          resource: str | None = None, details: dict | None = None, status_code: int | None = None) -> None:
    with c.db.session() as db:
        db.add(AuditLog(tenant_id=c.settings.tenant, actor=actor, role=role, action=action, resource=resource,
                        method=request.method if request else None, path=str(request.url.path) if request else None,
                        ip=request.client.host if request and request.client else None, details=details,
                        status_code=status_code))
        db.commit()


@router.post("/auth/login", tags=["auth"])
def login(body: Login, request: Request, c=Depends(ctx)) -> dict:
    ip = request.client.host if request.client else "unknown"
    q = c.login_attempts[ip]
    now = time.time()
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= MAX_ATTEMPTS:
        raise HTTPException(429, "too many login attempts, try again later")
    with c.db.session() as db:
        u = db.scalars(select(User).where(User.username == body.username)).first()
        if u is None or not u.is_active or not verify_password(body.password, u.password_hash):
            q.append(now)
            audit(c, body.username, "auth.login_failed", request, status_code=401)
            raise HTTPException(401, "invalid credentials")
        u.last_login = utcnow()
        db.commit()
        token = create_token(u.username, u.role, u.tenant_id, c.settings.jwt_secret, c.settings.jwt_ttl_minutes)
        audit(c, u.username, "auth.login", request, role=u.role, status_code=200)
        return {"access_token": token, "token_type": "bearer", "expires_in": c.settings.jwt_ttl_minutes * 60,
                "user": {"username": u.username, "role": u.role, "tenant": u.tenant_id}}


@router.get("/auth/me", tags=["auth"])
def me(p: Principal = Depends(current_user)) -> dict:
    return {"username": p.username, "role": p.role, "tenant": p.tenant, "via": p.via}


# ------------------------------------------------------------------ users
class UserIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=128, pattern=r"^[A-Za-z0-9_.@\-]+$")
    password: str = Field(..., min_length=10, max_length=256)
    role: Role
    email: str | None = None


class UserPatch(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=10, max_length=256)


@router.get("/users")
def list_users(p: Principal = Depends(require("user.manage")), c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        return [{"id": u.id, "username": u.username, "email": u.email, "role": u.role, "is_active": u.is_active,
                 "last_login": u.last_login.isoformat() if u.last_login else None}
                for u in db.scalars(select(User).where(User.tenant_id == p.tenant))]


@router.post("/users", status_code=201)
def create_user(body: UserIn, request: Request, p: Principal = Depends(require("user.manage")), c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        if db.scalars(select(User).where(User.username == body.username)).first():
            raise HTTPException(409, "username exists")
        u = User(tenant_id=p.tenant, username=body.username, email=body.email, role=body.role.value,
                 password_hash=hash_password(body.password))
        db.add(u)
        db.commit()
        audit(c, p.username, "user.create", request, p.role, f"user:{u.username}", {"role": u.role})
        return {"id": u.id, "username": u.username, "role": u.role}


@router.patch("/users/{user_id}")
def patch_user(user_id: int, body: UserPatch, request: Request, p: Principal = Depends(require("user.manage")),
               c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        u = db.get(User, user_id)
        if u is None or u.tenant_id != p.tenant:
            raise HTTPException(404, "user not found")
        if body.role is not None:
            u.role = body.role.value
        if body.is_active is not None:
            if u.username == p.username and not body.is_active:
                raise HTTPException(400, "cannot disable yourself")
            u.is_active = body.is_active
        if body.password:
            u.password_hash = hash_password(body.password)
        db.commit()
        audit(c, p.username, "user.update", request, p.role, f"user:{u.username}",
              {"role": u.role, "is_active": u.is_active, "password_changed": bool(body.password)})
        return {"id": u.id, "username": u.username, "role": u.role, "is_active": u.is_active}


# --------------------------------------------------------------- API keys
class KeyIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    role: Role = Role.viewer


@router.get("/api-keys")
def list_keys(p: Principal = Depends(require("apikey.manage")), c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        return [{"id": k.id, "name": k.name, "prefix": k.prefix, "role": k.role, "created_by": k.created_by,
                 "created_at": k.created_at.isoformat(), "last_used": k.last_used.isoformat() if k.last_used else None,
                 "revoked": k.revoked} for k in db.scalars(select(ApiKey).where(ApiKey.tenant_id == p.tenant))]


@router.post("/api-keys", status_code=201)
def create_key(body: KeyIn, request: Request, p: Principal = Depends(require("apikey.manage")), c=Depends(ctx)) -> dict:
    raw, prefix, h = generate_api_key()
    with c.db.session() as db:
        k = ApiKey(tenant_id=p.tenant, name=body.name, prefix=prefix, key_hash=h, role=body.role.value, created_by=p.username)
        db.add(k)
        db.commit()
        audit(c, p.username, "apikey.create", request, p.role, f"apikey:{body.name}", {"role": body.role.value})
        return {"id": k.id, "name": k.name, "role": k.role, "key": raw,
                "note": "Store this key now — it is not retrievable later."}


@router.delete("/api-keys/{key_id}")
def revoke_key(key_id: int, request: Request, p: Principal = Depends(require("apikey.manage")), c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        k = db.get(ApiKey, key_id)
        if k is None or k.tenant_id != p.tenant:
            raise HTTPException(404, "key not found")
        k.revoked = True
        db.commit()
        audit(c, p.username, "apikey.revoke", request, p.role, f"apikey:{k.name}")
        return {"revoked": True}


# --------------------------------------------------------- notifications
class ChannelIn(BaseModel):
    name: str = Field(..., max_length=128)
    type: str
    config: dict
    min_severity: str = "warning"
    enabled: bool = True


def _redact(cfg: dict) -> dict:
    return {k: ("••••••" if any(s in k.lower() for s in ("password", "secret", "token", "key")) else v)
            for k, v in cfg.items()}


@router.get("/notification-channels")
def list_channels(p: Principal = Depends(require("notification.manage")), c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        out = []
        for ch in db.scalars(select(NotificationChannel).where(NotificationChannel.tenant_id == p.tenant)):
            cfg = json.loads(c.engine.secretbox.decrypt(ch.config_encrypted))
            out.append({"id": ch.id, "name": ch.name, "type": ch.type, "config": _redact(cfg),
                        "min_severity": ch.min_severity, "enabled": ch.enabled,
                        "external_saas": ch.type in EXTERNAL_SAAS})
        return out


@router.post("/notification-channels", status_code=201)
def create_channel(body: ChannelIn, request: Request, p: Principal = Depends(require("notification.manage")),
                   c=Depends(ctx)) -> dict:
    if body.type not in NOTIFIERS:
        raise HTTPException(400, f"unsupported channel type; supported: {sorted(NOTIFIERS)}")
    with c.db.session() as db:
        ch = NotificationChannel(tenant_id=p.tenant, name=body.name, type=body.type, min_severity=body.min_severity,
                                 enabled=body.enabled, config_encrypted=c.engine.secretbox.encrypt(json.dumps(body.config)))
        db.add(ch)
        db.commit()
        audit(c, p.username, "notification.create", request, p.role, f"channel:{body.name}", {"type": body.type})
        return {"id": ch.id, "name": ch.name, "type": ch.type}


@router.delete("/notification-channels/{channel_id}")
def delete_channel(channel_id: int, request: Request, p: Principal = Depends(require("notification.manage")),
                   c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        ch = db.get(NotificationChannel, channel_id)
        if ch is None or ch.tenant_id != p.tenant:
            raise HTTPException(404, "channel not found")
        db.delete(ch)
        db.commit()
        audit(c, p.username, "notification.delete", request, p.role, f"channel:{ch.name}")
        return {"deleted": True}


@router.post("/notification-channels/{channel_id}/test")
def test_channel(channel_id: int, p: Principal = Depends(require("notification.manage")), c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        ch = db.get(NotificationChannel, channel_id)
        if ch is None or ch.tenant_id != p.tenant:
            raise HTTPException(404, "channel not found")
        cfg = json.loads(c.engine.secretbox.decrypt(ch.config_encrypted))
    if ch.type in EXTERNAL_SAAS and not c.settings.allow_external_notifications:
        return {"ok": False, "error": "external SaaS notifications disabled (SENTINEL_ALLOW_EXTERNAL_NOTIFICATIONS=false)"}
    test = {"id": "INC-TEST", "node": "test-node", "gpus": [0], "severity": "warning", "status": "OPEN",
            "title": "GPU Sentinel test notification", "summary": "This is a test notification.", "category": "test",
            "confidence": 1.0, "confidence_label": "high", "recommended_actions": ["No action required."]}
    try:
        NOTIFIERS[ch.type](cfg).send(incident_payload(test, "test"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


@router.get("/notification-log")
def notification_log(limit: int = 100, p: Principal = Depends(require("notification.manage")), c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        rows = db.scalars(select(NotificationLog).where(NotificationLog.tenant_id == p.tenant)
                          .order_by(NotificationLog.ts.desc()).limit(limit)).all()
        return [{"ts": r.ts.isoformat(), "channel_type": r.channel_type, "incident_id": r.incident_id, "event": r.event,
                 "status": r.status, "error": r.error} for r in rows]


# ---------------------------------------------------------------- settings
SAFE_SETTINGS = ["env", "demo_mode", "telemetry_source", "prometheus_url", "gpu_vendor", "analysis_interval_s",
                 "history_samples", "peer_group_keys", "peer_min_group", "incident_min_cycles", "incident_resolve_cycles",
                 "persist_every_cycles", "retention_samples_days", "retention_incidents_days", "retention_audit_days",
                 "llm_provider", "llm_url", "llm_model", "allow_cloud_llm", "allow_external_notifications",
                 "notify_min_severity", "jwt_ttl_minutes"]


@router.get("/settings")
def get_settings_view(p: Principal = Depends(require("read")), c=Depends(ctx)) -> dict:
    s = c.settings
    return {"settings": {k: getattr(s, k) for k in SAFE_SETTINGS},
            "security": {"secret_key_configured": bool(s.secret_key),
                         "default_jwt_secret": s.jwt_secret.startswith("change-me"),
                         "default_admin_password": s.bootstrap_admin_password == "sentinel-admin"},
            "offline_mode": {"telemetry_leaves_site": bool(c.explainer.provider and not c.explainer.provider.local)
                             or s.allow_external_notifications,
                             "llm_local": c.explainer.provider is None or c.explainer.provider.local,
                             "llm_provider": c.explainer.provider_name}}


class RuntimeSettings(BaseModel):
    incident_min_cycles: int | None = Field(None, ge=1, le=100)
    incident_resolve_cycles: int | None = Field(None, ge=1, le=1000)
    retention_samples_days: int | None = Field(None, ge=1, le=3650)
    retention_incidents_days: int | None = Field(None, ge=1, le=3650)
    retention_audit_days: int | None = Field(None, ge=30, le=3650)


@router.patch("/settings")
def patch_settings(body: RuntimeSettings, request: Request, p: Principal = Depends(require("settings.manage")),
                   c=Depends(ctx)) -> dict:
    changed = {k: v for k, v in body.model_dump().items() if v is not None}
    for k, v in changed.items():
        setattr(c.settings, k, v)
    c.engine.incidents.min_cycles = c.settings.incident_min_cycles
    c.engine.incidents.resolve_cycles = c.settings.incident_resolve_cycles
    audit(c, p.username, "settings.update", request, p.role, "settings", changed)
    return {"updated": changed}


@router.post("/retention/run")
def run_retention(request: Request, p: Principal = Depends(require("settings.manage")), c=Depends(ctx)) -> dict:
    with c.db.session() as db:
        res = c.engine.apply_retention(db)
        db.commit()
    audit(c, p.username, "retention.run", request, p.role, details=res)
    return res


# ------------------------------------------------------------------- audit
@router.get("/audit-logs")
def audit_logs(limit: int = 200, actor: str | None = None, p: Principal = Depends(require("audit.read")),
               c=Depends(ctx)) -> list[dict]:
    with c.db.session() as db:
        q = select(AuditLog).where(AuditLog.tenant_id == p.tenant)
        if actor:
            q = q.where(AuditLog.actor == actor)
        rows = db.scalars(q.order_by(AuditLog.ts.desc()).limit(min(limit, 5000))).all()
        return [{"ts": r.ts.isoformat(), "actor": r.actor, "role": r.role, "action": r.action, "resource": r.resource,
                 "method": r.method, "path": r.path, "status_code": r.status_code, "ip": r.ip, "details": r.details}
                for r in rows]


# ----------------------------------------------------------------- license
class LicenseIn(BaseModel):
    token: str


@router.get("/license")
def license_status(p: Principal = Depends(require("read")), c=Depends(ctx)) -> dict:
    snap = c.engine.state.snapshot
    gpus = len(snap.all_gpus()) if snap else 0
    nodes = len(snap.nodes) if snap else 0
    return usage_status(c.license, gpus, nodes)


@router.post("/license")
def install_license(body: LicenseIn, request: Request, p: Principal = Depends(require("license.manage")),
                    c=Depends(ctx)) -> dict:
    if not c.settings.license_public_key:
        raise HTTPException(400, "no license public key configured (SENTINEL_LICENSE_PUBLIC_KEY)")
    try:
        lic = verify_license(body.token, c.settings.license_public_key.encode())
    except Exception as e:
        raise HTTPException(400, f"invalid license: {e}") from e
    from datetime import datetime
    with c.db.session() as db:
        db.add(LicenseRow(tenant_id=p.tenant, customer=lic.customer, plan=lic.plan, max_gpus=lic.max_gpus,
                          max_nodes=lic.max_nodes, features=lic.features, issued_at=datetime.fromisoformat(lic.issued_at),
                          expires_at=datetime.fromisoformat(lic.expires_at) if lic.expires_at else None,
                          support_tier=lic.support_tier, raw=body.token))
        db.commit()
    c.license = lic
    audit(c, p.username, "license.install", request, p.role, f"license:{lic.customer}", {"plan": lic.plan})
    return license_status(p, c)


# ------------------------------------------------------------------ system
@router.get("/system/info")
def system_info(p: Principal = Depends(require("read")), c=Depends(ctx)) -> dict:
    st = c.engine.state
    return {"version": __version__, "telemetry_source": c.engine.source.name, "gpu_vendor": c.settings.gpu_vendor,
            "llm_provider": c.explainer.provider_name,
            "llm_local": c.explainer.provider is None or c.explainer.provider.local,
            "demo_mode": c.settings.demo_mode, "analysis_interval_s": c.settings.analysis_interval_s,
            "cycle": st.cycle, "last_cycle_ms": round(st.last_cycle_ms, 1), "last_error": st.last_error,
            "uptime_s": round(time.time() - st.started_at), "tracked_series": len(c.engine.history),
            "database": c.db.engine.dialect.name}
