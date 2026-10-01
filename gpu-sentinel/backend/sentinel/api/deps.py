"""Shared API dependencies: application context, authentication, RBAC."""
from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from sentinel.auth.security import decode_token, has_permission, hash_api_key
from sentinel.db import ApiKey, User
from sentinel.telemetry.models import utcnow

bearer = HTTPBearer(auto_error=False)


@dataclass
class Principal:
    username: str
    role: str
    tenant: str
    via: str  # jwt | apikey


def ctx(request: Request):
    return request.app.state.ctx


def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 x_api_key: str | None = Header(default=None)) -> Principal:
    c = request.app.state.ctx
    if creds and creds.scheme.lower() == "bearer":
        try:
            p = decode_token(creds.credentials, c.settings.jwt_secret)
        except jwt.PyJWTError as e:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token") from e
        with c.db.session() as db:
            u = db.scalars(select(User).where(User.username == p["sub"])).first()
            if u is None or not u.is_active:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user disabled")
            principal = Principal(u.username, u.role, u.tenant_id, "jwt")
    elif x_api_key:
        h = hash_api_key(x_api_key)
        with c.db.session() as db:
            k = db.scalars(select(ApiKey).where(ApiKey.key_hash == h, ApiKey.revoked.is_(False))).first()
            if k is None:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
            k.last_used = utcnow()
            db.commit()
            principal = Principal(f"apikey:{k.name}", k.role, k.tenant_id, "apikey")
    else:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required",
                            headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = principal
    return principal


def require(permission: str):
    def dep(p: Principal = Depends(current_user)) -> Principal:
        if not has_permission(p.role, permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{p.role}' lacks permission '{permission}'")
        return p
    return dep
