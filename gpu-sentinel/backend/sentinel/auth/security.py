"""Authentication primitives: password hashing, JWT, API keys, secret encryption, RBAC."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import timedelta
from enum import Enum

import jwt
from cryptography.fernet import Fernet

from sentinel.telemetry.models import utcnow

PBKDF2_ITERATIONS = 390_000


class Role(str, Enum):
    viewer = "viewer"
    operator = "operator"
    admin = "admin"


ROLE_RANK = {Role.viewer: 0, Role.operator: 1, Role.admin: 2}

# Permission matrix (documented in docs/ARCHITECTURE.md §8).
PERMISSIONS: dict[str, Role] = {
    "read": Role.viewer,              # dashboards, incidents, telemetry, Ask Sentinel
    "incident.update": Role.operator,  # ack / investigate / resolve / comment
    "simulator.inject": Role.operator,
    "notification.manage": Role.admin,
    "user.manage": Role.admin,
    "apikey.manage": Role.admin,
    "settings.manage": Role.admin,
    "license.manage": Role.admin,
    "audit.read": Role.admin,
}


def has_permission(role: str, permission: str) -> bool:
    need = PERMISSIONS[permission]
    return ROLE_RANK[Role(role)] >= ROLE_RANK[need]


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except Exception:
        return False


def create_token(subject: str, role: str, tenant: str, secret: str, ttl_minutes: int) -> str:
    now = utcnow()
    payload = {"sub": subject, "role": role, "tenant": tenant, "iat": now,
               "exp": now + timedelta(minutes=ttl_minutes), "iss": "gpu-sentinel"}
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"], issuer="gpu-sentinel")


def generate_api_key() -> tuple[str, str, str]:
    """Returns (plaintext_key, prefix, sha256_hash). Only the hash is stored."""
    raw = "gs_" + secrets.token_urlsafe(32)
    return raw, raw[:10], hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class SecretBox:
    """Encrypts credentials at rest (SMTP passwords, webhook secrets, tokens).

    The key comes from SENTINEL_SECRET_KEY (a Fernet key, e.g. mounted from a
    Kubernetes Secret / Vault). If absent, a key is derived from the JWT secret
    so development works — production deployments must set it explicitly.
    """

    def __init__(self, key: str | None, fallback_secret: str):
        if key:
            self.fernet = Fernet(key.encode())
        else:
            derived = hashlib.sha256(("sentinel-secretbox:" + fallback_secret).encode()).digest()
            self.fernet = Fernet(base64.urlsafe_b64encode(derived))

    def encrypt(self, plaintext: str) -> str:
        return self.fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self.fernet.decrypt(token.encode()).decode()
