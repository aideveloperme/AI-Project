
import jwt
import pytest

from sentinel.auth.security import (
    SecretBox,
    create_token,
    decode_token,
    generate_api_key,
    has_permission,
    hash_api_key,
    hash_password,
    verify_password,
)
from sentinel.licensing.license import License, generate_keypair, sign_license, usage_status, verify_license


def test_password_hashing():
    h = hash_password("correct horse battery")
    assert h.startswith("pbkdf2_sha256$") and "correct" not in h
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)
    assert hash_password("x") != hash_password("x")  # salted


def test_jwt_roundtrip_and_expiry():
    t = create_token("alice", "operator", "default", "s" * 32, 5)
    assert decode_token(t, "s" * 32)["role"] == "operator"
    with pytest.raises(jwt.PyJWTError):
        decode_token(t, "other-secret-" * 3)
    expired = create_token("alice", "operator", "default", "s" * 32, -1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(expired, "s" * 32)


def test_rbac_matrix():
    assert has_permission("viewer", "read")
    assert not has_permission("viewer", "incident.update")
    assert has_permission("operator", "incident.update") and has_permission("operator", "simulator.inject")
    assert not has_permission("operator", "user.manage")
    assert all(has_permission("admin", p) for p in ("user.manage", "audit.read", "license.manage"))


def test_api_key_only_hash_stored():
    raw, prefix, h = generate_api_key()
    assert raw.startswith("gs_") and raw.startswith(prefix) and h == hash_api_key(raw) and raw not in h


def test_secretbox():
    box = SecretBox(None, "jwt-secret")
    token = box.encrypt('{"password": "hunter2"}')
    assert "hunter2" not in token and box.decrypt(token) == '{"password": "hunter2"}'
    from cryptography.fernet import Fernet
    k = Fernet.generate_key().decode()
    assert SecretBox(k, "x").decrypt(SecretBox(k, "y").encrypt("a")) == "a"


def test_license_signing_and_soft_enforcement():
    priv, pub = generate_keypair()
    lic = License(customer="ACME AI", plan="professional", max_gpus=64, features=["rca"])
    tok = sign_license(lic, priv)
    assert verify_license(tok, pub).customer == "ACME AI"
    body, sig = tok.split(".")
    import base64
    import json
    forged = json.loads(base64.b64decode(body))
    forged["max_gpus"] = 100000
    with pytest.raises(ValueError):
        verify_license(base64.b64encode(json.dumps(forged, sort_keys=True).encode()).decode() + "." + sig, pub)
    st = usage_status(lic, gpus=96, nodes=12)
    assert not st["compliant"] and st["enforcement"] == "soft"


def test_login_rate_limit(client):
    for _ in range(10):
        client.post("/api/v1/auth/login", json={"username": "admin", "password": "nope"})
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "sentinel-admin"})
    assert r.status_code == 429


def test_security_headers_and_settings_redaction(client, admin):
    r = client.get("/api/v1/settings", headers=admin)
    assert r.headers["X-Frame-Options"] == "DENY"
    body = r.json()
    assert "jwt_secret" not in body["settings"] and "llm_api_key" not in body["settings"]
    assert body["offline_mode"]["telemetry_leaves_site"] is False
    assert body["offline_mode"]["llm_local"] is True


def test_notification_channel_secrets_encrypted(client, admin):
    r = client.post("/api/v1/notification-channels", headers=admin, json={
        "name": "ops-webhook", "type": "webhook", "config": {"url": "http://127.0.0.1:9/hook", "secret": "s3cr3t"}})
    assert r.status_code == 201
    listed = client.get("/api/v1/notification-channels", headers=admin).json()[0]
    assert listed["config"]["secret"] == "••••••"
    from sentinel.db import NotificationChannel
    with client.ctx.db.session() as db:
        row = db.get(NotificationChannel, listed["id"])
        assert "s3cr3t" not in row.config_encrypted
    # Slack is external SaaS: blocked unless explicitly allowed
    client.post("/api/v1/notification-channels", headers=admin, json={"name": "slack", "type": "slack", "config": {"url": "https://hooks.slack.com/x"}})
    sid = [c for c in client.get("/api/v1/notification-channels", headers=admin).json() if c["type"] == "slack"][0]["id"]
    assert client.post(f"/api/v1/notification-channels/{sid}/test", headers=admin).json()["ok"] is False
