"""License model (prepared for commercial packaging; no payment processing).

A license is a JSON document signed with Ed25519 by the vendor. The platform
only holds the public key, so licenses can be verified fully offline
(air-gapped sites). Enforcement in the MVP is *soft*: exceeding limits shows
warnings in the UI and API but never stops monitoring.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

PLANS = {
    "community": {"max_gpus": 16, "features": ["monitoring", "peer_benchmarking", "anomaly_detection"]},
    "professional": {"max_gpus": 512, "features": ["monitoring", "peer_benchmarking", "anomaly_detection", "rca",
                                                    "ai_explanations", "copilot", "notifications"]},
    "enterprise": {"max_gpus": None, "features": ["monitoring", "peer_benchmarking", "anomaly_detection", "rca",
                                                   "ai_explanations", "copilot", "notifications", "sso", "multi_tenant",
                                                   "audit_export", "priority_support"]},
}


@dataclass
class License:
    customer: str
    plan: str
    license_type: str = "per_gpu"          # per_gpu | per_node | enterprise
    max_gpus: int | None = None
    max_nodes: int | None = None
    features: list[str] = field(default_factory=list)
    issued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str | None = None
    support_tier: str = "standard"
    tenant: str = "default"

    def expired(self) -> bool:
        return bool(self.expires_at) and datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc)


DEMO_LICENSE = License(customer="Demo / Evaluation", plan="enterprise", license_type="enterprise",
                       features=PLANS["enterprise"]["features"], support_tier="evaluation")


def sign_license(lic: License, private_key_pem: bytes) -> str:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    assert isinstance(key, Ed25519PrivateKey)
    body = json.dumps(asdict(lic), sort_keys=True).encode()
    return base64.b64encode(body).decode() + "." + base64.b64encode(key.sign(body)).decode()


def verify_license(token: str, public_key_pem: bytes) -> License:
    body_b64, sig_b64 = token.strip().split(".")
    body = base64.b64decode(body_b64)
    key = serialization.load_pem_public_key(public_key_pem)
    assert isinstance(key, Ed25519PublicKey)
    try:
        key.verify(base64.b64decode(sig_b64), body)
    except InvalidSignature as e:
        raise ValueError("invalid license signature") from e
    return License(**json.loads(body))


def usage_status(lic: License, gpus: int, nodes: int) -> dict:
    warnings = []
    if lic.expired():
        warnings.append("License expired — monitoring continues, please renew.")
    if lic.max_gpus is not None and gpus > lic.max_gpus:
        warnings.append(f"GPU count {gpus} exceeds licensed {lic.max_gpus}.")
    if lic.max_nodes is not None and nodes > lic.max_nodes:
        warnings.append(f"Node count {nodes} exceeds licensed {lic.max_nodes}.")
    return {"license": asdict(lic), "usage": {"gpus": gpus, "nodes": nodes}, "compliant": not warnings,
            "warnings": warnings, "enforcement": "soft"}


def generate_keypair() -> tuple[bytes, bytes]:
    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    pub = k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, pub
