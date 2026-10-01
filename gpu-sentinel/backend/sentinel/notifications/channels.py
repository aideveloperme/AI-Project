"""Notification channels.

Every channel implements :class:`Notifier.send`. New integrations
(Slack, Teams, PagerDuty, Opsgenie, ServiceNow…) are one class each and are
registered in :data:`NOTIFIERS`. Dashboard alerts are always recorded in
``notification_log`` regardless of external channels.
"""
from __future__ import annotations

import abc
import hashlib
import hmac
import json
import logging
import smtplib
from email.message import EmailMessage

import httpx

log = logging.getLogger(__name__)

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def incident_payload(inc: dict, event: str) -> dict:
    """Stable, versioned outbound payload (never includes raw credentials)."""
    return {
        "schema": "gpu-sentinel.incident.v1",
        "event": event,  # opened | updated | escalated | resolved
        "incident": {k: inc.get(k) for k in (
            "id", "node", "gpus", "severity", "status", "title", "summary", "category", "confidence",
            "confidence_label", "perf_deviation_pct", "created_at", "updated_at", "recommended_actions", "workload_id")},
    }


class Notifier(abc.ABC):
    type: str

    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    def send(self, payload: dict) -> None: ...


class WebhookNotifier(Notifier):
    """Generic JSON webhook, HMAC-SHA256 signed (``X-Sentinel-Signature``)."""
    type = "webhook"

    def send(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "gpu-sentinel/0.1"}
        if secret := self.config.get("secret"):
            headers["X-Sentinel-Signature"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        headers.update(self.config.get("headers", {}))
        r = httpx.post(self.config["url"], content=body, headers=headers, timeout=10,
                       verify=self.config.get("verify_tls", True))
        r.raise_for_status()


class EmailNotifier(Notifier):
    type = "email"

    def send(self, payload: dict) -> None:
        inc = payload["incident"]
        msg = EmailMessage()
        msg["Subject"] = f"[GPU Sentinel][{inc['severity'].upper()}] {inc['id']} {inc['title']}"
        msg["From"] = self.config.get("from", "gpu-sentinel@localhost")
        msg["To"] = ", ".join(self.config["to"])
        lines = [inc["summary"], "", f"Status: {inc['status']}  Confidence: {inc.get('confidence_label')}", "",
                 "Recommended investigation:"] + [f"  {i + 1}. {a}" for i, a in enumerate(inc.get("recommended_actions") or [])]
        msg.set_content("\n".join(lines))
        with smtplib.SMTP(self.config["host"], int(self.config.get("port", 587)), timeout=15) as s:
            if self.config.get("starttls", True):
                s.starttls()
            if self.config.get("user"):
                s.login(self.config["user"], self.config["password"])
            s.send_message(msg)


class SlackNotifier(WebhookNotifier):
    """Slack incoming webhook (opt-in; external service)."""
    type = "slack"

    def send(self, payload: dict) -> None:
        inc = payload["incident"]
        text = f"*[{inc['severity'].upper()}] {inc['id']} — {inc['title']}*\n{inc['summary']}"
        httpx.post(self.config["url"], json={"text": text}, timeout=10).raise_for_status()


class TeamsNotifier(WebhookNotifier):
    type = "teams"

    def send(self, payload: dict) -> None:
        inc = payload["incident"]
        card = {"@type": "MessageCard", "@context": "https://schema.org/extensions",
                "summary": inc["title"], "title": f"[{inc['severity'].upper()}] {inc['id']} {inc['title']}",
                "text": inc["summary"]}
        httpx.post(self.config["url"], json=card, timeout=10).raise_for_status()


class PagerDutyNotifier(Notifier):
    type = "pagerduty"

    def send(self, payload: dict) -> None:
        inc = payload["incident"]
        action = "resolve" if payload["event"] == "resolved" else "trigger"
        body = {"routing_key": self.config["routing_key"], "event_action": action, "dedup_key": inc["id"],
                "payload": {"summary": f"{inc['id']} {inc['title']}", "source": inc["node"],
                            "severity": "critical" if inc["severity"] == "critical" else "warning",
                            "custom_details": inc}}
        url = self.config.get("url", "https://events.pagerduty.com/v2/enqueue")
        httpx.post(url, json=body, timeout=10).raise_for_status()


NOTIFIERS: dict[str, type[Notifier]] = {
    n.type: n for n in (WebhookNotifier, EmailNotifier, SlackNotifier, TeamsNotifier, PagerDutyNotifier)
}
# Channels that send data to third-party SaaS — blocked unless explicitly allowed in settings.
EXTERNAL_SAAS = {"slack", "teams", "pagerduty"}
