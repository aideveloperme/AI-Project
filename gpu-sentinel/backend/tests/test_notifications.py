import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from sentinel.notifications.channels import WebhookNotifier, incident_payload


def test_webhook_signed_payload():
    received = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received["body"], received["sig"] = body, self.headers["X-Sentinel-Signature"]
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    inc = {"id": "INC-1", "node": "gpu-01", "severity": "critical", "title": "t", "summary": "s", "status": "OPEN"}
    WebhookNotifier({"url": f"http://127.0.0.1:{srv.server_port}/", "secret": "k"}).send(incident_payload(inc, "opened"))
    srv.server_close()
    expected = "sha256=" + hmac.new(b"k", received["body"], hashlib.sha256).hexdigest()
    assert received["sig"] == expected
    data = json.loads(received["body"])
    assert data["schema"] == "gpu-sentinel.incident.v1" and data["incident"]["id"] == "INC-1"


def test_engine_dispatches_to_webhook_on_incident(client, admin, operator):
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        client.post("/api/v1/notification-channels", headers=admin, json={
            "name": "hook", "type": "webhook", "config": {"url": f"http://127.0.0.1:{srv.server_port}/"}, "min_severity": "warning"})
        client.cycles(15)
        client.post("/api/v1/demo/faults", headers=operator, json={"type": "thermal", "node": "gpu-03", "gpu": 0, "severity": 1.0})
        client.cycles(10)
    finally:
        srv.shutdown()
    assert any(p["event"] == "opened" and p["incident"]["node"] == "gpu-03" for p in got)
    log = client.get("/api/v1/notification-log", headers=admin).json()
    assert any(r["channel_type"] == "webhook" and r["status"] == "sent" for r in log)
