"""End-to-end: simulated telemetry → analytics → incident → API (first milestone)."""
from tests.conftest import login


def test_health_and_auth_required(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/api/v1/overview").status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401


def test_overview_healthy_fleet(client, admin):
    client.cycles(25)
    ov = client.get("/api/v1/overview", headers=admin).json()
    assert ov["total_nodes"] == 12 and ov["total_gpus"] == 96
    assert ov["healthy_gpus"] == 96, ov["top_problem_nodes"]
    assert ov["active_incidents"] == 0


def test_thermal_injection_creates_evidence_based_incident(client, admin, operator):
    client.cycles(20)
    r = client.post("/api/v1/demo/faults", headers=operator,
                    json={"type": "thermal", "node": "gpu-04", "gpu": 3, "severity": 0.9})
    assert r.status_code == 200, r.text
    client.cycles(12)

    incs = client.get("/api/v1/incidents?status=active", headers=admin).json()
    assert len(incs) == 1, incs
    inc = client.get(f"/api/v1/incidents/{incs[0]['id']}", headers=admin).json()
    assert inc["node"] == "gpu-04"
    assert 3 in inc["gpus"]
    assert inc["hypotheses"][0]["rule_id"] == "thermal_throttling"
    assert inc["confidence_label"] in ("medium", "high")
    assert inc["perf_deviation_pct"] < -8
    assert inc["observed"]["throughput"]["deviation_pct"] < -8
    metrics = {s["metric"] for s in inc["signals"]}
    assert {"temp_c", "sm_clock_mhz"} <= metrics
    # Peer comparison rows exist and show the abnormal GPU
    temp_row = next(r for r in inc["peer_comparison"] if r["metric"] == "temp_c")
    assert temp_row["status"] == "abnormal" and temp_row["value"] > temp_row["peer_median"]
    # Normal metrics are reported as normal (GPU utilization is not the problem)
    util = next(m for m in inc["observed"]["key_metrics"] if m["metric"] == "gpu_util")
    assert util["status"] == "normal"
    assert inc["recommended_actions"]
    assert inc["events"][0]["type"] == "opened"

    # AI explanation (deterministic, offline) separates observed / inferred / recommended
    exp = client.post(f"/api/v1/incidents/{inc['id']}/explain", headers=admin).json()
    assert exp["observed"] and exp["inferred"] and exp["recommended"]
    assert exp["inferred"][0]["cause"] == "Thermal throttling"
    assert "below its peer baseline" in exp["summary"]

    # Only one incident even though many metrics are abnormal (correlation)
    client.cycles(3)
    assert len(client.get("/api/v1/incidents?status=active", headers=admin).json()) == 1

    # GPU and node status reflect the problem
    node = client.get("/api/v1/nodes/gpu-04", headers=admin).json()
    assert node["status"] in ("warning", "critical")
    assert next(g for g in node["gpus"] if g["index"] == 3)["status"] != "healthy"
    assert node["peer_comparison"] and node["gpu_peer_summary"]


def test_incident_workflow_and_auto_resolve(client, admin, operator, viewer):
    client.cycles(15)
    client.post("/api/v1/demo/faults", headers=operator, json={"type": "cpu_bottleneck", "node": "gpu-06", "severity": 0.9})
    client.cycles(6)
    inc = client.get("/api/v1/incidents?status=active", headers=admin).json()[0]
    assert inc["top_hypothesis"] == "Host CPU / input-pipeline bottleneck"
    iid = inc["id"]
    # viewer cannot change incidents
    assert client.post(f"/api/v1/incidents/{iid}/status", headers=viewer, json={"status": "ACKNOWLEDGED"}).status_code == 403
    r = client.post(f"/api/v1/incidents/{iid}/status", headers=operator, json={"status": "ACKNOWLEDGED", "note": "on it"})
    assert r.status_code == 200 and r.json()["acknowledged_by"] == "operator"
    assert client.post(f"/api/v1/incidents/{iid}/comments", headers=operator, json={"text": "checking dataloader"}).status_code == 200
    # invalid transition
    assert client.post(f"/api/v1/incidents/{iid}/status", headers=operator, json={"status": "BOGUS"}).status_code == 409
    # clear the fault: incident auto-resolves after quiet cycles
    client.delete("/api/v1/demo/faults", headers=operator)
    client.cycles(12)
    inc = client.get(f"/api/v1/incidents/{iid}", headers=admin).json()
    assert inc["status"] == "RESOLVED"
    assert any(e["type"] == "auto_resolved" for e in inc["events"])
    assert any(e["type"] == "comment" for e in inc["events"])


def test_recurrence_is_tracked(client, admin, operator):
    client.cycles(15)
    for _ in range(2):
        client.post("/api/v1/demo/faults", headers=operator, json={"type": "power_cap", "node": "gpu-09", "gpu": 2})
        client.cycles(6)
        client.delete("/api/v1/demo/faults", headers=operator)
        client.cycles(10)
    incs = client.get("/api/v1/incidents?node=gpu-09", headers=admin).json()
    assert len(incs) == 2
    assert max(i["recurrence_count"] for i in incs) == 1


def test_rbac_and_api_keys(client, admin, viewer):
    client.cycles(2)
    assert client.post("/api/v1/demo/faults", headers=viewer, json={"type": "thermal", "node": "gpu-01"}).status_code == 403
    assert client.get("/api/v1/users", headers=viewer).status_code == 403
    assert client.get("/api/v1/audit-logs", headers=viewer).status_code == 403
    r = client.post("/api/v1/api-keys", headers=admin, json={"name": "grafana", "role": "viewer"})
    key = r.json()["key"]
    assert client.get("/api/v1/nodes", headers={"X-API-Key": key}).status_code == 200
    assert client.post("/api/v1/demo/faults", headers={"X-API-Key": key}, json={"type": "thermal", "node": "gpu-01"}).status_code == 403
    kid = r.json()["id"]
    client.delete(f"/api/v1/api-keys/{kid}", headers=admin)
    assert client.get("/api/v1/nodes", headers={"X-API-Key": key}).status_code == 401
    # user management
    r = client.post("/api/v1/users", headers=admin, json={"username": "alice", "password": "a-long-password", "role": "operator"})
    assert r.status_code == 201
    h = login(client, "alice", "a-long-password")
    assert client.get("/api/v1/auth/me", headers=h).json()["role"] == "operator"


def test_audit_log_records_actions(client, admin, operator):
    client.cycles(2)
    client.post("/api/v1/demo/faults", headers=operator, json={"type": "thermal", "node": "gpu-01"})
    logs = client.get("/api/v1/audit-logs", headers=admin).json()
    actions = [l["action"] for l in logs]
    assert "auth.login" in actions
    assert any(a == "POST /api/v1/demo/faults" and l["actor"] == "operator" for a, l in zip(actions, logs))


def test_fleet_endpoints(client, admin, operator):
    client.cycles(15)
    client.post("/api/v1/demo/scenarios/mixed-incidents", headers=operator)
    client.cycles(8)
    h = admin
    for path in ["/api/v1/clusters", "/api/v1/nodes", "/api/v1/gpus", "/api/v1/gpus/gpu-04/3", "/api/v1/peers?metric=temp_c",
                 "/api/v1/performance", "/api/v1/network", "/api/v1/workloads", "/api/v1/workloads/78421",
                 "/api/v1/anomalies", "/api/v1/rca/nodes", "/api/v1/rca/rules", "/api/v1/trends?minutes=60",
                 "/api/v1/nodes/gpu-04/history?metrics=throughput", "/api/v1/nodes/gpu-04/history?metrics=temp_c&gpu=3",
                 "/api/v1/alerts", "/api/v1/settings", "/api/v1/license", "/api/v1/system/info",
                 "/api/v1/metrics/catalog", "/api/v1/copilot/suggestions", "/api/v1/demo/faults", "/api/v1/demo/scenarios"]:
        r = client.get(path, headers=h)
        assert r.status_code == 200, (path, r.text)
    ov = client.get("/api/v1/overview", headers=h).json()
    assert ov["active_incidents"] == 3
    assert {p["node"] for p in ov["top_problem_nodes"]} >= {"gpu-04", "gpu-10", "gpu-02"}
    peers = client.get("/api/v1/peers?metric=temp_c", headers=h).json()
    assert peers["rows"][-1]["entity"] == "gpu-04/gpu3"  # hottest GPU sorts last
    alerts = client.get("/api/v1/alerts", headers=h).json()
    assert any(a["event"] == "opened" for a in alerts)
    job = client.get("/api/v1/workloads/78421", headers=h).json()
    assert {s["node"] for s in job["stragglers"]} >= {"gpu-04", "gpu-02"}
    rca = client.get("/api/v1/rca/nodes", headers=h).json()
    tops = {r["node"]: r["hypotheses"][0]["rule_id"] for r in rca if r["hypotheses"]}
    assert tops["gpu-04"] == "thermal_throttling"
    assert tops["gpu-10"] == "power_throttling"
    assert tops["gpu-02"] == "communication_bottleneck"
    assert client.get("/metrics").text.startswith("# TYPE")
