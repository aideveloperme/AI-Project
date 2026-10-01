import pytest

from sentinel.assistant.service import parse_window_minutes


@pytest.fixture
def degraded(client, operator):
    client.cycles(15)
    client.post("/api/v1/demo/scenarios/mixed-incidents", headers=operator)
    client.post("/api/v1/demo/faults", headers=operator, json={"type": "ecc_errors", "node": "gpu-07", "gpu": 5, "severity": 0.9})
    client.cycles(8)
    return client


def ask(client, headers, q):
    r = client.post("/api/v1/ask", headers=headers, json={"question": q})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("question,intent", [
    ("Why is GPU-04 slow?", "explain_node"),
    ("Why is gpu-004 slow?", "explain_node"),
    ("Which GPUs are abnormal?", "list_abnormal_gpus"),
    ("Show me nodes with thermal issues.", "find_by_category"),
    ("Why did training job 78421 slow down?", "explain_job"),
    ("Compare node 4 with healthy nodes.", "compare_node"),
    ("What changed during the last two hours?", "what_changed"),
    ("Which nodes have repeated ECC errors?", "repeated_ecc"),
    ("Show me GPUs with abnormal clocks.", "find_by_category"),
    ("Show open incidents", "list_incidents"),
    ("How is the cluster?", "fleet_summary"),
])
def test_routing(degraded, admin, question, intent):
    res = ask(degraded, admin, question)
    assert res["intent"] == intent
    assert res["answer"]
    assert res["tools"][0]["name"] == intent


def test_why_slow_answer_is_evidence_based(degraded, admin):
    res = ask(degraded, admin, "Why is GPU-04 slow?")
    a = res["answer"]
    assert "below its peer baseline" in a
    assert "OBSERVED" in a and "INFERRED" in a and "RECOMMENDED" in a
    assert "Thermal throttling" in a
    assert res["data"]["incident_id"]


def test_thermal_and_clock_queries(degraded, admin):
    th = ask(degraded, admin, "Show me nodes with thermal issues.")
    assert "gpu-04" in th["answer"] and "gpu-10" not in {g["node"] for g in th["data"]["gpus"]}
    cl = ask(degraded, admin, "Show me GPUs with abnormal clocks.")
    nodes = {g["node"] for g in cl["data"]["gpus"]}
    assert {"gpu-04", "gpu-10"} <= nodes


def test_job_and_compare(degraded, admin):
    job = ask(degraded, admin, "Why did training job 78421 slow down?")
    assert {"gpu-04", "gpu-02"} <= {s["node"] for s in job["data"]["stragglers"]}
    cmp_ = ask(degraded, admin, "Compare node 4 with healthy nodes.")
    assert "| Metric |" in cmp_["answer"]
    assert any(r["deviation_pct"] < -8 for r in cmp_["data"]["rows"])


def test_ecc_and_unknown(degraded, admin):
    ecc = ask(degraded, admin, "Which nodes have repeated ECC errors?")
    assert "gpu-07" in ecc["answer"]
    unk = ask(degraded, admin, "Why is node 99 slow?")
    assert unk["intent"] in ("list_abnormal_gpus", "fleet_summary", "explain_node")


def test_window_parser():
    assert parse_window_minutes("what changed during the last two hours") == 120
    assert parse_window_minutes("in the past 30 minutes") == 30
    assert parse_window_minutes("last hour") == 60
    assert parse_window_minutes("last day") == 1440
    assert parse_window_minutes("recently", 45) == 45
