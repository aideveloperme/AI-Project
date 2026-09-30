import json

import pytest

from sentinel.ai.explainer import Explainer, check_grounding, evidence_packet, template_explanation
from sentinel.ai.providers import LLMProvider, LLMUnavailable, OpenAICompatibleProvider, build_provider, is_private_endpoint
from sentinel.config import Settings

INCIDENT = {
    "id": "INC-000001", "node": "gpu-42", "gpus": [3], "severity": "warning", "recurrence_count": 0,
    "observed": {
        "gpu_model": "NVIDIA H100 80GB HBM3", "workload": {"job_id": "78421", "name": "llama", "kind": "training"},
        "throughput": {"value": 1116.0, "peer_median": 1450.0, "deviation_pct": -23.0, "unit": "samples/s", "n_peers": 7},
        "throttle_reasons": {"sw_thermal_slowdown": [3]},
        "key_metrics": [
            {"metric": "gpu_util", "label": "GPU utilization", "unit": "%", "scope": "GPU 3", "value": 96.5,
             "peer_median": 97.1, "deviation_pct": -0.6, "status": "normal"},
            {"metric": "temp_c", "label": "GPU temperature", "unit": "°C", "scope": "GPU 3", "value": 83.0,
             "peer_median": 75.0, "deviation_pct": 10.7, "status": "abnormal"},
        ],
    },
    "signals": [
        {"gpu_index": 3, "label": "SM clock", "unit": "MHz", "value": 1420.0, "expected": 1650.0, "deviation_pct": -13.9,
         "direction": "low", "methods": ["peer_deviation"], "description": "SM clock low"},
        {"gpu_index": 3, "label": "GPU temperature", "unit": "°C", "value": 83.0, "expected": 75.0, "deviation_pct": 10.7,
         "direction": "high", "methods": ["peer_deviation"], "description": "temp high"},
    ],
    "hypotheses": [{"rule_id": "thermal_throttling", "title": "Thermal throttling", "confidence": 0.82,
                    "confidence_label": "high", "evidence_for": ["GPU 3: temp high", "throttle flag"],
                    "evidence_against": [], "explanation": "Consistent with thermal slowdown."}],
    "recommended_actions": ["Check cooling/airflow.", "Inspect power telemetry."],
}


class FakeLLM(LLMProvider):
    name, model = "fake", "fake-1"

    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = []

    def chat(self, system, user, json_mode=True):
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_template_separates_observed_inferred_recommended():
    exp = template_explanation(INCIDENT)
    assert exp["summary"].startswith("gpu-42 is performing 23.0% below its peer baseline")
    assert any("GPU utilization is normal" in o for o in exp["observed"])
    assert any("SM clock is 1,420 MHz, 13.9% below expected 1,650 MHz" in o for o in exp["observed"])
    assert exp["inferred"][0] == {"cause": "Thermal throttling", "confidence": "high",
                                  "reasoning": "Consistent with thermal slowdown."}
    assert exp["recommended"] == INCIDENT["recommended_actions"]
    assert exp["caveats"]
    # template output is itself grounded
    ok, bad, _ = check_grounding({k: exp[k] for k in ("summary", "observed")}, evidence_packet(INCIDENT))
    assert ok, bad


def test_llm_output_accepted_when_grounded():
    good = {"summary": "gpu-42 is 23% below peers; possible thermal throttling.",
            "observed": ["GPU 3 temperature 83°C vs peer 75°C", "SM clock 1420 MHz vs 1650 MHz"],
            "inferred": [{"cause": "Thermal throttling", "confidence": "high", "reasoning": "hot and slow"}],
            "recommended": ["Check cooling"], "operator_explanation": "GPU 3 runs hot and slows down.", "caveats": []}
    llm = FakeLLM(json.dumps(good))
    exp = Explainer(llm).explain(INCIDENT)
    assert exp["provider"] == "fake:fake-1"
    assert exp["grounding"]["method"] == "llm+validator"
    assert "EVIDENCE" in llm.calls[0][1] and "Never invent" in llm.calls[0][0]


def test_llm_fabricated_numbers_rejected():
    bad = {"summary": "gpu-42 is 41% slower; fan speed 3200 RPM.", "observed": ["Fan 3200 RPM"],
           "inferred": [{"cause": "Thermal throttling", "confidence": "high", "reasoning": ""}],
           "recommended": ["x"], "operator_explanation": "x"}
    exp = Explainer(FakeLLM(json.dumps(bad))).explain(INCIDENT)
    assert exp["provider"] == "template"
    assert any("ungrounded" in p for p in exp["grounding"]["llm_rejected"])
    assert 3200.0 in exp["grounding"]["ungrounded"]


def test_llm_invented_cause_rejected():
    bad = {"summary": "Possible cosmic rays.", "observed": [], "recommended": [], "operator_explanation": "x",
           "inferred": [{"cause": "Cosmic ray bit flips", "confidence": "high", "reasoning": ""}]}
    exp = Explainer(FakeLLM(json.dumps(bad))).explain(INCIDENT)
    assert exp["provider"] == "template"
    assert any("cause not in hypotheses" in p for p in exp["grounding"]["llm_rejected"])


def test_llm_failure_falls_back():
    exp = Explainer(FakeLLM(LLMUnavailable("down"))).explain(INCIDENT)
    assert exp["provider"] == "template" and "llm_error" in exp["grounding"]
    exp = Explainer(FakeLLM("not json")).explain(INCIDENT)
    assert exp["provider"] == "template"


def test_cloud_llm_blocked_by_default():
    assert is_private_endpoint("http://localhost:11434")
    assert is_private_endpoint("http://ollama:11434")
    assert is_private_endpoint("http://10.0.0.5:8000")
    with pytest.raises(LLMUnavailable):
        OpenAICompatibleProvider("https://8.8.8.8", "m", None, 5, allow_cloud=False)
    p = OpenAICompatibleProvider("https://8.8.8.8", "m", "k", 5, allow_cloud=True)
    assert p.local is False
    assert build_provider(Settings(llm_provider="openai_compatible", llm_url="https://8.8.8.8")) is None
    assert build_provider(Settings(llm_provider="template")) is None
