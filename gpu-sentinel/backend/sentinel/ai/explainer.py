"""Evidence-grounded incident explanations.

Pipeline:
  incident (deterministic analytics) → evidence packet (JSON) → [LLM] → validator → explanation

* The deterministic *template* explanation is always produced first and is a
  complete answer on its own (works fully offline, no model required).
* If a local LLM is configured, it may rephrase/prioritise the explanation,
  but only from the evidence packet. The output is validated:
    - must be valid JSON with the OBSERVED / INFERRED / RECOMMENDED structure
    - every number it mentions must exist in the evidence (±1.5 %)
    - inferred causes must be among the engine's hypotheses
  Otherwise the template explanation is returned and the rejection recorded.
"""
from __future__ import annotations

import json
import logging
import re

from sentinel.ai.providers import LLMProvider, LLMUnavailable

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are GPU Sentinel AI, an assistant for data-center GPU operators.
You explain incidents detected by a deterministic monitoring engine.
STRICT RULES:
- Use ONLY the facts in the EVIDENCE JSON. Never invent metrics, values, GPUs, nodes or events.
- Every number you write must appear in the evidence.
- Clearly separate OBSERVED facts, INFERRED possible causes, and RECOMMENDED actions.
- Never state a root cause as certain. Use the provided confidence labels ("possible", "likely").
- Only list causes that appear in evidence.hypotheses.
- Recommend investigation steps; never recommend destructive actions without a maintenance window.
Respond with a JSON object with keys:
 "summary" (string, 1-2 sentences),
 "observed" (array of strings),
 "inferred" (array of {"cause": string, "confidence": "high"|"medium"|"low", "reasoning": string}),
 "recommended" (array of strings, ordered),
 "operator_explanation" (string, plain language, 2-4 sentences),
 "caveats" (array of strings)."""

NUM_RE = re.compile(r"(?<![A-Za-z_\-/])[-+]?\d+(?:[.,]\d+)?")


def evidence_packet(inc: dict) -> dict:
    """The only data the LLM ever sees."""
    obs = inc.get("observed") or {}
    return {
        "incident_id": inc.get("id"),
        "node": inc.get("node"),
        "gpus": inc.get("gpus"),
        "gpu_model": obs.get("gpu_model"),
        "workload": obs.get("workload"),
        "severity": inc.get("severity"),
        "performance": obs.get("throughput"),
        "throttle_reasons": obs.get("throttle_reasons"),
        "abnormal_signals": [
            {k: s.get(k) for k in ("gpu_index", "label", "unit", "value", "expected", "deviation_pct", "direction", "methods")}
            for s in (inc.get("signals") or [])[:14]
        ],
        "normal_metrics": [m for m in obs.get("key_metrics", []) if m.get("status") == "normal"][:8],
        "hypotheses": [
            {k: h.get(k) for k in ("title", "confidence", "confidence_label", "evidence_for", "evidence_against", "explanation")}
            for h in (inc.get("hypotheses") or [])[:3]
        ],
        "recommended_actions": inc.get("recommended_actions"),
        "recurrence_count": inc.get("recurrence_count", 0),
    }


def _numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(x.replace(",", ".")) for x in NUM_RE.findall(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
        for k in obj.keys():
            out += _numbers(k) if isinstance(k, str) else []
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v)
    return out


def check_grounding(text_obj, evidence: dict) -> tuple[bool, list[float], int]:
    allowed = _numbers(evidence)
    allowed_abs = [abs(a) for a in allowed]
    found = _numbers(text_obj)
    bad = []
    for n in found:
        a = abs(n)
        if a <= 10 and float(a).is_integer():
            continue  # list numbering, GPU indices, small counts
        ok = any(abs(a - x) <= max(0.015 * x, 0.51) for x in allowed_abs)
        # allow rounded percentages / values
        ok = ok or any(abs(round(x) - a) < 1e-9 or abs(round(x, 1) - a) < 1e-9 for x in allowed_abs)
        if not ok:
            bad.append(n)
    return not bad, bad, len(found)


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return "n/a"
    s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}"
    return f"{s}{' ' if unit and unit not in ('%', '°C') else ''}{unit}"


def template_explanation(inc: dict) -> dict:
    obs = inc.get("observed") or {}
    node = inc.get("node")
    thr = obs.get("throughput") or {}
    hyps = inc.get("hypotheses") or []
    top = hyps[0] if hyps else None

    if thr.get("deviation_pct") is not None and thr["deviation_pct"] <= -3:
        lead = (f"{node} is performing {abs(thr['deviation_pct']):.1f}% below its peer baseline "
                f"({_fmt(thr.get('value'))} vs. peer median {_fmt(thr.get('peer_median'))} {thr.get('unit', '')}).")
    else:
        lead = f"{node} shows abnormal telemetry compared with its peers or its own recent history."
    if top:
        lead += f" Possible contributing factor: {top['title'].lower()} (confidence: {top['confidence_label']})."

    observed = []
    for s in (inc.get("signals") or [])[:10]:
        where = f"GPU {s['gpu_index']}" if s.get("gpu_index") is not None else "Node"
        if s.get("expected") is not None and s.get("deviation_pct") is not None:
            observed.append(f"{where}: {s['label']} is {_fmt(s['value'], s['unit'])}, "
                            f"{abs(s['deviation_pct']):.1f}% {'above' if s['direction'] == 'high' else 'below'} "
                            f"expected {_fmt(s['expected'], s['unit'])}.")
        else:
            observed.append(f"{where}: {s.get('description')}.")
    for m in obs.get("key_metrics", []):
        if m.get("status") == "normal":
            observed.append(f"{m['label']} is normal ({_fmt(m['value'], m['unit'])} vs. peer median {_fmt(m['peer_median'], m['unit'])}).")
    for r, gpus in (obs.get("throttle_reasons") or {}).items():
        observed.append(f"Throttle reason '{r}' reported on GPU(s) {', '.join(map(str, gpus))}.")

    inferred = [{"cause": h["title"], "confidence": h["confidence_label"], "reasoning": h["explanation"]} for h in hyps[:3]]
    caveats = ["Causes are inferred from correlated telemetry and are not confirmed until verified on the system."]
    if top and top["confidence_label"] != "high":
        caveats.append("Evidence is partial; confirm with the recommended checks before taking corrective action.")
    op = (f"{lead} " + (f"The engine matched the pattern '{top['title']}' because: "
                        + "; ".join(top["evidence_for"][:3]) + "." if top else
                        "No known failure pattern matched; treat as an open investigation."))
    return {
        "summary": lead,
        "observed": observed,
        "inferred": inferred,
        "recommended": list(inc.get("recommended_actions") or []),
        "operator_explanation": op,
        "caveats": caveats,
        "provider": "template",
        "grounding": {"passed": True, "ungrounded": [], "checked_numbers": 0, "method": "deterministic"},
    }


class Explainer:
    def __init__(self, provider: LLMProvider | None):
        self.provider = provider

    @property
    def provider_name(self) -> str:
        return f"{self.provider.name}:{self.provider.model}" if self.provider else "template"

    def explain(self, inc: dict, use_llm: bool = True) -> dict:
        base = template_explanation(inc)
        if not (use_llm and self.provider):
            return base
        ev = evidence_packet(inc)
        try:
            raw = self.provider.chat(SYSTEM_PROMPT, "EVIDENCE:\n" + json.dumps(ev, default=str))
            data = json.loads(raw)
        except (LLMUnavailable, json.JSONDecodeError, KeyError) as e:
            base["grounding"]["llm_error"] = str(e)[:300]
            return base
        problems = []
        for key in ("summary", "observed", "inferred", "recommended", "operator_explanation"):
            if key not in data:
                problems.append(f"missing key {key}")
        allowed_causes = {h["title"].lower() for h in inc.get("hypotheses") or []}
        for c in data.get("inferred") or []:
            cause = str(c.get("cause", "")).lower() if isinstance(c, dict) else str(c).lower()
            if allowed_causes and not any(a in cause or cause in a for a in allowed_causes):
                problems.append(f"cause not in hypotheses: {cause[:80]}")
        ok, bad, n = check_grounding({k: data.get(k) for k in ("summary", "observed", "inferred", "operator_explanation")}, ev)
        if not ok:
            problems.append(f"ungrounded numbers: {bad[:8]}")
        if problems:
            log.info("LLM explanation rejected for %s: %s", inc.get("id"), problems)
            base["grounding"] = {"passed": True, "method": "deterministic", "llm_rejected": problems,
                                 "ungrounded": bad, "checked_numbers": n}
            return base
        return {
            "summary": str(data["summary"]),
            "observed": [str(x) for x in data["observed"]],
            "inferred": data["inferred"],
            "recommended": [str(x) for x in data["recommended"]],
            "operator_explanation": str(data["operator_explanation"]),
            "caveats": [str(x) for x in data.get("caveats", [])] or base["caveats"],
            "provider": self.provider_name,
            "grounding": {"passed": True, "ungrounded": [], "checked_numbers": n, "method": "llm+validator"},
            "deterministic": base,
        }
