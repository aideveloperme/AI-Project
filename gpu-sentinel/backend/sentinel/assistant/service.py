"""Ask Sentinel: natural-language questions → controlled, read-only tools.

Security model: the assistant can only call functions in :data:`TOOLS`. There is
no shell, SQL or HTTP passthrough. Intent routing is deterministic first; an
LLM (if configured) may pick a tool from the allow-list when the router can't,
and may rephrase the answer — but the rephrased answer is number-grounded
against the tool output, otherwise the deterministic answer is returned.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import timedelta

import numpy as np
from sqlalchemy import select

from sentinel.ai.explainer import check_grounding
from sentinel.ai.providers import LLMProvider, LLMUnavailable
from sentinel.db import AnomalyRow, Incident
from sentinel.incidents.service import ACTIVE_STATUSES
from sentinel.telemetry.catalog import CATALOG
from sentinel.telemetry.models import utcnow

log = logging.getLogger(__name__)

CATEGORY_WORDS = {
    "thermal": ["thermal", "temperature", "hot", "overheat", "heat", "cooling"],
    "clocks": ["clock", "clocks", "frequency", "mhz"],
    "ecc": ["ecc", "memory error", "xid", "dbe", "sbe", "retired"],
    "network": ["network", "infiniband", " ib ", "fabric", "latency", "packet"],
    "communication": ["nccl", "all-reduce", "allreduce", "collective", "communication"],
    "power": ["power", "watt", "power cap"],
    "memory": ["hbm", "memory bandwidth", "bandwidth"],
    "nvlink": ["nvlink"],
    "pcie": ["pcie", "pci-e"],
    "cpu": ["cpu", "data loader", "dataloader"],
}
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "twelve": 12, "a": 1, "an": 1}


def parse_window_minutes(q: str, default: int = 60) -> int:
    m = re.search(r"(?:last|past|previous)\s+(\d+|one|two|three|four|five|six|twelve|a|an)?\s*(minute|min|hour|hr|day)s?", q)
    if not m:
        return default
    n = m.group(1)
    num = WORDNUM.get(n, None) if n and not n.isdigit() else (int(n) if n else 1)
    unit = m.group(2)
    return int((num or 1) * (1 if unit.startswith("min") else 60 if unit in ("hour", "hr") else 1440))


def _basis(s) -> str:
    if "peer_deviation" in s.methods:
        return "vs peers"
    if s.method == "static_threshold":
        return "vs threshold"
    return "vs own baseline"


class Assistant:
    def __init__(self, engine, provider: LLMProvider | None = None):
        self.engine = engine
        self.provider = provider
        self.tools: dict[str, Callable[..., dict]] = {
            "fleet_summary": self.fleet_summary,
            "list_abnormal_gpus": self.list_abnormal_gpus,
            "explain_node": self.explain_node,
            "compare_node": self.compare_node,
            "find_by_category": self.find_by_category,
            "explain_job": self.explain_job,
            "what_changed": self.what_changed,
            "list_incidents": self.list_incidents,
            "repeated_ecc": self.repeated_ecc,
        }

    # -------------------------------------------------------------- routing
    def resolve_node(self, q: str) -> str | None:
        snap = self.engine.state.snapshot
        if snap is None:
            return None
        names = [n.node for n in snap.nodes]
        for n in names:
            if n.lower() in q:
                return n
        m = re.search(r"\b(?:gpu|node|host|server)[\s\-_#]*0*(\d{1,4})\b", q)
        if m:
            num = int(m.group(1))
            for n in names:
                d = re.findall(r"\d+", n)
                if d and int(d[-1]) == num:
                    return n
        return None

    def resolve_job(self, q: str) -> str | None:
        snap = self.engine.state.snapshot
        if snap is None:
            return None
        for w in snap.workloads:
            if w.job_id.lower() in q or w.name.lower() in q:
                return w.job_id
        m = re.search(r"\bjob\s*#?\s*(\w[\w\-]*)", q)
        return m.group(1) if m else None

    def route(self, question: str) -> tuple[str, dict]:
        q = " " + question.lower().strip() + " "
        node = self.resolve_node(q)
        if re.search(r"\bjob\b|training run|\bslurm\b", q):
            return "explain_job", {"job_id": self.resolve_job(q)}
        if re.search(r"what changed|what has changed|changes? (in|during|over)|recent changes", q):
            return "what_changed", {"minutes": parse_window_minutes(q, 60)}
        if "repeated" in q or "recurring" in q or "repeat" in q:
            if any(w in q for w in CATEGORY_WORDS["ecc"]):
                return "repeated_ecc", {"days": 30}
        if node and re.search(r"compare|versus|\bvs\b|healthy nodes|peers?", q):
            return "compare_node", {"node": node}
        if node:
            return "explain_node", {"node": node}
        if "incident" in q:
            status = "active" if re.search(r"open|active|current", q) else "all"
            return "list_incidents", {"status": status}
        for cat, words in CATEGORY_WORDS.items():
            if any(w in q for w in words):
                if cat == "ecc" and ("repeat" in q or "history" in q):
                    return "repeated_ecc", {"days": 30}
                return "find_by_category", {"category": cat}
        if re.search(r"abnormal|anomal|unhealthy|problem|issue|wrong|slow|degrad", q):
            return "list_abnormal_gpus", {}
        if re.search(r"overview|summary|status|health|how is|how are", q):
            return "fleet_summary", {}
        return "", {}

    def ask(self, question: str, use_llm: bool = True) -> dict:
        if self.engine.state.snapshot is None:
            return {"answer": "No telemetry has been collected yet. Try again in a few seconds.", "intent": None,
                    "tools": [], "data": {}, "provider": "template"}
        tool, args = self.route(question)
        via = "router"
        if not tool and use_llm and self.provider:
            tool, args = self._llm_route(question)
            via = "llm-router"
        if not tool:
            tool, args = "fleet_summary", {}
            via = "fallback"
        result = self.tools[tool](**args)
        answer = result.pop("answer")
        provider = "template"
        if use_llm and self.provider:
            better = self._llm_phrase(question, answer, result)
            if better:
                answer, provider = better, f"{self.provider.name}:{self.provider.model}"
        return {"answer": answer, "intent": tool, "routing": via, "tools": [{"name": tool, "args": args}],
                "data": result, "provider": provider}

    def _llm_route(self, question: str) -> tuple[str, dict]:
        sys = ("Select ONE tool for the operator question. Tools: " + ", ".join(self.tools) +
               ". Args: explain_node/compare_node {node}, find_by_category {category}, explain_job {job_id}, "
               "what_changed {minutes}, list_incidents {status}. Reply JSON {\"tool\":..., \"args\":{...}}.")
        try:
            data = json.loads(self.provider.chat(sys, question))
            tool = data.get("tool")
            if tool in self.tools:
                args = {k: v for k, v in (data.get("args") or {}).items() if isinstance(v, (str, int))}
                import inspect
                params = set(inspect.signature(self.tools[tool]).parameters)
                return tool, {k: v for k, v in args.items() if k in params}
        except (LLMUnavailable, json.JSONDecodeError, AttributeError, TypeError):
            pass
        return "", {}

    def _llm_phrase(self, question: str, draft: str, data: dict) -> str | None:
        sys = ("You are Ask Sentinel, GPU Sentinel's operator assistant. Rewrite the DRAFT answer to the QUESTION for a data-center "
               "operator. Use only facts and numbers present in DATA or DRAFT. Keep OBSERVED facts, INFERRED causes "
               "(with confidence) and RECOMMENDED steps clearly separated. Never claim certainty. "
               "Reply JSON {\"answer\": markdown string}.")
        try:
            out = json.loads(self.provider.chat(sys, json.dumps({"QUESTION": question, "DRAFT": draft,
                                                                  "DATA": data}, default=str)[:12000]))
            ans = out.get("answer")
            if not isinstance(ans, str) or not ans.strip():
                return None
            ok, bad, _ = check_grounding(ans, {"d": data, "draft": draft})
            return ans if ok else None
        except (LLMUnavailable, json.JSONDecodeError, AttributeError):
            return None

    # ---------------------------------------------------------------- tools
    def _gpu_rows(self, filt=None) -> list[dict]:
        st = self.engine.state
        rows: dict[str, dict] = {}
        for s in st.signals:
            if s.gpu_index is None or (filt and not filt(s)):
                continue
            r = rows.setdefault(s.entity, {"gpu": s.entity, "node": s.node, "index": s.gpu_index,
                                           "status": st.gpu_status.get(s.entity, "warning"), "signals": []})
            r["signals"].append({"metric": s.metric, "label": s.label, "value": round(s.value, 1), "unit": s.unit,
                                 "deviation_pct": round(s.deviation_pct, 1) if s.deviation_pct is not None else None,
                                 "basis": _basis(s), "severity": s.severity})
        return sorted(rows.values(), key=lambda r: (r["status"] != "critical", -len(r["signals"])))

    def fleet_summary(self) -> dict:
        st = self.engine.state
        snap = st.snapshot
        gpus = snap.all_gpus()
        counts = {k: sum(1 for v in st.gpu_status.values() if v == k) for k in ("healthy", "warning", "critical")}
        with self.engine.db.session() as db:
            active = db.scalars(select(Incident).where(Incident.status.in_(ACTIVE_STATUSES))).all()
        util = np.mean([g.metrics.get("gpu_util", 0) for g in gpus])
        temp = np.mean([g.metrics.get("temp_c", 0) for g in gpus])
        lines = [f"**Fleet status** — {len(snap.nodes)} nodes, {len(gpus)} GPUs.",
                 f"- Healthy: {counts['healthy']}, warning: {counts['warning']}, critical: {counts['critical']}",
                 f"- Average GPU utilization {util:.1f}%, average temperature {temp:.1f}°C",
                 f"- Active incidents: {len(active)}"]
        for i in active[:5]:
            lines.append(f"  - {i.id} [{i.severity}] {i.title}")
        return {"answer": "\n".join(lines), "counts": counts, "active_incidents": [i.id for i in active],
                "avg_util": round(float(util), 1), "avg_temp": round(float(temp), 1)}

    def list_abnormal_gpus(self) -> dict:
        rows = self._gpu_rows()
        abnormal_nodes = [n for n, s in self.engine.state.node_status.items() if s != "healthy"]
        if not rows and not abnormal_nodes:
            return {"answer": "**OBSERVED:** No GPUs are currently abnormal; all devices are within their peer baselines.",
                    "gpus": [], "nodes": []}
        lines = [f"**OBSERVED:** {len(rows)} GPU(s) on {len(abnormal_nodes)} node(s) deviate from their peer baseline:"]
        for r in rows[:15]:
            sig = "; ".join(f"{x['label']} {x['value']}{x['unit']}" + (f" ({x['deviation_pct']:+.1f}% {x['basis']})" if x['deviation_pct'] is not None else "")
                            for x in r["signals"][:3])
            lines.append(f"- **{r['node']} GPU {r['index']}** [{r['status']}]: {sig}")
        node_only = [n for n in abnormal_nodes if not any(r["node"] == n for r in rows)]
        if node_only:
            lines.append(f"- Node-level anomalies (no GPU-level signal): {', '.join(node_only)}")
        lines.append("\n**RECOMMENDED:** open the node pages for details, or ask *\"Why is <node> slow?\"*.")
        return {"answer": "\n".join(lines), "gpus": rows[:30], "nodes": abnormal_nodes}

    def find_by_category(self, category: str) -> dict:
        cats = {category}
        if category == "memory":
            cats = {"memory"}
        rows = self._gpu_rows(lambda s: s.category in cats)
        node_sigs = [s for s in self.engine.state.signals if s.gpu_index is None and s.category in cats]
        if category == "communication":
            node_sigs += [s for s in self.engine.state.signals if s.gpu_index is None and s.metric.startswith("nccl")]
        if not rows and not node_sigs:
            return {"answer": f"**OBSERVED:** No {category} anomalies are active right now across the fleet.",
                    "gpus": [], "node_signals": []}
        lines = [f"**OBSERVED — {category} anomalies:**"]
        for r in rows[:15]:
            sig = "; ".join(f"{x['label']} {x['value']}{x['unit']}" + (f" ({x['deviation_pct']:+.1f}% {x['basis']})" if x['deviation_pct'] is not None else "")
                            for x in r["signals"][:3])
            lines.append(f"- {r['node']} GPU {r['index']}: {sig}")
        for s in node_sigs[:10]:
            lines.append(f"- {s.node} (node): {s.description}")
        results = self.engine.state.results
        inferred = {n: h[1][0] for n, h in results.items() if h[1] and h[1][0].category == category}
        if inferred:
            lines.append("\n**INFERRED:** " + "; ".join(f"{n}: possible {h.title.lower()} ({h.confidence_label} confidence)"
                                                        for n, h in inferred.items()))
        return {"answer": "\n".join(lines), "gpus": rows,
                "node_signals": [s.to_dict() for s in node_sigs[:20]]}

    def explain_node(self, node: str) -> dict:
        st = self.engine.state
        res = st.results.get(node)
        if res is None:
            return {"answer": f"I have no telemetry for node **{node}**.", "node": node}
        ev, hyps, sev = res
        with self.engine.db.session() as db:
            inc = db.scalars(select(Incident).where(Incident.node == node, Incident.status.in_(ACTIVE_STATUSES))
                             .order_by(Incident.created_at.desc())).first()
            inc_id = inc.id if inc else None
        if not ev.signals:
            thr = ev.perf
            extra = (f" Throughput is {thr.value:.0f} vs. peer median {thr.peer.median:.0f} ({thr.deviation_pct:+.1f}%)."
                     if thr else "")
            return {"answer": f"**OBSERVED:** {node} is within its expected range (peers and its own history) — no abnormal signals.{extra}",
                    "node": node, "status": "healthy"}
        # Headline is always built from the live diagnosis so it can't disagree with the list below.
        lines = []
        if ev.perf is not None and ev.perf.deviation_pct <= -3:
            head = (f"{node} is performing {abs(ev.perf.deviation_pct):.1f}% below its peer baseline "
                    f"({ev.perf.value:.0f} vs. peer median {ev.perf.peer.median:.0f}).")
        else:
            basis = "its peers" if any("peer_deviation" in s.methods for s in ev.signals) else "its own recent history"
            head = f"{node} shows abnormal telemetry compared with {basis}."
        if hyps:
            head += f" Possible contributing factor: {hyps[0].title.lower()} (confidence {hyps[0].confidence_label})."
        lines.append(f"**{head}**")
        lines.append("\n**OBSERVED:**")
        for s in sorted(ev.signals, key=lambda s: -abs(s.zscore or 0))[:8]:
            where = f"GPU {s.gpu_index}" if s.gpu_index is not None else "node"
            lines.append(f"- {where}: {s.description}")
        if not any(s.metric == "gpu_util" for s in ev.signals):
            ns = self.engine.state.snapshot.node(node)
            utils = [g.metrics["gpu_util"] for g in (ns.gpus if ns else []) if "gpu_util" in g.metrics]
            if utils:
                has_peers = any("gpu_util" in st.peers.get(g.key, {}) for g in ns.gpus)
                lines.append(f"- GPU utilization is normal ({sum(utils) / len(utils):.0f}%"
                             f"{', in line with peers' if has_peers else ''}).")
        if hyps:
            lines.append("\n**INFERRED (possible contributing factors):**")
            for h in hyps[:3]:
                lines.append(f"- {h.title} — confidence **{h.confidence_label}** ({h.confidence:.2f})")
            lines.append("\n**RECOMMENDED investigation:**")
            for i, a in enumerate(hyps[0].recommended_actions, 1):
                lines.append(f"{i}. {a}")
        if inc_id:
            lines.append(f"\nTracked as incident **{inc_id}**.")
        return {"answer": "\n".join(lines), "node": node, "status": sev, "incident_id": inc_id,
                "signals": [s.to_dict() for s in ev.signals[:20]], "hypotheses": [h.to_dict() for h in hyps[:3]],
                "perf_deviation_pct": ev.perf_deviation_pct}

    def compare_node(self, node: str) -> dict:
        st = self.engine.state
        peers = st.peers
        snap = st.snapshot
        ns = snap.node(node)
        if ns is None:
            return {"answer": f"I have no telemetry for node **{node}**.", "node": node}
        rows = []
        for m in ("throughput", "cpu_util", "nccl_busbw_gbps", "ib_rx_gbps", "dataloader_wait_pct"):
            pc = peers.get(node, {}).get(m)
            if pc:
                rows.append((pc.label, "node", pc.value, pc.peer.median, pc.deviation_pct, pc.unit))
        for m in ("gpu_util", "sm_clock_mhz", "temp_c", "power_w", "hbm_bw_gbps", "perf_index"):
            vals, meds = [], []
            for g in ns.gpus:
                pc = peers.get(g.key, {}).get(m)
                if pc:
                    vals.append(pc.value)
                    meds.append(pc.peer.median)
            if vals:
                spec = CATALOG[("gpu", m)]
                v, md = float(np.mean(vals)), float(np.mean(meds))
                rows.append((spec.label + " (node avg)", "gpus", v, md, (v - md) / md * 100 if md else 0.0, spec.unit))
                worst = min(vals) if spec.direction == "low_bad" else max(vals)
                wi = vals.index(worst)
                rows.append((spec.label + f" (worst: GPU {ns.gpus[wi].index})", "gpu", worst, meds[wi],
                             (worst - meds[wi]) / meds[wi] * 100 if meds[wi] else 0.0, spec.unit))
        group = next(iter(peers.get(node, {}).values())).group if peers.get(node) else "n/a"
        lines = [f"**{node} vs. healthy peers** (peer group: {group})", "",
                 f"| Metric | {node} | Peer median | Deviation |", "|---|---:|---:|---:|"]
        for label, _, v, md, dev, unit in rows:
            flag = " ⚠" if abs(dev) >= 8 else ""
            lines.append(f"| {label} | {v:,.1f} {unit} | {md:,.1f} {unit} | {dev:+.1f}%{flag} |")
        return {"answer": "\n".join(lines), "node": node, "group": group,
                "rows": [{"metric": r[0], "value": round(r[2], 2), "peer_median": round(r[3], 2),
                          "deviation_pct": round(r[4], 2), "unit": r[5]} for r in rows]}

    def explain_job(self, job_id: str | None) -> dict:
        snap = self.engine.state.snapshot
        wl = snap.workload(job_id) if job_id else None
        if wl is None:
            jobs = ", ".join(f"{w.job_id} ({w.name}, {w.scheduler})" for w in snap.workloads)
            return {"answer": f"I couldn't find job **{job_id}**. Known workloads: {jobs}.", "job_id": job_id}
        from sentinel.analytics.workloads import job_analysis
        ja = job_analysis(self.engine, wl.job_id)
        lines = [f"**Job {wl.job_id} — {wl.name}** ({wl.scheduler}, {wl.kind}, {len(wl.nodes)} nodes)"]
        if ja["baseline_throughput"]:
            lines.append(f"- Aggregate throughput {ja['throughput']:,.0f} {wl.throughput_unit} vs. its own recent baseline "
                         f"{ja['baseline_throughput']:,.0f} ({ja['change_pct']:+.1f}%).")
        else:
            lines.append(f"- Aggregate throughput {ja['throughput']:,.0f} {wl.throughput_unit}.")
        if ja["stragglers"]:
            lines.append("\n**OBSERVED — slow nodes (vs. peers in the job):**")
            for s in ja["stragglers"]:
                lines.append(f"- {s['node']}: {s['throughput']:,.0f} ({s['deviation_pct']:+.1f}% vs. job median)"
                             + (f" — {s['top_hypothesis']} ({s['confidence']} confidence)" if s.get("top_hypothesis") else ""))
            if wl.kind == "training":
                lines.append("\n**INFERRED:** In synchronous data-parallel training every step waits for the slowest rank, "
                             "so these node(s) likely gate the whole job.")
            lines.append("\n**RECOMMENDED:** investigate the straggler node(s) first"
                         + (f" — see incident(s) {', '.join(ja['incidents'])}." if ja["incidents"] else "."))
        else:
            lines.append("- **OBSERVED:** no straggler nodes; all job nodes are within their peer baseline.")
            if ja["change_pct"] is not None and ja["change_pct"] <= -5:
                lines.append("- **INFERRED:** the slowdown is job-wide; check input data pipeline, storage and recent "
                             "configuration/software changes.")
        return {"answer": "\n".join(lines), **ja}

    def what_changed(self, minutes: int = 60) -> dict:
        eng = self.engine
        snap = eng.state.snapshot
        since = utcnow() - timedelta(minutes=minutes)
        changes = []
        entities = [(n.node, "node") for n in snap.nodes] + [(g.key, "gpu") for g in snap.all_gpus()]
        interval = max(1.0, eng.s.analysis_interval_s)
        back = int(minutes * 60 / interval)
        for ent, level in entities:
            for m in (("throughput", "cpu_util", "nccl_busbw_gbps", "ib_rx_gbps") if level == "node"
                      else ("temp_c", "sm_clock_mhz", "gpu_util", "power_w", "hbm_bw_gbps")):
                v = eng.history.values(ent, m)
                if v.size < 6:
                    continue
                old = float(v[max(0, v.size - back - 3): max(3, v.size - back)].mean())
                new = float(v[-3:].mean())
                if old and abs(new - old) / abs(old) >= 0.08:
                    spec = CATALOG[(level, m)]
                    changes.append({"entity": ent, "metric": m, "label": spec.label, "unit": spec.unit,
                                    "before": round(old, 1), "now": round(new, 1),
                                    "change_pct": round((new - old) / old * 100, 1)})
        changes.sort(key=lambda c: -abs(c["change_pct"]))
        with eng.db.session() as db:
            opened = db.scalars(select(Incident).where(Incident.created_at >= since)).all()
            resolved = db.scalars(select(Incident).where(Incident.resolved_at >= since)).all()
            new_anoms = db.scalars(select(AnomalyRow).where(AnomalyRow.first_seen >= since)).all()
        lines = [f"**What changed in the last {minutes} min**"]
        if not changes and not opened and not resolved:
            lines.append("- **OBSERVED:** No significant changes (all tracked metrics within ±8%).")
        for c in changes[:12]:
            lines.append(f"- {c['entity']}: {c['label']} {c['before']} → {c['now']} {c['unit']} ({c['change_pct']:+.1f}%)")
        if len(changes) > 12:
            lines.append(f"- … {len(changes) - 12} more changes")
        if opened:
            lines.append("\n**Incidents opened:** " + ", ".join(f"{i.id} ({i.title})" for i in opened[:8]))
        if resolved:
            lines.append("**Incidents resolved:** " + ", ".join(i.id for i in resolved[:8]))
        lines.append(f"\n{len(new_anoms)} new anomaly signal(s) first seen in this window.")
        return {"answer": "\n".join(lines), "minutes": minutes, "changes": changes[:50],
                "opened": [i.id for i in opened], "resolved": [i.id for i in resolved], "new_anomalies": len(new_anoms)}

    def list_incidents(self, status: str = "active") -> dict:
        with self.engine.db.session() as db:
            q = select(Incident).order_by(Incident.created_at.desc()).limit(20)
            if status == "active":
                q = q.where(Incident.status.in_(ACTIVE_STATUSES))
            incs = db.scalars(q).all()
        if not incs:
            return {"answer": "No incidents found.", "incidents": []}
        lines = [f"**{'Active' if status == 'active' else 'Recent'} incidents:**"]
        for i in incs:
            lines.append(f"- **{i.id}** [{i.severity}/{i.status}] {i.title} — {i.summary}")
        return {"answer": "\n".join(lines), "incidents": [i.id for i in incs]}

    def repeated_ecc(self, days: int = 30) -> dict:
        since = utcnow() - timedelta(days=days)
        with self.engine.db.session() as db:
            incs = db.scalars(select(Incident).where(Incident.category == "ecc", Incident.created_at >= since)).all()
        by_node: dict[str, list[str]] = {}
        for i in incs:
            by_node.setdefault(i.node, []).append(i.id)
        snap = self.engine.state.snapshot
        live = [{"gpu": g.key, "ecc_sbe_rate": g.metrics.get("ecc_sbe_rate", 0), "ecc_dbe_total": g.metrics.get("ecc_dbe_total", 0),
                 "retired_pages": g.metrics.get("retired_pages", 0)}
                for g in snap.all_gpus()
                if g.metrics.get("ecc_dbe_total", 0) > 0 or g.metrics.get("retired_pages", 0) > 0 or g.metrics.get("ecc_sbe_rate", 0) >= 5]
        lines = [f"**ECC history (last {days} days)**"]
        if by_node:
            for n, ids in sorted(by_node.items(), key=lambda x: -len(x[1])):
                tag = " — **repeated**" if len(ids) > 1 else ""
                lines.append(f"- {n}: {len(ids)} ECC incident(s) ({', '.join(ids[:6])}){tag}")
        else:
            lines.append("- No ECC incidents recorded in this period.")
        if live:
            lines.append("\n**Currently reporting ECC errors:**")
            for g in live[:10]:
                lines.append(f"- {g['gpu']}: SBE {g['ecc_sbe_rate']:.1f}/min, DBE {g['ecc_dbe_total']:.0f}, retired pages {g['retired_pages']:.0f}")
            lines.append("\n**RECOMMENDED:** drain nodes with uncorrectable errors and open an RMA if errors recur on the same GPU.")
        return {"answer": "\n".join(lines), "by_node": by_node, "live": live}
