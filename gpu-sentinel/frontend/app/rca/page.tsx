"use client";
import Link from "next/link";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { pct } from "@/lib/format";
import { Card, Empty, Loading, PageHead, StatusBadge } from "@/components/ui";
import { Hypotheses } from "@/components/Hypotheses";

export default function RCA() {
  const { data, error } = usePoll<any[]>("/rca/nodes", 10000);
  const { data: rules } = usePoll<any[]>("/rca/rules", 0);
  const [tab, setTab] = useState<"live" | "rules">("live");
  return (
    <>
      <PageHead title="Root Cause Analysis" desc="Multi-metric correlation: many signals on one node become a few ranked, evidence-backed hypotheses. Deterministic rules run before any AI." />
      <div className="tabs"><button className={tab === "live" ? "on" : ""} onClick={() => setTab("live")}>Live diagnoses</button>
        <button className={tab === "rules" ? "on" : ""} onClick={() => setTab("rules")}>Rule catalogue ({rules?.length ?? 0})</button></div>
      {tab === "live" && (!data ? <Loading error={error} /> : data.length === 0 ? <Card><Empty>No abnormal nodes — nothing to diagnose.</Empty></Card> : (
        <div className="grid" style={{ gap: 14 }}>{data.map((n) => (
          <Card key={n.node} title={<span><Link href={`/node/?id=${n.node}`}>{n.node}</Link> · throughput {pct(n.perf_deviation_pct)} vs peers · {n.signal_count} signals</span>}
            actions={<StatusBadge status={n.severity} />}>
            <div className="row small" style={{ marginBottom: 10 }}>
              Signals by category: {Object.entries(n.signals_by_category).map(([c, k]) => <span key={c} className="pill">{c}: {k as number}</span>)}
              {Object.keys(n.throttle).length > 0 && <>· Throttle flags: {Object.entries(n.throttle).map(([r, g]: any) => <span key={r} className="pill">{r} (GPU {g.join(",")})</span>)}</>}
            </div>
            <Hypotheses items={n.hypotheses} />
          </Card>))}</div>))}
      {tab === "rules" && rules && (
        <div className="grid g2">{rules.map((r) => (
          <Card key={r.id} title={r.title} actions={<span className="pill">{r.category}</span>}>
            <p className="ink2" style={{ marginTop: 0 }}>{r.explanation}</p>
            <div className="small"><b>IF</b> {r.required.join(" AND ")}</div>
            {r.supporting.length > 0 && <div className="small"><b>Supported by:</b> {r.supporting.join("; ")}</div>}
            {r.contradicting.length > 0 && <div className="small"><b>Weakened by:</b> {r.contradicting.join("; ")}</div>}
            <div className="small muted">Confidence range {r.base_confidence}–{r.max_confidence}</div>
          </Card>))}</div>)}
    </>
  );
}
