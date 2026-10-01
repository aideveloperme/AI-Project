"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { can, getUser, post, usePoll } from "@/lib/api";
import { ago, num, pct } from "@/lib/format";
import { Card, Confidence, IncidentStatus, Loading, PageHead, StatusBadge } from "@/components/ui";
import { DeviationBar } from "@/components/charts";
import { Hypotheses, SignalsTable } from "@/components/Hypotheses";

function Explanation({ id, cached }: { id: string; cached: any }) {
  const [exp, setExp] = useState<any>(cached);
  const [busy, setBusy] = useState(false);
  const load = async (force = false) => {
    setBusy(true);
    try { setExp(await post(`/incidents/${id}/explain?force=${force}`)); } finally { setBusy(false); }
  };
  // Use the engine-maintained explanation (regenerated when evidence changes); fetch only if missing.
  useEffect(() => { if (cached) setExp(cached); else load(); }, [id, cached?.generated_at]);  // eslint-disable-line react-hooks/exhaustive-deps
  if (!exp) return <Loading what="AI explanation" />;
  return (
    <Card title="AI incident explanation" actions={
      <div className="row"><span className="pill">{exp.provider === "template" ? "deterministic (offline)" : exp.provider}</span>
        <span className="pill" title="All numbers verified against the evidence">grounding: {exp.grounding?.method}</span>
        {exp.generated_at && <span className="pill">generated {ago(exp.generated_at)}</span>}
        <button onClick={() => load(true)} disabled={busy}>Regenerate</button></div>}>
      <p style={{ fontSize: 15, marginTop: 0 }}><b>{exp.summary}</b></p>
      <div className="grid g3">
        <div><div className="section-label obs">Observed (facts from telemetry)</div><ul className="tight">{exp.observed.map((o: string, i: number) => <li key={i}>{o}</li>)}</ul></div>
        <div><div className="section-label inf">Inferred (possible causes)</div><ul className="tight">{exp.inferred.map((c: any, i: number) => (
          <li key={i}><b>{c.cause}</b> <Confidence label={c.confidence} /><div className="ink2 small">{c.reasoning}</div></li>))}</ul></div>
        <div><div className="section-label rec">Recommended investigation</div><ol className="tight">{exp.recommended.map((r: string, i: number) => <li key={i}>{r}</li>)}</ol></div>
      </div>
      <div className="banner" style={{ marginTop: 12, marginBottom: 0 }}><b>Operator summary: </b>{exp.operator_explanation}</div>
      {exp.caveats?.length > 0 && <div className="muted small" style={{ marginTop: 8 }}>{exp.caveats.join(" ")}</div>}
    </Card>
  );
}

function IncidentDetail() {
  const id = useSearchParams().get("id") ?? "";
  const { data: d, error, reload } = usePoll<any>(id ? `/incidents/${id}` : null, 10000);
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const user = getUser();
  if (!d) return <Loading error={error} what={`incident ${id}`} />;
  const act = async (status: string) => {
    try { await post(`/incidents/${id}/status`, { status, note: note || undefined }); setNote(""); setMsg(null); reload(); }
    catch (e: any) { setMsg(e.message); }
  };
  const comment = async () => { if (!note) return; await post(`/incidents/${id}/comments`, { text: note }); setNote(""); reload(); };
  const thr = d.observed?.throughput;
  return (
    <>
      <PageHead title={`${d.id} · ${d.title}`} desc={`${d.node}${d.gpus?.length ? ` · GPU ${d.gpus.join(", ")}` : ""} · opened ${ago(d.created_at)} · last signal ${ago(d.last_signal_at)}`}>
        <StatusBadge status={d.severity} /><IncidentStatus status={d.status} />
      </PageHead>
      <div className="banner"><b>{d.summary}</b>{thr && <> — throughput {num(thr.value)} vs. peer median {num(thr.peer_median)} {thr.unit} ({pct(thr.deviation_pct)}, n={thr.n_peers} peers).</>}
        {d.recurrence_count > 0 && <span className="err"> Recurring: {d.recurrence_count} prior similar incident(s) on this node in 30 days.</span>}</div>

      {can(user, "operate") && (
        <Card title="Operator actions">
          <div className="row">
            <input style={{ flex: 1, minWidth: 240 }} placeholder="Note / resolution / comment" value={note} onChange={(e) => setNote(e.target.value)} />
            {d.status === "OPEN" && <button className="primary" onClick={() => act("ACKNOWLEDGED")}>Acknowledge</button>}
            {["OPEN", "ACKNOWLEDGED"].includes(d.status) && <button onClick={() => act("INVESTIGATING")}>Investigating</button>}
            {!["RESOLVED", "FALSE_POSITIVE"].includes(d.status) && <><button onClick={() => act("RESOLVED")}>Resolve</button><button onClick={() => act("FALSE_POSITIVE")}>False positive</button></>}
            {["RESOLVED", "FALSE_POSITIVE"].includes(d.status) && <button onClick={() => act("OPEN")}>Reopen</button>}
            <button onClick={comment} disabled={!note}>Comment</button>
          </div>
          {msg && <div className="err small">{msg}</div>}
          {d.acknowledged_by && <div className="muted small" style={{ marginTop: 6 }}>Acknowledged by {d.acknowledged_by} {ago(d.acknowledged_at)}</div>}
          {d.resolution && <div className="muted small">Resolution: {d.resolution}</div>}
        </Card>)}
      <div style={{ height: 14 }} />
      <Explanation id={id} cached={d.ai_explanation} />
      <div style={{ height: 14 }} />
      <Card title="Root-cause hypotheses (rule engine)"><Hypotheses items={d.hypotheses} /></Card>
      <div className="grid g2" style={{ margin: "14px 0" }}>
        <Card title="Peer comparison" flush>
          <div className="table-wrap"><table>
            <thead><tr><th>Scope</th><th>Metric</th><th className="num">Value</th><th className="num">Peer median</th><th className="num">p10–p90</th><th>Deviation</th><th>Status</th></tr></thead>
            <tbody>{d.peer_comparison.map((r: any, i: number) => (
              <tr key={i}><td>{r.scope}</td><td>{r.label}</td><td className="num">{num(r.value, 2)} {r.unit}</td><td className="num">{num(r.peer_median, 2)}</td>
                <td className="num muted">{num(r.p10)}–{num(r.p90)}</td><td><DeviationBar pct={r.deviation_pct} /></td>
                <td>{r.status === "abnormal" ? <StatusBadge status="warning" label="abnormal" /> : <StatusBadge status="good" label="normal" />}</td></tr>))}
            </tbody></table></div>
        </Card>
        <Card title="Timeline">
          <div className="timeline">{d.events.map((e: any, i: number) => (
            <div key={i}><div className="small muted">{new Date(e.ts).toLocaleString()} · {e.actor} · {e.type}</div><div>{e.message}</div></div>))}</div>
          {d.related_incidents?.length > 0 && <div className="small">Related: {d.related_incidents.map((r: string) => <Link key={r} href={`/incident/?id=${r}`} style={{ marginRight: 8 }}>{r}</Link>)}</div>}
        </Card>
      </div>
      <Card title={`Anomaly signals (${d.signals.length})`} flush><SignalsTable signals={d.signals} /></Card>
    </>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><IncidentDetail /></Suspense>;
}
