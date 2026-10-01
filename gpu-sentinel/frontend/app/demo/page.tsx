"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { can, del, getUser, post, usePoll } from "@/lib/api";
import { FAULT_LABELS } from "@/lib/format";
import { Card, Empty, Loading, PageHead } from "@/components/ui";

export default function Demo() {
  const user = getUser();
  const { data: types } = usePoll<any[]>("/demo/fault-types", 0);
  const { data: scen } = usePoll<any>("/demo/scenarios", 0);
  const { data: faults, reload } = usePoll<any[]>("/demo/faults", 5000);
  const { data: nodes } = usePoll<any[]>("/nodes", 0);
  const [f, setF] = useState({ type: "thermal", node: "gpu-04", gpu: "3", severity: 0.85, duration_s: 900 });
  const [msg, setMsg] = useState<string | null>(null);
  const gpuScoped = types?.find((t) => t.type === f.type)?.gpu_scoped;
  useEffect(() => { if (!gpuScoped) setF((x) => ({ ...x, gpu: "" })); }, [gpuScoped]);
  const allowed = can(user, "operate");
  const inject = async () => {
    try {
      await post("/demo/faults", { ...f, gpu: f.gpu === "" ? null : +f.gpu });
      setMsg(`Injected ${FAULT_LABELS[f.type]} into ${f.node}${f.gpu !== "" ? ` GPU ${f.gpu}` : ""}. Detection typically takes 2–6 analysis cycles.`);
      reload();
    } catch (e: any) { setMsg(e.message); }
  };
  return (
    <>
      <PageHead title="Demo Control" desc="Telemetry simulator for trade-show demos: inject realistic faults into a virtual GPU cluster and watch GPU Sentinel detect, correlate, diagnose and explain them." />
      {!allowed && <div className="banner warn">Your role is read-only. Sign in as operator or admin to inject faults.</div>}
      <div className="grid g2">
        <Card title="Inject a fault">
          <div className="grid g2">
            <label className="field">Fault type<select value={f.type} onChange={(e) => setF({ ...f, type: e.target.value })}>
              {types?.map((t) => <option key={t.type} value={t.type}>{FAULT_LABELS[t.type] ?? t.type}</option>)}</select></label>
            <label className="field">Node<select value={f.node} onChange={(e) => setF({ ...f, node: e.target.value })}>
              {nodes?.map((n) => <option key={n.node} value={n.node}>{n.node}</option>)}</select></label>
            <label className="field">GPU {gpuScoped ? "" : "(node-level fault)"}<select value={f.gpu} disabled={!gpuScoped} onChange={(e) => setF({ ...f, gpu: e.target.value })}>
              <option value="">All GPUs</option>{Array.from({ length: 8 }, (_, i) => <option key={i} value={i}>GPU {i}</option>)}</select></label>
            <label className="field">Severity: {f.severity.toFixed(2)}<input type="range" min={0.1} max={1} step={0.05} value={f.severity} onChange={(e) => setF({ ...f, severity: +e.target.value })} /></label>
            <label className="field">Duration<select value={f.duration_s} onChange={(e) => setF({ ...f, duration_s: +e.target.value })}>
              <option value={300}>5 min</option><option value={900}>15 min</option><option value={3600}>1 hour</option></select></label>
          </div>
          <p className="muted small">{types?.find((t) => t.type === f.type)?.description}</p>
          <button className="primary" disabled={!allowed} onClick={inject}>Inject fault</button>
          {msg && <div className="small" style={{ marginTop: 8 }}>{msg} <Link href="/incidents/">Watch incidents →</Link></div>}
        </Card>
        <Card title="Scenarios (one click)">
          <div className="grid" style={{ gap: 8 }}>
            {scen && Object.entries(scen).map(([name, items]: any) => (
              <div key={name} className="row" style={{ justifyContent: "space-between" }}>
                <div><b>{name}</b><div className="muted small">{items.map((i: any) => `${FAULT_LABELS[i.type]} → ${i.node}${i.gpu !== null ? `/GPU ${i.gpu}` : ""}`).join(" · ")}</div></div>
                <button disabled={!allowed} onClick={async () => { await post(`/demo/scenarios/${name}`); reload(); setMsg(`Scenario ${name} started.`); }}>Run</button>
              </div>))}
          </div>
        </Card>
      </div>
      <div style={{ height: 14 }} />
      <Card title="Active faults" flush actions={<button className="danger" disabled={!allowed || !faults?.length} onClick={async () => { await del("/demo/faults"); reload(); }}>Clear all (simulate repair)</button>}>
        {!faults ? <Loading /> : faults.length === 0 ? <Empty>No active faults — the cluster is healthy.</Empty> : (
          <div className="table-wrap"><table><thead><tr><th>Fault</th><th>Target</th><th className="num">Severity</th><th>Remaining</th><th></th></tr></thead>
            <tbody>{faults.map((x) => (
              <tr key={x.id}><td>{FAULT_LABELS[x.type]}</td><td><Link href={`/node/?id=${x.node}`}>{x.node}</Link>{x.gpu !== null ? ` / GPU ${x.gpu}` : ""}</td>
                <td className="num">{x.severity.toFixed(2)}</td>
                <td>{x.duration_s ? `${Math.max(0, Math.round((x.started_at + x.duration_s - Date.now() / 1000) / 60))} min` : "∞"}</td>
                <td><button disabled={!allowed} onClick={async () => { await del(`/demo/faults?fault_id=${x.id}`); reload(); }}>Clear</button></td></tr>))}
            </tbody></table></div>)}
      </Card>
    </>
  );
}
