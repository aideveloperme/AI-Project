"use client";
import Link from "next/link";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { ago, pct } from "@/lib/format";
import { Card, Confidence, Empty, IncidentStatus, Loading, PageHead, StatusBadge } from "@/components/ui";

export default function Incidents() {
  const [status, setStatus] = useState("active");
  const [sev, setSev] = useState("");
  const { data, error } = usePoll<any[]>(`/incidents?${status ? `status=${status}&` : ""}${sev ? `severity=${sev}` : ""}`, 10000);
  return (
    <>
      <PageHead title="Incidents" desc="Correlated, evidence-based incidents. One incident per affected node; signals, peer comparison and probable causes are preserved.">
        <div className="row">
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="active">Active</option><option value="">All</option><option value="OPEN">Open</option><option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="INVESTIGATING">Investigating</option><option value="RESOLVED">Resolved</option><option value="FALSE_POSITIVE">False positive</option>
          </select>
          <select value={sev} onChange={(e) => setSev(e.target.value)}><option value="">All severities</option><option value="critical">Critical</option><option value="warning">Warning</option></select>
        </div>
      </PageHead>
      {!data ? <Loading error={error} what="incidents" /> : (
        <Card flush>
          {data.length === 0 ? <Empty>No incidents match.</Empty> : (
            <div className="table-wrap"><table>
              <thead><tr><th>ID</th><th>Severity</th><th>Status</th><th>Node / GPUs</th><th>Title</th><th className="num">Perf vs peers</th><th>Confidence</th><th className="num">Recurrence</th><th>Opened</th><th>Updated</th></tr></thead>
              <tbody>{data.map((i) => (
                <tr key={i.id}>
                  <td><Link href={`/incident/?id=${i.id}`}>{i.id}</Link></td><td><StatusBadge status={i.severity} /></td><td><IncidentStatus status={i.status} /></td>
                  <td><Link href={`/node/?id=${i.node}`}>{i.node}</Link>{i.gpus?.length ? <span className="muted small"> · GPU {i.gpus.join(",")}</span> : null}</td>
                  <td>{i.title}</td><td className="num">{pct(i.perf_deviation_pct)}</td><td><Confidence label={i.confidence_label} /></td>
                  <td className="num">{i.recurrence_count > 0 ? `${i.recurrence_count}× before` : "—"}</td>
                  <td className="muted">{ago(i.created_at)}</td><td className="muted">{ago(i.updated_at)}</td>
                </tr>))}
              </tbody></table></div>)}
        </Card>)}
    </>
  );
}
