"use client";
import Link from "next/link";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { ago, num } from "@/lib/format";
import { Card, Empty, Loading, PageHead, StatusBadge } from "@/components/ui";

const CATS = ["", "thermal", "clocks", "power", "compute", "memory", "ecc", "nvlink", "pcie", "cpu", "network", "communication", "performance", "workload"];

export default function Anomalies() {
  const [active, setActive] = useState(true);
  const [cat, setCat] = useState("");
  const { data, error } = usePoll<any[]>(`/anomalies?active=${active}${cat ? `&category=${cat}` : ""}`, 10000);
  return (
    <>
      <PageHead title="Anomalies" desc="Per-metric signals from the deterministic detectors (static thresholds, moving average, rolling z-score, peer deviation, historical baseline, rate of change). Correlated into incidents per node.">
        <div className="row">
          <select value={cat} onChange={(e) => setCat(e.target.value)}>{CATS.map((c) => <option key={c} value={c}>{c || "All categories"}</option>)}</select>
          <label className="row small"><input type="checkbox" checked={active} onChange={(e) => setActive(e.target.checked)} /> Active only</label>
        </div>
      </PageHead>
      {!data ? <Loading error={error} what="anomalies" /> : (
        <Card flush title={`${data.length} signals`}>
          {data.length === 0 ? <Empty>No anomalies. All telemetry within expected ranges.</Empty> : (
            <div className="table-wrap"><table>
              <thead><tr><th>Entity</th><th>Category</th><th>Metric</th><th className="num">Value</th><th className="num">Expected</th><th className="num">Deviation</th><th>Detected by</th><th>Severity</th><th>Incident</th><th>First seen</th></tr></thead>
              <tbody>{data.map((a) => (
                <tr key={a.id}>
                  <td><Link href={a.gpu_index !== null ? `/gpu/?node=${a.node}&index=${a.gpu_index}` : `/node/?id=${a.node}`}>{a.entity}</Link></td>
                  <td><span className="pill">{a.category}</span></td><td>{a.metric}</td>
                  <td className="num">{num(a.value, 2)}</td><td className="num">{num(a.expected, 2)}</td>
                  <td className="num">{a.deviation_pct !== null ? `${a.deviation_pct > 0 ? "+" : ""}${a.deviation_pct.toFixed(1)}%` : "—"}</td>
                  <td>{a.methods.map((m: string) => <span className="pill" key={m}>{m.replace("_", " ")}</span>)}</td>
                  <td><StatusBadge status={a.severity} /></td>
                  <td>{a.incident_id ? <Link href={`/incident/?id=${a.incident_id}`}>{a.incident_id}</Link> : "—"}</td>
                  <td className="muted">{ago(a.first_seen)}</td>
                </tr>))}
              </tbody></table></div>)}
        </Card>)}
    </>
  );
}
