"use client";
import Link from "next/link";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { Card, Loading, PageHead, StatusBadge } from "@/components/ui";
import { LineChart } from "@/components/charts";

export default function Trends() {
  const [minutes, setMinutes] = useState(180);
  const [node, setNode] = useState("");
  const { data: nodes } = usePoll<any[]>("/nodes", 0);
  const metrics = node ? "throughput,cpu_util,nccl_busbw_gbps,ib_rx_gbps" : "gpu_util,temp_c,power_kw,throughput";
  const { data, error } = usePoll<any>(`/trends?minutes=${minutes}&metrics=${metrics}${node ? `&node=${node}` : ""}`, 30000);
  const labels: Record<string, string> = { gpu_util: "Avg GPU utilization (%)", temp_c: "Avg GPU temperature (°C)", power_kw: "GPU power (kW)",
    throughput: "Throughput", cpu_util: "CPU utilization (%)", nccl_busbw_gbps: "NCCL bus bandwidth (GB/s)", ib_rx_gbps: "InfiniBand RX (Gb/s)" };
  const markers = (data?.incidents ?? []).map((i: any) => ({ x: i.ts, label: `${i.id} ${i.title}`, color: i.severity === "critical" ? "var(--critical)" : "var(--warning)" }));
  return (
    <>
      <PageHead title="Historical Trends" desc="Downsampled history from the time-series store. Dashed lines mark incident openings.">
        <div className="row">
          <select value={node} onChange={(e) => setNode(e.target.value)}><option value="">Whole fleet</option>{nodes?.map((n) => <option key={n.node} value={n.node}>{n.node}</option>)}</select>
          <select value={minutes} onChange={(e) => setMinutes(+e.target.value)}>
            <option value={30}>30 minutes</option><option value={180}>3 hours</option><option value={1440}>24 hours</option><option value={10080}>7 days</option>
          </select>
        </div>
      </PageHead>
      {!data ? <Loading error={error} what="trends" /> : (<>
        <div className="grid g2">{Object.entries(data.series).map(([k, pts]: any) => (
          <Card key={k} title={labels[k] ?? k}><LineChart series={[{ name: labels[k] ?? k, points: pts }]} markers={markers} /></Card>))}</div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <Card title={`Incidents in window (${data.incident_count})`} flush>
            <div className="table-wrap"><table><thead><tr><th>ID</th><th>Node</th><th>Category</th><th>Severity</th><th>Opened</th></tr></thead>
              <tbody>{data.incidents.slice().reverse().map((i: any) => (
                <tr key={i.id}><td><Link href={`/incident/?id=${i.id}`}>{i.id}</Link></td><td>{i.node}</td><td>{i.category}</td><td><StatusBadge status={i.severity} /></td>
                  <td className="muted">{new Date(i.ts).toLocaleString()}</td></tr>))}</tbody></table></div>
          </Card>
          <Card title="Anomaly signals by category (window)" flush>
            <div className="table-wrap"><table><thead><tr><th>Category</th><th className="num">Signals</th></tr></thead>
              <tbody>{Object.entries(data.anomalies_by_category).sort((a: any, b: any) => b[1] - a[1]).map(([c, n]: any) => <tr key={c}><td>{c}</td><td className="num">{n}</td></tr>)}</tbody></table></div>
          </Card>
        </div>
      </>)}
    </>
  );
}
