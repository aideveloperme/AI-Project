"use client";
import Link from "next/link";
import { usePoll } from "@/lib/api";
import { ago, num, pct } from "@/lib/format";
import { Card, Confidence, Empty, Loading, PageHead, Stat, StatusBadge } from "@/components/ui";
import { LineChart } from "@/components/charts";

export default function Overview() {
  const { data: d, error } = usePoll<any>("/overview", 10000);
  if (!d) return <Loading error={error} what="overview" />;
  const trend = (k: string) => d.trend.map((t: any) => [new Date(t.ts).getTime(), t[k] ?? null] as [number, number | null]);
  const fleetStatus = d.critical_gpus ? "critical" : d.warning_gpus ? "warning" : "healthy";
  return (
    <>
      <PageHead title="Overview" desc={`Cluster ${d.clusters.join(", ")} · ${d.total_nodes} nodes · updated ${ago(d.timestamp)}`}>
        <StatusBadge status={fleetStatus} label={fleetStatus === "healthy" ? "All systems nominal" : fleetStatus === "warning" ? "Degraded" : "Critical issues"} />
      </PageHead>
      <div className="grid g6" style={{ marginBottom: 14 }}>
        <Stat label="Total GPUs" value={d.total_gpus} sub={`${d.total_nodes} nodes`} />
        <Stat label="Healthy GPUs" value={d.healthy_gpus} sub={`${((d.healthy_gpus / d.total_gpus) * 100).toFixed(0)}% of fleet`} />
        <Stat label="Warning GPUs" value={d.warning_gpus} status={d.warning_gpus ? "warning" : undefined} />
        <Stat label="Critical GPUs" value={d.critical_gpus} status={d.critical_gpus ? "critical" : undefined} />
        <Stat label="Active incidents" value={d.active_incidents} sub={`${d.critical_incidents} critical`} status={d.critical_incidents ? "critical" : d.active_incidents ? "warning" : undefined} />
        <Stat label="Performance anomalies" value={d.performance_anomalies} sub={`${d.active_anomalies} active signals`} />
      </div>
      <div className="grid g3" style={{ marginBottom: 14 }}>
        <Stat label="Average GPU utilization" value={`${num(d.avg_gpu_util)}%`} />
        <Stat label="Average GPU temperature" value={`${num(d.avg_temp_c)}°C`} />
        <Stat label="Fleet GPU power" value={`${num(d.total_power_kw)} kW`} />
      </div>
      <div className="grid g2" style={{ marginBottom: 14 }}>
        <Card title="Top problem nodes" flush>
          {d.top_problem_nodes.length === 0 ? <Empty>No abnormal nodes. Every node is within its peer baseline.</Empty> : (
            <div className="table-wrap"><table>
              <thead><tr><th>Node</th><th>Status</th><th className="num">vs peers</th><th>Possible contributing factor</th><th>Incident</th></tr></thead>
              <tbody>{d.top_problem_nodes.map((p: any) => (
                <tr key={p.node}>
                  <td><Link href={`/node/?id=${p.node}`}>{p.node}</Link></td>
                  <td><StatusBadge status={p.status} /></td>
                  <td className="num">{pct(p.perf_deviation_pct)}</td>
                  <td>{p.top_hypothesis ?? <span className="muted">{p.signals} signals</span>} {p.confidence && <Confidence label={p.confidence} />}</td>
                  <td>{p.incident_id ? <Link href={`/incident/?id=${p.incident_id}`}>{p.incident_id}</Link> : <span className="muted">pending</span>}</td>
                </tr>))}
              </tbody></table></div>)}
        </Card>
        <Card title="Active incidents" flush actions={<Link href="/incidents/" className="small">All incidents →</Link>}>
          {d.recent_incidents.length === 0 ? <Empty>No active incidents.</Empty> : (
            <div className="table-wrap"><table>
              <thead><tr><th>ID</th><th>Severity</th><th>Title</th><th>Opened</th></tr></thead>
              <tbody>{d.recent_incidents.map((i: any) => (
                <tr key={i.id}><td><Link href={`/incident/?id=${i.id}`}>{i.id}</Link></td><td><StatusBadge status={i.severity} /></td>
                  <td>{i.title}</td><td className="muted">{ago(i.created_at)}</td></tr>))}
              </tbody></table></div>)}
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Fleet GPU utilization (%)"><LineChart series={[{ name: "Avg GPU utilization", points: trend("gpu_util"), unit: "%" }]} unit="%" /></Card>
        <Card title="Fleet average temperature (°C)"><LineChart series={[{ name: "Avg temperature", points: trend("temp_c"), unit: "°C" }]} /></Card>
        <Card title="Aggregate workload throughput"><LineChart series={[{ name: "Throughput", points: trend("throughput") }]} /></Card>
        <Card title="Fleet GPU power (kW)"><LineChart series={[{ name: "Power", points: trend("power_kw"), unit: " kW" }]} /></Card>
      </div>
    </>
  );
}
