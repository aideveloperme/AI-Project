"use client";
import Link from "next/link";
import { usePoll } from "@/lib/api";
import { num, pct } from "@/lib/format";
import { Card, Loading, PageHead, Stat, StatusBadge } from "@/components/ui";
import { DeviationBar } from "@/components/charts";

export default function Workloads() {
  const { data, error } = usePoll<any[]>("/workloads", 10000);
  if (!data) return <Loading error={error} what="workloads" />;
  return (
    <>
      <PageHead title="Workloads" desc="Slurm jobs and Kubernetes deployments. For synchronous training, the slowest rank gates the job — stragglers are highlighted." />
      {data.map((w) => (
        <Card key={w.job_id} title={`${w.job_id} — ${w.name}`} actions={<div className="row"><span className="pill">{w.scheduler}</span><span className="pill">{w.kind}</span><span className="pill">{w.framework}</span></div>}>
          <div className="grid g4" style={{ marginBottom: 12 }}>
            <Stat label={`Aggregate throughput (${w.unit})`} value={num(w.throughput)} sub={w.baseline_throughput ? `baseline ${num(w.baseline_throughput)} (${pct(w.change_pct)})` : "building baseline…"} />
            <Stat label="Effective (gated by slowest rank)" value={w.kind === "training" ? num(w.effective_throughput) : "n/a"} />
            <Stat label="Nodes / GPUs" value={`${w.nodes.length} / ${w.gpus}`} sub={w.user ? `user ${w.user}` : w.namespace ? `namespace ${w.namespace}` : ""} />
            <Stat label="Stragglers" value={w.stragglers.length} status={w.stragglers.length ? "warning" : "healthy"} sub={w.incidents.length ? `incidents: ${w.incidents.join(", ")}` : ""} />
          </div>
          <div className="table-wrap"><table>
            <thead><tr><th>Node</th><th>Status</th><th className="num">Throughput</th><th>vs job median</th><th className="num">GPU util</th><th className="num">NCCL busbw</th><th className="num">Step time</th></tr></thead>
            <tbody>{w.per_node.map((n: any) => (
              <tr key={n.node}><td><Link href={`/node/?id=${n.node}`}>{n.node}</Link></td><td><StatusBadge status={n.status} /></td>
                <td className="num">{num(n.throughput)}</td><td><DeviationBar pct={n.deviation_pct} /></td><td className="num">{num(n.gpu_util)}%</td>
                <td className="num">{num(n.nccl_busbw_gbps)} GB/s</td><td className="num">{num(n.step_time_ms)} ms</td></tr>))}
            </tbody></table></div>
        </Card>))}
    </>
  );
}
