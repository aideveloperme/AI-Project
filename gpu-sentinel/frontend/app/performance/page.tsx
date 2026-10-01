"use client";
import Link from "next/link";
import { useState } from "react";
import { usePoll } from "@/lib/api";
import { num } from "@/lib/format";
import { Card, Loading, PageHead, StatusBadge } from "@/components/ui";
import { DeviationBar } from "@/components/charts";

const METRICS: [string, string, string][] = [
  ["perf_index", "gpu", "Tensor activity (throughput proxy)"], ["sm_clock_mhz", "gpu", "SM clock"], ["temp_c", "gpu", "GPU temperature"],
  ["hbm_bw_gbps", "gpu", "HBM bandwidth"], ["power_w", "gpu", "Power"], ["gpu_util", "gpu", "GPU utilization"],
  ["nvlink_bw_gbps", "gpu", "NVLink bandwidth"], ["throughput", "node", "Node workload throughput"], ["nccl_busbw_gbps", "node", "NCCL bus bandwidth"],
];

export default function Performance() {
  const { data, error } = usePoll<any>("/performance", 10000);
  const [m, setM] = useState(0);
  const [metric, level] = METRICS[m];
  const { data: peers } = usePoll<any>(`/peers?metric=${metric}&level=${level}`, 10000);
  if (!data) return <Loading error={error} what="performance" />;
  return (
    <>
      <PageHead title="Performance & GPU Peer Benchmarking"
        desc="Each node and GPU is compared with comparable peers (same GPU model, server type and workload class) using robust median/MAD statistics." />
      <Card title="Node throughput vs. peer median" flush>
        <div className="table-wrap"><table>
          <thead><tr><th>Node</th><th>Workload</th><th>Peer group</th><th className="num">Throughput</th><th className="num">Peer median</th><th>Deviation</th><th className="num">Step time</th><th>Status</th><th>Possible factor</th></tr></thead>
          <tbody>{data.nodes.map((n: any) => (
            <tr key={n.node}><td><Link href={`/node/?id=${n.node}`}>{n.node}</Link></td><td>{n.workload}</td><td className="muted small">{n.group}</td>
              <td className="num">{num(n.throughput)} {n.unit}</td><td className="num">{num(n.peer_median)}</td><td><DeviationBar pct={n.deviation_pct} /></td>
              <td className="num">{num(n.step_time_ms)} ms</td><td><StatusBadge status={n.status} /></td><td>{n.top_hypothesis ?? ""}</td></tr>))}
          </tbody></table></div>
      </Card>
      <div style={{ height: 14 }} />
      <Card title="Peer distribution by metric" actions={
        <select value={m} onChange={(e) => setM(+e.target.value)}>{METRICS.map((x, i) => <option key={x[0] + x[1]} value={i}>{x[2]}</option>)}</select>}>
        {!peers ? <Loading /> : (<>
          <div className="row small muted" style={{ marginBottom: 10 }}>
            {Object.entries(peers.groups).map(([g, s]: any) => <span key={g} className="pill">{g}: median {num(s.median)} {peers.unit} (p10 {num(s.p10)} – p90 {num(s.p90)}, n={s.n})</span>)}
          </div>
          <div className="table-wrap" style={{ maxHeight: 520, overflowY: "auto" }}><table>
            <thead><tr><th>{level === "gpu" ? "GPU" : "Node"}</th><th className="num">Value</th><th className="num">Peer median</th><th>Deviation from peer median</th><th className="num">Robust z</th><th className="num">Percentile</th><th>Status</th></tr></thead>
            <tbody>{peers.rows.map((r: any) => (
              <tr key={r.entity}><td>{level === "gpu" ? <Link href={`/gpu/?node=${r.entity.split("/")[0]}&index=${r.entity.split("gpu").pop()}`}>{r.entity}</Link> : <Link href={`/node/?id=${r.entity}`}>{r.entity}</Link>}</td>
                <td className="num">{num(r.value, 2)} {peers.unit}</td><td className="num">{num(r.peer_median, 2)}</td><td><DeviationBar pct={r.deviation_pct} /></td>
                <td className="num">{num(r.robust_z)}</td><td className="num">{num(r.percentile, 0)}</td><td>{r.status && <StatusBadge status={r.status} />}</td></tr>))}
            </tbody></table></div>
        </>)}
      </Card>
    </>
  );
}
