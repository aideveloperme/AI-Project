"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { usePoll } from "@/lib/api";
import { ago, num, pct } from "@/lib/format";
import { Card, KV, Loading, PageHead, Stat, StatusBadge } from "@/components/ui";
import { DeviationBar, LineChart, PeerStrip } from "@/components/charts";
import { Hypotheses, SignalsTable } from "@/components/Hypotheses";

const GPU_CHART_METRICS: [string, string, string][] = [
  ["temp_c", "Temperature", "°C"], ["sm_clock_mhz", "SM clock", " MHz"], ["power_w", "Power", " W"],
  ["gpu_util", "GPU utilization", "%"], ["hbm_bw_gbps", "HBM bandwidth", " GB/s"], ["perf_index", "Tensor activity", "%"],
];
const NODE_CHART_METRICS: [string, string, string][] = [
  ["throughput", "Workload throughput", ""], ["cpu_util", "CPU utilization", "%"],
  ["nccl_busbw_gbps", "NCCL bus bandwidth", " GB/s"], ["ib_rx_gbps", "InfiniBand RX", " Gb/s"],
];

function NodeHistory({ node, gpu, metrics, minutes }: { node: string; gpu?: number; metrics: [string, string, string][]; minutes: number }) {
  const q = `/nodes/${node}/history?metrics=${metrics.map((m) => m[0]).join(",")}&minutes=${minutes}${gpu !== undefined ? `&gpu=${gpu}` : ""}`;
  const { data } = usePoll<any>(q, 15000);
  if (!data) return <Loading what="history" />;
  return (
    <div className="grid g2">
      {metrics.map(([k, label, unit]) => (
        <div key={k}><h3>{label}</h3><LineChart series={[{ name: label, points: data.series[k] ?? [], unit }]} height={150} /></div>))}
    </div>
  );
}

function NodeDetail() {
  const id = useSearchParams().get("id") ?? "";
  const { data: d, error } = usePoll<any>(id ? `/nodes/${id}` : null, 10000);
  const [gpu, setGpu] = useState<number | undefined>(undefined);
  const [minutes, setMinutes] = useState(30);
  if (!d) return <Loading error={error} what={`node ${id}`} />;
  const focus = gpu ?? (d.anomalies.find((a: any) => a.gpu_index !== null)?.gpu_index ?? 0);
  const perf = d.peer_comparison.find((p: any) => p.metric === "throughput");
  return (
    <>
      <PageHead title={d.node} desc={`${d.server_type} · ${d.gpu_count}× ${d.gpu_model} · rack ${d.rack} · driver ${d.driver_version} / CUDA ${d.cuda_version} · peer group: ${d.peer_group}`}>
        <StatusBadge status={d.status} />
      </PageHead>
      <div className="grid g6" style={{ marginBottom: 14 }}>
        <Stat label="Throughput vs peers" value={pct(d.perf_deviation_pct)} sub={perf ? `${num(perf.value)} vs median ${num(perf.peer.median)}` : ""}
          status={d.perf_deviation_pct !== null && d.perf_deviation_pct <= -8 ? "warning" : undefined} />
        <Stat label="Avg GPU util" value={`${num(d.avg_gpu_util)}%`} />
        <Stat label="Max GPU temp" value={`${num(d.max_temp_c)}°C`} />
        <Stat label="GPU power" value={`${num(d.power_kw, 2)} kW`} />
        <Stat label="CPU util" value={`${num(d.metrics.cpu_util)}%`} sub={`load ${num(d.metrics.load1)}`} />
        <Stat label="Host memory" value={`${num(d.metrics.mem_used_pct)}%`} sub={`PSI ${num(d.metrics.mem_psi_some)}%`} />
      </div>

      {d.hypotheses.length > 0 && (
        <Card title="Diagnosis (inferred from correlated evidence)" className="" >
          <Hypotheses items={d.hypotheses} />
        </Card>)}

      <div className="grid" style={{ margin: "14px 0" }}>
        <Card title="Peer comparison — node" flush>
          <div className="table-wrap"><table>
            <thead><tr><th>Metric</th><th className="num">{d.node}</th><th className="num">Peer median</th><th>Deviation</th><th className="num">Own baseline</th></tr></thead>
            <tbody>{d.peer_comparison.map((p: any) => (
              <tr key={p.metric}><td>{p.label}</td><td className="num">{num(p.value, 2)} {p.unit}</td><td className="num">{num(p.peer.median, 2)}</td>
                <td><DeviationBar pct={p.deviation_pct} /></td><td className="num muted">{p.historical_baseline !== null ? num(p.historical_baseline, 2) : "—"}</td></tr>))}
            </tbody></table></div>
        </Card>
        <Card title="Peer comparison — GPUs (worst GPU on this node vs peer group)" flush>
          <div className="table-wrap"><table>
            <thead><tr><th>Metric</th><th className="num">Worst GPU</th><th>Band = peer-group p10–p90 · | median · dots = this node's GPUs · ● worst GPU</th><th>Deviation</th></tr></thead>
            <tbody>{d.gpu_peer_summary.map((r: any) => (
              <tr key={r.metric}><td>{r.label}</td><td className="num">GPU {r.worst_gpu}: {num(r.worst_value)} {r.unit}</td>
                <td><PeerStrip values={d.gpus.map((g: any) => g.metrics[r.metric])} highlight={r.worst_value} p10={r.p10} p90={r.p90} median={r.peer_median} unit={r.unit} /></td>
                <td><DeviationBar pct={r.worst_deviation_pct} /></td></tr>))}
            </tbody></table></div>
        </Card>
      </div>

      <Card title="GPU inventory" flush>
        <div className="table-wrap"><table>
          <thead><tr><th>GPU</th><th>Status</th><th className="num">Util</th><th className="num">SM MHz</th><th className="num">Mem MHz</th><th className="num">Temp</th><th className="num">HBM temp</th>
            <th className="num">Power / limit</th><th className="num">Mem used</th><th className="num">HBM GB/s</th><th className="num">PCIe</th><th className="num">NVLink GB/s</th><th className="num">ECC SBE/min</th><th>Throttle</th><th>Process</th></tr></thead>
          <tbody>{d.gpus.map((g: any) => {
            const m = g.metrics;
            return (
              <tr key={g.index} style={{ cursor: "pointer", outline: g.index === focus ? "1px solid var(--accent)" : undefined }} onClick={() => setGpu(g.index)}>
                <td><Link href={`/gpu/?node=${d.node}&index=${g.index}`}>GPU {g.index}</Link></td>
                <td><StatusBadge status={g.status} /></td>
                <td className="num">{num(m.gpu_util)}%</td><td className="num">{num(m.sm_clock_mhz)}</td><td className="num">{num(m.mem_clock_mhz)}</td>
                <td className="num">{num(m.temp_c)}°C</td><td className="num">{num(m.mem_temp_c)}°C</td>
                <td className="num">{num(m.power_w)} / {num(m.power_limit_w)} W</td><td className="num">{num(m.mem_used_pct)}%</td>
                <td className="num">{num(m.hbm_bw_gbps)}</td><td className="num">Gen{m.pcie_link_gen} x{m.pcie_link_width}</td>
                <td className="num">{num(m.nvlink_bw_gbps)}</td><td className="num">{num(m.ecc_sbe_rate, 2)}</td>
                <td>{g.throttle_reasons.map((t: string) => <span key={t} className="pill">{t}</span>)}</td>
                <td className="small muted">{g.processes.map((p: any) => `${p.name ?? "job"} (${p.job_id})`).join(", ")}</td>
              </tr>);
          })}</tbody></table></div>
      </Card>

      <div className="grid g2" style={{ margin: "14px 0" }}>
        <Card title="Host, network & communication">
          <KV items={[["CPU frequency", `${num(d.metrics.cpu_freq_mhz)} MHz`], ["Context switches", `${num(d.metrics.ctx_switches_k)} k/s`],
              ["iowait", `${num(d.metrics.cpu_iowait)}%`], ["Ethernet RX / TX", `${num(d.metrics.net_rx_gbps, 2)} / ${num(d.metrics.net_tx_gbps, 2)} Gb/s`],
              ["InfiniBand RX / TX", `${num(d.metrics.ib_rx_gbps)} / ${num(d.metrics.ib_tx_gbps)} Gb/s`],
              ["Net errors / drops", `${num(d.metrics.net_err_rate, 2)} / ${num(d.metrics.net_drop_rate, 2)} per s`],
              ["IB symbol errors", `${num(d.metrics.ib_symbol_err_rate, 2)} /s`], ["Fabric latency", `${num(d.metrics.net_latency_us, 2)} µs`],
              ["NCCL all-reduce busbw", `${num(d.metrics.nccl_busbw_gbps)} GB/s`], ["Communication share", `${num(d.metrics.nccl_comm_ratio)}% of step`],
              ["Data-loader wait", `${num(d.metrics.dataloader_wait_pct)}%`], ["Step time", `${num(d.metrics.step_time_ms)} ms`]]} />
        </Card>
        <Card title="Running workload">
          {d.workload ? (<dl className="kv">
            <dt>Job</dt><dd><Link href={`/workloads/?job=${d.workload.job_id}`}>{d.workload.job_id}</Link> — {d.workload.name}</dd>
            <dt>Scheduler</dt><dd>{d.workload.scheduler}{d.workload.namespace ? ` (ns ${d.workload.namespace})` : ""}</dd>
            <dt>Type</dt><dd>{d.workload.kind} · {d.workload.framework}</dd>
            <dt>Nodes</dt><dd>{d.workload.nodes.length} ({d.workload.nodes.join(", ")})</dd>
          </dl>) : <span className="muted">No workload.</span>}
          <h3 style={{ marginTop: 16 }}>Incident history</h3>
          {d.incidents.length === 0 ? <div className="muted small">No incidents recorded for this node.</div> : (
            <ul className="tight">{d.incidents.map((i: any) => (
              <li key={i.id}><Link href={`/incident/?id=${i.id}`}>{i.id}</Link> {i.title} — <span className="muted">{i.status}, {ago(i.created_at)}</span></li>))}</ul>)}
        </Card>
      </div>

      <Card title={`Active anomalies (${d.anomalies.length})`} flush><SignalsTable signals={d.anomalies} /></Card>

      <Card title="Historical trends" className="" actions={
        <div className="row">
          <select value={minutes} onChange={(e) => setMinutes(+e.target.value)}>
            <option value={15}>Last 15 min</option><option value={30}>Last 30 min</option><option value={120}>Last 2 h</option><option value={1440}>Last 24 h</option>
          </select>
        </div>}>
        <h3>Node</h3>
        <NodeHistory node={d.node} metrics={NODE_CHART_METRICS} minutes={minutes} />
        <h3 style={{ marginTop: 16 }}>GPU {focus} (click a GPU row to change)</h3>
        <NodeHistory node={d.node} gpu={focus} metrics={GPU_CHART_METRICS} minutes={minutes} />
      </Card>
    </>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><NodeDetail /></Suspense>;
}
