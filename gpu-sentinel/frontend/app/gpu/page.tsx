"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { usePoll } from "@/lib/api";
import { num } from "@/lib/format";
import { Card, KV, Loading, PageHead, StatusBadge } from "@/components/ui";
import { DeviationBar, LineChart } from "@/components/charts";
import { Hypotheses, SignalsTable } from "@/components/Hypotheses";

function GPUDetail() {
  const sp = useSearchParams();
  const node = sp.get("node") ?? "";
  const index = sp.get("index") ?? "0";
  const { data: d, error } = usePoll<any>(`/gpus/${node}/${index}`, 10000);
  const { data: h } = usePoll<any>(`/nodes/${node}/history?gpu=${index}&minutes=30&metrics=temp_c,sm_clock_mhz,power_w,gpu_util,hbm_bw_gbps,perf_index`, 15000);
  if (!d) return <Loading error={error} what="GPU" />;
  const charts: [string, string, string][] = [["temp_c", "Temperature", "°C"], ["sm_clock_mhz", "SM clock", " MHz"], ["power_w", "Power", " W"],
    ["gpu_util", "Utilization", "%"], ["hbm_bw_gbps", "HBM bandwidth", " GB/s"], ["perf_index", "Tensor activity", "%"]];
  return (
    <>
      <PageHead title={`${node} / GPU ${index}`} desc={`${d.model} · UUID ${d.uuid} · job ${d.workload_id}`}>
        <Link href={`/node/?id=${node}`}>← Node {node}</Link><StatusBadge status={d.status} />
      </PageHead>
      {d.throttle_reasons.length > 0 && <div className="banner warn">Clock throttle reasons reported by the driver: {d.throttle_reasons.join(", ")}</div>}
      {d.hypotheses.length > 0 && <Card title="Diagnosis involving this GPU"><Hypotheses items={d.hypotheses} /></Card>}
      <div className="grid g2" style={{ margin: "14px 0" }}>
        <Card title="Peer benchmarking" flush>
          <div className="table-wrap"><table>
            <thead><tr><th>Metric</th><th className="num">This GPU</th><th className="num">Peer median</th><th className="num">p10–p90</th><th>Deviation</th><th className="num">Percentile</th></tr></thead>
            <tbody>{d.peer_comparison.map((p: any) => (
              <tr key={p.metric}><td>{p.label}</td><td className="num">{num(p.value, 2)} {p.unit}</td><td className="num">{num(p.peer.median, 2)}</td>
                <td className="num muted">{num(p.peer.p10)}–{num(p.peer.p90)}</td><td><DeviationBar pct={p.deviation_pct} /></td><td className="num">{num(p.percentile_rank, 0)}</td></tr>))}
            </tbody></table></div>
          <div className="muted small" style={{ padding: "8px 16px" }}>Peers: n={d.peer_comparison[0]?.peer.n} GPUs in group “{d.peer_comparison[0]?.group}”.</div>
        </Card>
        <Card title="Current telemetry">
          <KV items={Object.entries(d.metrics).map(([k, v]) => [k, num(v as number, 2)])} />
        </Card>
      </div>
      <Card title="Active anomalies" flush><SignalsTable signals={d.anomalies} /></Card>
      <Card title="Last 30 minutes" className="">
        {h ? <div className="grid g3">{charts.map(([k, l, u]) => <div key={k}><h3>{l}</h3><LineChart series={[{ name: l, points: h.series[k] ?? [], unit: u }]} height={140} /></div>)}</div> : <Loading />}
      </Card>
    </>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><GPUDetail /></Suspense>;
}
