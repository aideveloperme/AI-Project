"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { usePoll } from "@/lib/api";
import { num } from "@/lib/format";
import { Card, Loading, PageHead, StatusBadge } from "@/components/ui";
import { DeviationBar } from "@/components/charts";

const COLS: [string, string, number?][] = [
  ["gpu_util", "Util %"], ["sm_clock_mhz", "SM clock"], ["temp_c", "Temp °C"], ["power_w", "Power W"],
  ["hbm_bw_gbps", "HBM GB/s"], ["nvlink_bw_gbps", "NVLink GB/s"], ["ecc_sbe_rate", "SBE/min", 2],
];

export default function Fleet() {
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<string>("perf_deviation_pct");
  const { data, error } = usePoll<any[]>(`/gpus${status ? `?status=${status}` : ""}`, 10000);
  const rows = useMemo(() => {
    const r = (data ?? []).filter((g) => !q || g.key.includes(q.toLowerCase()));
    return [...r].sort((a, b) => (a[sort] ?? 0) - (b[sort] ?? 0));
  }, [data, q, sort]);
  return (
    <>
      <PageHead title="GPU Fleet" desc="Every GPU with live telemetry and its throughput proxy (tensor activity) versus peer median.">
        <div className="row">
          <input placeholder="Filter e.g. gpu-04" value={q} onChange={(e) => setQ(e.target.value)} />
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All statuses</option><option value="healthy">Healthy</option><option value="warning">Warning</option><option value="critical">Critical</option>
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="perf_deviation_pct">Sort: worst performance</option><option value="gpu_util">Sort: lowest util</option>
            <option value="sm_clock_mhz">Sort: lowest clock</option>
          </select>
        </div>
      </PageHead>
      {!data ? <Loading error={error} what="GPUs" /> : (
        <Card flush title={`${rows.length} GPUs`}>
          <div className="table-wrap"><table>
            <thead><tr><th>GPU</th><th>Status</th><th>Model</th>{COLS.map((c) => <th key={c[0]} className="num">{c[1]}</th>)}<th>Throughput proxy vs peers</th><th>Throttle</th></tr></thead>
            <tbody>{rows.map((g) => (
              <tr key={g.key}>
                <td><Link href={`/gpu/?node=${g.node}&index=${g.index}`}>{g.node} / GPU {g.index}</Link></td>
                <td><StatusBadge status={g.status} /></td>
                <td className="muted small">{g.model.replace("NVIDIA ", "")}</td>
                {COLS.map((c) => <td key={c[0]} className="num">{num(g[c[0]], c[2] ?? 1)}</td>)}
                <td><DeviationBar pct={g.perf_deviation_pct} /></td>
                <td>{g.throttle_reasons.map((t: string) => <span className="pill" key={t}>{t}</span>)}</td>
              </tr>))}
            </tbody></table></div>
        </Card>)}
    </>
  );
}
