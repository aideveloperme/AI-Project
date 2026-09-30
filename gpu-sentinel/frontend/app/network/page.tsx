"use client";
import Link from "next/link";
import { usePoll } from "@/lib/api";
import { num } from "@/lib/format";
import { Card, Loading, PageHead, StatusBadge } from "@/components/ui";

const COLS: [string, string, number][] = [
  ["ib_rx_gbps", "IB RX Gb/s", 0], ["ib_tx_gbps", "IB TX Gb/s", 0], ["ib_symbol_err_rate", "IB sym err/s", 2], ["net_rx_gbps", "Eth RX Gb/s", 1],
  ["net_err_rate", "Eth err/s", 2], ["net_drop_rate", "Drops/s", 2], ["net_latency_us", "Latency µs", 2],
  ["nccl_busbw_gbps", "NCCL busbw GB/s", 0], ["nccl_comm_ratio", "Comm % of step", 1], ["nvlink_avg_gbps", "NVLink avg GB/s", 0], ["nvlink_crc_rate", "NVLink CRC/s", 2],
];

export default function Network() {
  const { data, error } = usePoll<any[]>("/network", 10000);
  if (!data) return <Loading error={error} what="network" />;
  return (
    <>
      <PageHead title="Network & Communication" desc="InfiniBand, Ethernet, fabric latency, NVLink and NCCL collective performance per node. Flagged cells deviate from peers." />
      <Card flush>
        <div className="table-wrap"><table>
          <thead><tr><th>Node</th><th>Status</th>{COLS.map((c) => <th key={c[0]} className="num">{c[1]}</th>)}</tr></thead>
          <tbody>{data.map((n) => (
            <tr key={n.node}><td><Link href={`/node/?id=${n.node}`}>{n.node}</Link></td><td><StatusBadge status={n.status} /></td>
              {COLS.map(([k, , d]) => {
                const flagged = n.flagged.includes(k);
                const p = n.peer[k];
                return <td key={k} className="num" title={p ? `peer median ${p.median} (${p.deviation_pct}%)` : ""}
                  style={flagged ? { outline: "1px solid var(--warning)", fontWeight: 700 } : undefined}>{flagged ? "⚠ " : ""}{num(n[k], d)}</td>;
              })}</tr>))}
          </tbody></table></div>
      </Card>
    </>
  );
}
