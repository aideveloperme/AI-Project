"use client";
import Link from "next/link";
import { usePoll } from "@/lib/api";
import { Card, Loading, PageHead, StatusBadge } from "@/components/ui";

const COLOR: Record<string, string> = { healthy: "var(--good)", warning: "var(--warning)", critical: "var(--critical)" };

export default function ClusterHealth() {
  const { data, error } = usePoll<any[]>("/clusters", 10000);
  if (!data) return <Loading error={error} what="clusters" />;
  return (
    <>
      <PageHead title="Cluster Health" desc="Physical layout by rack. Each cell is one GPU; hover for temperature and utilization." />
      {data.map((c) => (
        <Card key={c.cluster} title={`${c.cluster} — ${c.nodes} nodes, ${c.gpus} GPUs`}
          actions={<div className="row">{Object.entries(c.status).map(([s, n]) => <StatusBadge key={s} status={s} label={`${n} ${s}`} />)}</div>}>
          <div className="muted small" style={{ marginBottom: 10 }}>Peer groups: {c.peer_groups.join(" · ")}</div>
          <div className="grid g4">
            {c.racks.map((r: any) => (
              <div key={r.rack}>
                <h3>Rack {r.rack}</h3>
                <div className="rack">
                  {r.nodes.map((n: any) => (
                    <div className="node-tile" key={n.node}>
                      <div className="row" style={{ justifyContent: "space-between" }}>
                        <Link href={`/node/?id=${n.node}`}><b>{n.node}</b></Link><StatusBadge status={n.status} />
                      </div>
                      <div className="muted small">{n.workload} · {n.peer_group}</div>
                      <div className="gpu-cells">
                        {n.gpus.map((g: any) => (
                          <Link key={g.index} href={`/gpu/?node=${n.node}&index=${g.index}`}
                            title={`GPU ${g.index}: ${g.status} · ${g.temp_c}°C · ${g.util}% util`}>
                            <div className="gpu-cell" style={{ background: COLOR[g.status] }} />
                          </Link>))}
                      </div>
                    </div>))}
                </div>
              </div>))}
          </div>
          <div className="row small muted" style={{ marginTop: 12 }}>
            Legend: <StatusBadge status="healthy" /> <StatusBadge status="warning" /> <StatusBadge status="critical" />
          </div>
        </Card>))}
    </>
  );
}
