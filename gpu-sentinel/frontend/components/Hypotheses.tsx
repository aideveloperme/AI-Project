"use client";
import { Confidence } from "@/components/ui";

/** Ranked hypotheses with preserved evidence — INFERRED, never asserted as fact. */
export function Hypotheses({ items, showActions = true }: { items: any[]; showActions?: boolean }) {
  if (!items?.length) return <div className="muted">No failure pattern matched.</div>;
  return (
    <div className="grid" style={{ gap: 10 }}>
      {items.map((h, i) => (
        <div key={h.rule_id} className="card" style={{ background: "var(--surface-2)" }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <b>{i === 0 ? "Most likely: " : "Alternative: "}{h.title}</b>
            <Confidence label={h.confidence_label} value={h.confidence} />
          </div>
          <p className="ink2" style={{ margin: "6px 0" }}>{h.explanation}</p>
          <div className="grid g2">
            <div>
              <div className="section-label obs">Supporting evidence (observed)</div>
              <ul className="tight evidence">{h.evidence_for.map((e: string, j: number) => <li key={j}>{e}</li>)}</ul>
            </div>
            <div>
              <div className="section-label muted">Contradicting / not observed</div>
              {h.evidence_against.length ? <ul className="tight evidence">{h.evidence_against.map((e: string, j: number) => <li key={j}>{e}</li>)}</ul> : <div className="muted small">None</div>}
            </div>
          </div>
          {showActions && i === 0 && (<>
            <div className="section-label rec">Recommended investigation</div>
            <ol className="tight">{h.recommended_actions.map((a: string, j: number) => <li key={j}>{a}</li>)}</ol>
          </>)}
          {h.affected_gpus?.length > 0 && <div className="muted small" style={{ marginTop: 6 }}>Affected GPUs: {h.affected_gpus.join(", ")}</div>}
        </div>))}
    </div>
  );
}

export function SignalsTable({ signals }: { signals: any[] }) {
  if (!signals?.length) return <div className="muted" style={{ padding: 12 }}>No active anomaly signals.</div>;
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>Scope</th><th>Metric</th><th className="num">Value</th><th className="num">Expected</th><th className="num">Deviation</th><th className="num">z</th><th>Methods</th><th>Severity</th></tr></thead>
      <tbody>{signals.map((s, i) => (
        <tr key={i}>
          <td>{s.gpu_index !== null && s.gpu_index !== undefined ? `GPU ${s.gpu_index}` : "node"}</td>
          <td>{s.label}</td>
          <td className="num">{s.value?.toFixed?.(1)} {s.unit}</td>
          <td className="num">{s.expected !== null && s.expected !== undefined ? `${s.expected.toFixed(1)} ${s.unit}` : "—"}</td>
          <td className="num">{s.deviation_pct !== null && s.deviation_pct !== undefined ? `${s.deviation_pct > 0 ? "+" : ""}${s.deviation_pct.toFixed(1)}%` : "—"}</td>
          <td className="num">{s.zscore !== null && s.zscore !== undefined ? s.zscore.toFixed(1) : "—"}</td>
          <td>{(s.methods ?? [s.method]).map((m: string) => <span className="pill" key={m}>{m.replace("_", " ")}</span>)}</td>
          <td>{s.severity}</td>
        </tr>))}
      </tbody></table></div>
  );
}
