"use client";
import React from "react";

const STATUS: Record<string, { color: string; icon: string; label: string }> = {
  healthy: { color: "var(--good)", icon: "✓", label: "Healthy" },
  good: { color: "var(--good)", icon: "✓", label: "Healthy" },
  info: { color: "var(--good)", icon: "✓", label: "Info" },
  warning: { color: "var(--warning)", icon: "!", label: "Warning" },
  serious: { color: "var(--serious)", icon: "!", label: "Serious" },
  critical: { color: "var(--critical)", icon: "✕", label: "Critical" },
};

/** Status is never color-alone: dot + icon + label. */
export function StatusBadge({ status, label }: { status: string; label?: string }) {
  const s = STATUS[status] ?? { color: "var(--ink-3)", icon: "•", label: status };
  return (
    <span className="badge" title={s.label}>
      <span className="dot" style={{ background: s.color }} aria-hidden />
      <span aria-hidden style={{ color: s.color, fontSize: 11 }}>{s.icon}</span>
      {label ?? s.label}
    </span>
  );
}

const INC_STATUS: Record<string, string> = {
  OPEN: "critical", ACKNOWLEDGED: "warning", INVESTIGATING: "serious", RESOLVED: "good", FALSE_POSITIVE: "info",
};
export function IncidentStatus({ status }: { status: string }) {
  return <StatusBadge status={INC_STATUS[status] ?? "info"} label={status.replace("_", " ")} />;
}

export function Confidence({ label, value }: { label: string; value?: number }) {
  const c = label === "high" ? "var(--good-text)" : label === "medium" ? "var(--warning)" : "var(--ink-3)";
  return (
    <span className="conf" style={{ color: c }} title="Confidence of the inferred cause (not a certainty)">
      {label} confidence{value !== undefined ? ` · ${value.toFixed(2)}` : ""}
    </span>
  );
}

export function Stat({ label, value, sub, status }: { label: string; value: React.ReactNode; sub?: React.ReactNode; status?: string }) {
  return (
    <div className="card stat">
      <div className="label row" style={{ justifyContent: "space-between" }}>
        <span>{label}</span>
        {status && <StatusBadge status={status} />}
      </div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Card({ title, children, actions, flush, className }: {
  title?: React.ReactNode; children: React.ReactNode; actions?: React.ReactNode; flush?: boolean; className?: string;
}) {
  return (
    <section className={`card ${flush ? "flush" : ""} ${className ?? ""}`}>
      {(title || actions) && (
        <div className="row" style={{ justifyContent: "space-between", marginBottom: flush ? 0 : 10, padding: flush ? "14px 16px 8px" : 0 }}>
          {title && <h2 style={{ margin: 0 }}>{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Loading({ error, what = "data" }: { error?: string | null; what?: string }) {
  if (error) return <div className="banner warn">Could not load {what}: {error}</div>;
  return <div className="muted" style={{ padding: 16 }}>Loading {what}…</div>;
}

export function PageHead({ title, desc, children }: { title: string; desc?: string; children?: React.ReactNode }) {
  return (
    <div className="page-head">
      <div style={{ flex: 1 }}>
        <h1>{title}</h1>
        {desc && <p>{desc}</p>}
      </div>
      {children}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="muted" style={{ padding: "18px 4px" }}>{children}</div>;
}

export function KV({ items }: { items: [string, React.ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([k, v]) => (
        <React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>
      ))}
    </dl>
  );
}
