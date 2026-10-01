"use client";
/* Hand-rolled SVG charts (no chart library → small, offline-friendly bundle).
 * Rules: one y-axis, 2px lines, recessive grid, crosshair tooltip, legend for ≥2 series,
 * categorical colors in fixed slot order, text in ink tokens (never series color). */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { num, time } from "@/lib/format";

export type Series = { name: string; points: [number, number | null][]; color?: string; unit?: string };
const SLOTS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) { min -= 1; max += 1; }
  const step0 = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let v = Math.floor(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function LineChart({ series, height = 200, unit = "", markers = [], yMin }: {
  series: Series[]; height?: number; unit?: string; yMin?: number;
  markers?: { x: number; label: string; color?: string }[];
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [W, setW] = useState(800);
  useEffect(() => {
    // Draw in real pixels so text and strokes are never stretched.
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(200, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [series.length > 0 && series[0].points.length > 0]);
  const H = height, L = 48, R = 12, T = 10, B = 24;
  const all = series.flatMap((s) => s.points.filter((p) => p[1] !== null) as [number, number][]);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let lo = yMin ?? Math.min(...ys), hi = Math.max(...ys);
  const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.05 || 1;
  lo = yMin ?? lo - pad; hi += pad;
  const ticks = niceTicks(lo, hi);
  if (ticks.length) { lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]); }
  const sx = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - lo) / (hi - lo || 1)) * (H - T - B);
  const xTicks = useMemo(() => {
    const n = Math.max(2, Math.min(6, Math.floor(W / 130))); const out = [];
    for (let i = 0; i <= n; i++) out.push(x0 + ((x1 - x0) * i) / n);
    return out;
  }, [x0, x1, W]);
  if (!all.length) return <div ref={ref} className="muted small" style={{ height }}>No data yet.</div>;

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const xv = x0 + ((px - L) / (W - L - R)) * (x1 - x0);
    setHover(Math.max(x0, Math.min(x1, xv)));
  };
  const nearest = (s: Series, x: number) => {
    let best: [number, number | null] | null = null;
    for (const p of s.points) if (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
    return best;
  };
  return (
    <div className="chart" ref={ref}>
      {series.length > 1 && (
        <div className="legend">
          {series.map((s, i) => (
            <span key={s.name}><i style={{ background: s.color ?? SLOTS[i % 4] }} />{s.name}</span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
        role="img" aria-label={series.map((s) => s.name).join(", ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
            <text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize={11} fill="var(--ink-3)">{num(t)}</text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text key={i} x={sx(t)} y={H - 6} textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"} fontSize={11} fill="var(--ink-3)">{time(t, x1 - x0 < 20 * 60000)}</text>
        ))}
        {markers.filter((m) => m.x >= x0 && m.x <= x1).map((m, i) => (
          <g key={i}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={T} y2={H - B} stroke={m.color ?? "var(--critical)"} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
            <title>{m.label}</title>
          </g>
        ))}
        {series.map((s, i) => {
          const pts = s.points.filter((p) => p[1] !== null) as [number, number][];
          const d = pts.map((p, j) => `${j ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
          return <path key={s.name} d={d} fill="none" stroke={s.color ?? SLOTS[i % 4]} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />;
        })}
        {hover !== null && (
          <g>
            <line x1={sx(hover)} x2={sx(hover)} y1={T} y2={H - B} stroke="var(--axis)" vectorEffect="non-scaling-stroke" />
            {series.map((s, i) => {
              const p = nearest(s, hover);
              return p && p[1] !== null ? <circle key={s.name} cx={sx(p[0])} cy={sy(p[1])} r={4} fill={s.color ?? SLOTS[i % 4]} stroke="var(--surface-1)" strokeWidth={2} /> : null;
            })}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: `min(calc(${((sx(hover) / W) * 100).toFixed(1)}% + 10px), calc(100% - 180px))`, top: 20 }}>
          <div className="muted">{new Date(nearest(series[0], hover)?.[0] ?? hover).toLocaleTimeString()}</div>
          {series.map((s, i) => {
            const p = nearest(s, hover);
            return (
              <div key={s.name}>
                <i style={{ display: "inline-block", width: 10, height: 3, background: s.color ?? SLOTS[i % 4], marginRight: 6, verticalAlign: "middle" }} />
                {s.name}: <b>{num(p?.[1] ?? null, 2)}</b>{s.unit ?? unit}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Diverging bar: deviation from the peer median (gray midline = peer median). */
export function DeviationBar({ pct, range = 40, badHigh }: { pct: number | null; range?: number; badHigh?: boolean }) {
  if (pct === null || pct === undefined) return <span className="muted">—</span>;
  const c = Math.max(-range, Math.min(range, pct));
  const w = (Math.abs(c) / range) * 50;
  const color = pct < 0 ? "var(--div-neg)" : "var(--div-pos)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }} title={`${pct.toFixed(1)}% vs peer median`}>
      <svg width={120} height={12} aria-hidden>
        <rect x={0} y={5} width={120} height={2} fill="var(--grid)" />
        <rect x={pct < 0 ? 60 - w * 1.2 : 60} y={1} width={Math.max(2, w * 1.2)} height={10} rx={2} fill={color} />
        <rect x={59} y={0} width={2} height={12} fill="var(--div-mid)" />
      </svg>
      <span style={{ minWidth: 54, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        {pct > 0 ? "+" : ""}{pct.toFixed(1)}%{badHigh !== undefined && Math.abs(pct) >= 8 ? " ⚠" : ""}
      </span>
    </div>
  );
}

/** Peer distribution strip: every peer as a dot, p10–p90 band, median tick, highlighted entity. */
export function PeerStrip({ values, highlight, p10, p90, median, unit = "" }: {
  values: number[]; highlight?: number; p10: number; p90: number; median: number; unit?: string;
}) {
  const [tip, setTip] = useState<string | null>(null);
  const all = highlight !== undefined ? [...values, highlight] : values;
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1;
  const W = 300, H = 34;
  const sx = (v: number) => 6 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 12);
  return (
    <div className="chart" style={{ width: "100%", maxWidth: 320 }}>
      <svg viewBox={`0 0 ${W} ${H}`} height={H} onMouseLeave={() => setTip(null)}>
        <rect x={sx(p10)} y={10} width={Math.max(2, sx(p90) - sx(p10))} height={14} rx={3} fill="var(--surface-3)" />
        {values.map((v, i) => <circle key={i} cx={sx(v)} cy={17} r={2.5} fill="var(--ink-3)" opacity={0.7} />)}
        <rect x={sx(median) - 1} y={6} width={2} height={22} fill="var(--ink-2)" />
        {highlight !== undefined && (
          <circle cx={sx(highlight)} cy={17} r={6} fill="var(--series-2)" stroke="var(--surface-1)" strokeWidth={2}
            onMouseEnter={() => setTip(`this: ${num(highlight, 2)}${unit} · median ${num(median, 2)}${unit}`)} />
        )}
      </svg>
      {tip && <div className="tip" style={{ left: 0, top: -28 }}>{tip}</div>}
    </div>
  );
}

export function Sparkline({ points, width = 100, height = 24 }: { points: number[]; width?: number; height?: number }) {
  if (points.length < 2) return null;
  const lo = Math.min(...points), hi = Math.max(...points);
  const d = points.map((v, i) => `${i ? "L" : "M"}${((i / (points.length - 1)) * width).toFixed(1)},${(height - 2 - ((v - lo) / (hi - lo || 1)) * (height - 4)).toFixed(1)}`).join("");
  return <svg width={width} height={height} aria-hidden><path d={d} fill="none" stroke="var(--series-1)" strokeWidth={2} /></svg>;
}
