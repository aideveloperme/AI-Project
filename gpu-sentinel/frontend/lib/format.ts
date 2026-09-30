export function num(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(a >= 100 ? 0 : digits);
}
export function pct(v: number | null | undefined, digits = 1, signed = true): string {
  if (v === null || v === undefined) return "—";
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
export function unit(u: string): string {
  return u && u !== "%" && u !== "°C" ? ` ${u}` : u;
}
export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
export function time(ms: number, seconds = false): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}) });
}
export const FAULT_LABELS: Record<string, string> = {
  thermal: "Thermal problem",
  power_cap: "Power cap",
  memory_bw: "HBM bandwidth degradation",
  clock_reduction: "Clock reduction",
  ecc_errors: "ECC errors",
  nvlink_degradation: "NVLink degradation",
  pcie_degradation: "PCIe degradation",
  cpu_bottleneck: "CPU bottleneck",
  host_memory_pressure: "Host memory pressure",
  network_degradation: "Network degradation",
  communication_bottleneck: "Communication (NCCL) bottleneck",
  app_regression: "Software regression",
};
