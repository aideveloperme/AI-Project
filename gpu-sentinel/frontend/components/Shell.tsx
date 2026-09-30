"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { api, clearSession, getUser, post, usePoll, type User } from "@/lib/api";
import { ago } from "@/lib/format";

const NAV: ([string, string, string] | null)[] = [
  ["/", "Overview", "◉"], ["/cluster/", "Cluster Health", "▦"], ["/fleet/", "GPU Fleet", "▤"],
  ["/performance/", "Performance", "↗"], ["/anomalies/", "Anomalies", "⚠"], ["/incidents/", "Incidents", "✚"],
  ["/rca/", "Root Cause Analysis", "⌖"], null,
  ["/workloads/", "Workloads", "⚙"], ["/network/", "Network", "⇄"], ["/trends/", "Historical Trends", "∿"], null,
  ["/copilot/", "AI Copilot", "✦"], ["/demo/", "Demo Control", "▶"], ["/settings/", "Settings", "☰"],
];

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    try {
      const t = (localStorage.getItem("sentinel.theme") as "dark" | "light") || "dark";
      setTheme(t);
      document.documentElement.dataset.theme = t;
    } catch {}
  }, []);
  const toggle = () => {
    const t = theme === "dark" ? "light" : "dark";
    setTheme(t);
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("sentinel.theme", t); } catch {}
  };
  return { theme, toggle };
}

export default function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const { theme, toggle } = useTheme();
  const isLogin = path.startsWith("/login");

  useEffect(() => {
    const u = getUser();
    setUser(u);
    setReady(true);
    if (!u && !isLogin) router.replace("/login/");
  }, [path, isLogin, router]);

  if (isLogin) return <>{children}</>;
  if (!ready || !user) return null;
  return <Authed user={user} path={path} theme={theme} toggle={toggle}>{children}</Authed>;
}

function Authed({ user, path, theme, toggle, children }: { user: User; path: string; theme: string; toggle: () => void; children: React.ReactNode }) {
  const { data: info } = usePoll<any>("/system/info", 15000);
  const { data: alerts, reload } = usePoll<any[]>("/alerts?unread_only=true&limit=20", 10000);
  const [open, setOpen] = useState(false);
  const unread = alerts?.length ?? 0;
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")));
  return (
    <div className="app">
      <aside className="side">
        <div className="brand"><div className="logo">GS</div><div>GPU Sentinel AI<small>Performance & Health Intelligence</small></div></div>
        <nav className="nav">
          {NAV.map((n, i) => n === null ? <div className="sep" key={i} /> : (
            <Link key={n[0]} href={n[0]} className={active(n[0]) ? "active" : ""}>
              <span aria-hidden style={{ width: 16, textAlign: "center" }}>{n[2]}</span>{n[1]}
            </Link>
          ))}
        </nav>
        <div className="muted small" style={{ padding: "18px 10px 0" }}>
          {info && (<>
            <div>Source: <b className="ink2">{info.telemetry_source}</b></div>
            <div>AI: <b className="ink2">{info.llm_provider}</b> {info.llm_local ? "(local)" : "(cloud)"}</div>
            <div>Cycle #{info.cycle} · {info.last_cycle_ms} ms</div>
          </>)}
        </div>
      </aside>
      <div className="main">
        <header className="top">
          {info?.demo_mode && <span className="pill" style={{ background: "var(--series-4)", color: "#000" }}>DEMO MODE</span>}
          {info?.last_error && <span className="err small">Engine: {info.last_error}</span>}
          <div className="spacer" />
          <div style={{ position: "relative" }}>
            <button onClick={() => setOpen(!open)} aria-label="Alerts">🔔 Alerts {unread > 0 && <span className="count" style={{ background: "var(--critical)", color: "#fff", borderRadius: 9, padding: "0 6px", marginLeft: 4 }}>{unread}</span>}</button>
            {open && (
              <div className="card" style={{ position: "absolute", right: 0, top: 38, width: 380, maxHeight: 420, overflowY: "auto", zIndex: 20 }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
                  <b>Dashboard alerts</b>
                  <button onClick={async () => { await post("/alerts/read"); reload(); }}>Mark all read</button>
                </div>
                {unread === 0 && <div className="muted small">No unread alerts.</div>}
                {alerts?.map((a) => (
                  <div key={a.id} style={{ padding: "6px 0", borderTop: "1px solid var(--border)" }}>
                    <Link href={`/incident/?id=${a.incident_id}`} onClick={() => setOpen(false)}>{a.incident_id}</Link>{" "}
                    <span className="pill">{a.event}</span> <span className="muted small">{ago(a.ts)}</span>
                    <div className="small">{a.message}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button onClick={toggle} aria-label="Toggle theme">{theme === "dark" ? "☀ Light" : "☾ Dark"}</button>
          <span className="small ink2">{user.username} · <b>{user.role}</b></span>
          <button onClick={() => { clearSession(); window.location.href = "/login/"; }}>Sign out</button>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
