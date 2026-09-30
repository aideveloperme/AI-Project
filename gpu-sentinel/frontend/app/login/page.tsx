"use client";
import { useState } from "react";
import { post, setSession } from "@/lib/api";

export default function Login() {
  const [username, setU] = useState("admin");
  const [password, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await post<any>("/auth/login", { username, password });
      setSession(r.access_token, r.user);
      window.location.href = "/";
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 16 }}>
      <form onSubmit={submit} className="card" style={{ width: 360, display: "grid", gap: 14 }}>
        <div className="brand" style={{ padding: 0 }}><div className="logo">GS</div><div>GPU Sentinel AI<small>Sign in to your cluster</small></div></div>
        <label className="field">Username<input value={username} onChange={(e) => setU(e.target.value)} autoComplete="username" required /></label>
        <label className="field">Password<input type="password" value={password} onChange={(e) => setP(e.target.value)} autoComplete="current-password" required /></label>
        {err && <div className="err small">{err}</div>}
        <button className="primary" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        <div className="muted small">Demo accounts: admin / sentinel-admin · operator / sentinel-operator · viewer / sentinel-viewer</div>
      </form>
    </div>
  );
}
