"use client";
import { useState } from "react";
import { can, del, getUser, patch, post, usePoll } from "@/lib/api";
import { ago } from "@/lib/format";
import { Card, KV, Loading, PageHead, StatusBadge } from "@/components/ui";

function General() {
  const { data } = usePoll<any>("/settings", 0);
  const { data: lic } = usePoll<any>("/license", 0);
  const { data: info } = usePoll<any>("/system/info", 10000);
  if (!data) return <Loading />;
  const s = data.settings;
  const sec = data.security;
  return (
    <div className="grid g2">
      <Card title="Offline / on-premise status">
        <KV items={[
          ["Telemetry leaves site", data.offline_mode.telemetry_leaves_site ? <StatusBadge status="warning" label="Yes" /> : <StatusBadge status="good" label="No — fully on-prem" />],
          ["AI inference", `${data.offline_mode.llm_provider} (${data.offline_mode.llm_local ? "local" : "cloud"})`],
          ["Cloud LLM allowed", s.allow_cloud_llm ? "yes (explicitly enabled)" : "no (default)"],
          ["External SaaS notifications", s.allow_external_notifications ? "allowed" : "blocked (default)"],
          ["Database", info?.database], ["Telemetry source", s.telemetry_source], ["GPU vendor mapping", s.gpu_vendor],
        ]} />
      </Card>
      <Card title="Security posture">
        <KV items={[
          ["Secret encryption key", sec.secret_key_configured ? <StatusBadge status="good" label="configured" /> : <StatusBadge status="warning" label="derived (set SENTINEL_SECRET_KEY)" />],
          ["JWT secret", sec.default_jwt_secret ? <StatusBadge status="critical" label="DEFAULT — change it" /> : <StatusBadge status="good" label="custom" />],
          ["Bootstrap admin password", sec.default_admin_password ? <StatusBadge status="warning" label="default" /> : <StatusBadge status="good" label="changed" />],
          ["Session lifetime", `${s.jwt_ttl_minutes} min`],
        ]} />
      </Card>
      <Card title="Analysis & retention">
        <KV items={[
          ["Analysis interval", `${s.analysis_interval_s} s`], ["Peer group keys", s.peer_group_keys], ["Min peer group size", s.peer_min_group],
          ["Cycles before incident opens", s.incident_min_cycles], ["Quiet cycles before auto-resolve", s.incident_resolve_cycles],
          ["Sample persistence", `every ${s.persist_every_cycles} cycles`], ["Raw sample retention", `${s.retention_samples_days} days`],
          ["Incident retention", `${s.retention_incidents_days} days`], ["Audit retention", `${s.retention_audit_days} days`],
        ]} />
      </Card>
      <Card title="License">
        {lic && <KV items={[
          ["Customer", lic.license.customer], ["Plan", `${lic.license.plan} (${lic.license.license_type})`],
          ["Limits", `${lic.license.max_gpus ?? "∞"} GPUs · ${lic.license.max_nodes ?? "∞"} nodes`],
          ["Usage", `${lic.usage.gpus} GPUs · ${lic.usage.nodes} nodes`],
          ["Compliance", lic.compliant ? <StatusBadge status="good" label="compliant" /> : <StatusBadge status="warning" label={lic.warnings.join(" ")} />],
          ["Support tier", lic.license.support_tier], ["Enforcement", lic.enforcement],
        ]} />}
      </Card>
    </div>
  );
}

function Users() {
  const { data, reload } = usePoll<any[]>("/users", 0);
  const [f, setF] = useState({ username: "", password: "", role: "viewer" });
  const [err, setErr] = useState<string | null>(null);
  return (
    <Card title="Users & roles" flush>
      <div className="table-wrap"><table><thead><tr><th>User</th><th>Role</th><th>Active</th><th>Last login</th><th></th></tr></thead>
        <tbody>{data?.map((u) => (
          <tr key={u.id}><td>{u.username}</td>
            <td><select value={u.role} onChange={async (e) => { await patch(`/users/${u.id}`, { role: e.target.value }); reload(); }}>
              <option>viewer</option><option>operator</option><option>admin</option></select></td>
            <td>{u.is_active ? "yes" : "no"}</td><td className="muted">{ago(u.last_login)}</td>
            <td><button onClick={async () => { try { await patch(`/users/${u.id}`, { is_active: !u.is_active }); reload(); } catch (e: any) { setErr(e.message); } }}>{u.is_active ? "Disable" : "Enable"}</button></td></tr>))}
        </tbody></table></div>
      <div className="row" style={{ padding: 16 }}>
        <input placeholder="username" value={f.username} onChange={(e) => setF({ ...f, username: e.target.value })} />
        <input placeholder="password (≥10 chars)" type="password" value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} />
        <select value={f.role} onChange={(e) => setF({ ...f, role: e.target.value })}><option>viewer</option><option>operator</option><option>admin</option></select>
        <button className="primary" onClick={async () => { try { await post("/users", f); setF({ username: "", password: "", role: "viewer" }); setErr(null); reload(); } catch (e: any) { setErr(e.message); } }}>Add user</button>
        {err && <span className="err small">{err}</span>}
      </div>
      <div className="muted small" style={{ padding: "0 16px 16px" }}>viewer: read dashboards & Ask Sentinel · operator: + acknowledge/resolve incidents, demo injection · admin: + users, API keys, notifications, settings, audit, license.</div>
    </Card>
  );
}

function ApiKeys() {
  const { data, reload } = usePoll<any[]>("/api-keys", 0);
  const [name, setName] = useState("");
  const [role, setRole] = useState("viewer");
  const [created, setCreated] = useState<string | null>(null);
  return (
    <Card title="API keys" flush>
      <div className="table-wrap"><table><thead><tr><th>Name</th><th>Prefix</th><th>Role</th><th>Created</th><th>Last used</th><th></th></tr></thead>
        <tbody>{data?.map((k) => (
          <tr key={k.id}><td>{k.name}</td><td className="mono">{k.prefix}…</td><td>{k.role}</td><td className="muted">{ago(k.created_at)} by {k.created_by}</td>
            <td className="muted">{ago(k.last_used)}</td><td>{k.revoked ? <span className="muted">revoked</span> : <button onClick={async () => { await del(`/api-keys/${k.id}`); reload(); }}>Revoke</button>}</td></tr>))}
        </tbody></table></div>
      <div className="row" style={{ padding: 16 }}>
        <input placeholder="key name (e.g. grafana)" value={name} onChange={(e) => setName(e.target.value)} />
        <select value={role} onChange={(e) => setRole(e.target.value)}><option>viewer</option><option>operator</option><option>admin</option></select>
        <button className="primary" disabled={name.length < 2} onClick={async () => { const r = await post<any>("/api-keys", { name, role }); setCreated(r.key); setName(""); reload(); }}>Create key</button>
      </div>
      {created && <div className="banner warn" style={{ margin: "0 16px 16px" }}>Copy this key now — it is shown only once: <code>{created}</code><br />Use it as header <code>X-API-Key</code>.</div>}
    </Card>
  );
}

function Notifications() {
  const { data, reload } = usePoll<any[]>("/notification-channels", 0);
  const { data: log } = usePoll<any[]>("/notification-log?limit=30", 15000);
  const [f, setF] = useState({ name: "", type: "webhook", url: "", secret: "", to: "", host: "", min_severity: "warning" });
  const [res, setRes] = useState<string | null>(null);
  const create = async () => {
    const config: any = f.type === "email" ? { to: f.to.split(",").map((s) => s.trim()), host: f.host } : { url: f.url, ...(f.secret ? { secret: f.secret } : {}) };
    try { await post("/notification-channels", { name: f.name, type: f.type, config, min_severity: f.min_severity }); reload(); setRes(null); } catch (e: any) { setRes(e.message); }
  };
  return (
    <div className="grid g2">
      <Card title="Notification channels" flush>
        <div className="table-wrap"><table><thead><tr><th>Name</th><th>Type</th><th>Min severity</th><th>Config</th><th></th></tr></thead>
          <tbody>{data?.map((c) => (
            <tr key={c.id}><td>{c.name}</td><td>{c.type}{c.external_saas && <span className="pill">external</span>}</td><td>{c.min_severity}</td>
              <td className="mono small">{JSON.stringify(c.config)}</td>
              <td className="row"><button onClick={async () => { const r = await post<any>(`/notification-channels/${c.id}/test`); setRes(r.ok ? `Test to ${c.name} delivered.` : `Test failed: ${r.error}`); }}>Test</button>
                <button onClick={async () => { await del(`/notification-channels/${c.id}`); reload(); }}>Delete</button></td></tr>))}
          </tbody></table></div>
        <div className="grid g2" style={{ padding: 16 }}>
          <input placeholder="name" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} />
          <select value={f.type} onChange={(e) => setF({ ...f, type: e.target.value })}>
            <option value="webhook">Webhook (HMAC-signed)</option><option value="email">Email (SMTP)</option><option value="slack">Slack</option><option value="teams">Microsoft Teams</option><option value="pagerduty">PagerDuty</option></select>
          {f.type === "email" ? (<>
            <input placeholder="recipients, comma-separated" value={f.to} onChange={(e) => setF({ ...f, to: e.target.value })} />
            <input placeholder="SMTP host" value={f.host} onChange={(e) => setF({ ...f, host: e.target.value })} />
          </>) : (<>
            <input placeholder="URL" value={f.url} onChange={(e) => setF({ ...f, url: e.target.value })} />
            <input placeholder="signing secret (optional)" value={f.secret} onChange={(e) => setF({ ...f, secret: e.target.value })} />
          </>)}
          <select value={f.min_severity} onChange={(e) => setF({ ...f, min_severity: e.target.value })}><option value="warning">warning+</option><option value="critical">critical only</option></select>
          <button className="primary" onClick={create} disabled={!f.name}>Add channel</button>
        </div>
        {res && <div className="small" style={{ padding: "0 16px 16px" }}>{res}</div>}
        <div className="muted small" style={{ padding: "0 16px 16px" }}>Credentials are encrypted at rest. Slack/Teams/PagerDuty send data to third-party SaaS and stay blocked unless SENTINEL_ALLOW_EXTERNAL_NOTIFICATIONS=true.</div>
      </Card>
      <Card title="Delivery log" flush>
        <div className="table-wrap"><table><thead><tr><th>When</th><th>Channel</th><th>Incident</th><th>Event</th><th>Status</th></tr></thead>
          <tbody>{log?.map((l, i) => <tr key={i}><td className="muted">{ago(l.ts)}</td><td>{l.channel_type}</td><td>{l.incident_id}</td><td>{l.event}</td>
            <td title={l.error ?? ""}>{l.status}</td></tr>)}</tbody></table></div>
      </Card>
    </div>
  );
}

function Audit() {
  const { data } = usePoll<any[]>("/audit-logs?limit=300", 15000);
  return (
    <Card title="Audit log" flush>
      <div className="table-wrap"><table><thead><tr><th>Time</th><th>Actor</th><th>Role</th><th>Action</th><th>Resource</th><th>Status</th><th>IP</th></tr></thead>
        <tbody>{data?.map((a, i) => <tr key={i}><td className="muted">{new Date(a.ts).toLocaleString()}</td><td>{a.actor}</td><td>{a.role}</td><td>{a.action}</td>
          <td className="small">{a.resource}</td><td>{a.status_code}</td><td className="muted">{a.ip}</td></tr>)}</tbody></table></div>
    </Card>
  );
}

export default function Settings() {
  const user = getUser();
  const admin = can(user, "admin");
  const [tab, setTab] = useState("general");
  const tabs = [["general", "General"], ...(admin ? [["users", "Users"], ["keys", "API keys"], ["notify", "Notifications"], ["audit", "Audit log"]] : [])];
  return (
    <>
      <PageHead title="Settings" desc="Deployment, security, access control, notifications and licensing." />
      <div className="tabs">{tabs.map(([k, l]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}</div>
      {tab === "general" && <General />}
      {tab === "users" && <Users />}
      {tab === "keys" && <ApiKeys />}
      {tab === "notify" && <Notifications />}
      {tab === "audit" && <Audit />}
    </>
  );
}
