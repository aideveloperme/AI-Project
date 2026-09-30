"use client";
import { useEffect, useRef, useState } from "react";
import { post, usePoll } from "@/lib/api";
import { PageHead } from "@/components/ui";
import { Markdown } from "@/components/Markdown";

type Msg = { role: "user" | "bot"; text: string; meta?: any };

export default function Copilot() {
  const { data: sugg } = usePoll<string[]>("/copilot/suggestions", 30000);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);
  const ask = async (question: string) => {
    if (!question.trim()) return;
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setQ("");
    setBusy(true);
    try {
      const r = await post<any>("/copilot/ask", { question });
      setMsgs((m) => [...m, { role: "bot", text: r.answer, meta: r }]);
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "bot", text: `Error: ${e.message}` }]);
    } finally { setBusy(false); }
  };
  return (
    <>
      <PageHead title="AI Copilot" desc="Ask about your cluster in plain language. Questions are answered with read-only tools over live telemetry and incidents — never by guessing, and never with shell access." />
      <div className="chat">
        {msgs.length === 0 && (
          <div className="card"><h3>Try asking</h3><div className="row">{sugg?.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div></div>)}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.role === "bot" ? <Markdown text={m.text} /> : m.text}
            {m.meta && <div className="muted small" style={{ marginTop: 8 }}>
              Tool: <code>{m.meta.tools?.[0]?.name}({JSON.stringify(m.meta.tools?.[0]?.args)})</code> · routing: {m.meta.routing} · answer by: {m.meta.provider}
            </div>}
          </div>))}
        {busy && <div className="msg bot muted">Querying telemetry…</div>}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); ask(q); }} className="row" style={{ marginTop: 16, position: "sticky", bottom: 12 }}>
        <input style={{ flex: 1, padding: 10 }} placeholder="e.g. Why is gpu-04 slow?" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="primary" disabled={busy}>Ask</button>
      </form>
    </>
  );
}
