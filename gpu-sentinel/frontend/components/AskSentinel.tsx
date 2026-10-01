"use client";
/* Ask Sentinel: GPU Sentinel's own chat over live telemetry and incidents.
 * Runs entirely on the customer's server (backend /api/v1/ask); optional local LLM. */
import { useEffect, useRef, useState } from "react";
import { post, usePoll } from "@/lib/api";
import { Markdown } from "@/components/Markdown";

type Msg = { role: "user" | "bot"; text: string; meta?: any };

export function AskSentinel({ compact = false }: { compact?: boolean }) {
  const { data: sugg } = usePoll<string[]>("/ask/suggestions", 60000);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { if (!compact) end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, compact]);

  const ask = async (question: string) => {
    if (!question.trim()) return;
    setMsgs((m) => [...(compact ? [] : m), { role: "user", text: question }]);
    setQ("");
    setBusy(true);
    try {
      const r = await post<any>("/ask", { question });
      setMsgs((m) => [...m, { role: "bot", text: r.answer, meta: r }]);
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "bot", text: `Error: ${e.message}` }]);
    } finally { setBusy(false); }
  };

  const chips = (compact ? ["How is the cluster?", "Which GPUs are abnormal?", "What changed during the last hour?"] : sugg) ?? [];
  return (
    <div className="chat">
      {(msgs.length === 0 || compact) && (
        <div className="row">{chips.map((s) => <button key={s} onClick={() => ask(s)} disabled={busy}>{s}</button>)}</div>)}
      {msgs.map((m, i) => (
        <div key={i} className={`msg ${m.role}`} style={compact ? { maxWidth: "none" } : undefined}>
          {m.role === "bot" ? <Markdown text={m.text} /> : m.text}
          {m.meta && <div className="muted small" style={{ marginTop: 8 }}>
            Answered from live telemetry via <code>{m.meta.tools?.[0]?.name}</code> · wording: {m.meta.provider === "template" ? "built-in engine" : m.meta.provider}
          </div>}
        </div>))}
      {busy && <div className="msg bot muted">Checking live telemetry…</div>}
      <div ref={end} />
      <form onSubmit={(e) => { e.preventDefault(); ask(q); }} className="row" style={compact ? {} : { position: "sticky", bottom: 12 }}>
        <input style={{ flex: 1, padding: 10 }} placeholder="Ask about your cluster, e.g. How is the cluster?" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="primary" disabled={busy}>Ask</button>
      </form>
    </div>
  );
}
