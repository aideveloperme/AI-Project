"use client";
/* Minimal, safe markdown renderer for copilot answers (no HTML injection):
 * headings-as-bold, **bold**, *italic*, `code`, lists, pipe tables, paragraphs. */
import React from "react";

function inline(text: string, key: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) out.push(<b key={`${key}-${i++}`}>{t.slice(2, -2)}</b>);
    else if (t.startsWith("`")) out.push(<code key={`${key}-${i++}`}>{t.slice(1, -1)}</code>);
    else out.push(<em key={`${key}-${i++}`}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.trim().startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().slice(1, -1).split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      blocks.push(
        <div className="table-wrap" key={i}><table><thead><tr>{rows[0].map((c, j) => <th key={j}>{inline(c, `h${j}`)}</th>)}</tr></thead>
          <tbody>{rows.slice(1).map((r, k) => <tr key={k}>{r.map((c, j) => <td key={j} className={j ? "num" : ""}>{inline(c, `c${k}${j}`)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (/^\s*[-*] /.test(l) || /^\s*\d+\. /.test(l)) {
      const ordered = /^\s*\d+\. /.test(l);
      const items: string[] = [];
      while (i < lines.length && (/^\s*[-*] /.test(lines[i]) || /^\s*\d+\. /.test(lines[i]))) {
        items.push(lines[i].replace(/^\s*([-*]|\d+\.) /, "")); i++;
      }
      const Tag = ordered ? "ol" : "ul";
      blocks.push(<Tag key={i}>{items.map((it, j) => <li key={j}>{inline(it, `li${i}${j}`)}</li>)}</Tag>);
      continue;
    }
    if (l.trim()) blocks.push(<p key={i}>{inline(l.replace(/^#+\s*/, ""), `p${i}`)}</p>);
    i++;
  }
  return <div className="md">{blocks}</div>;
}
