"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import Logo from "./Logo";
import type { ModelDirectoryRow } from "@/lib/configurations";
import { SortControl, compareValues, type Column, type Sort } from "./cardSort";
import s from "./RegistryCards.module.css";

type SortKey = "name" | "family" | "released" | "hardware" | "configurations" | "capabilities";
const COLUMNS: Column<SortKey>[] = [
  { key: "name", label: "Model" },
  { key: "family", label: "Family" },
  { key: "released", label: "Released" },
  { key: "hardware", label: "Hardware", numeric: true },
  { key: "configurations", label: "Configurations", numeric: true },
  { key: "capabilities", label: "Capabilities" },
];

export default function ModelDirectory({ rows }: { rows: ModelDirectoryRow[] }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort<SortKey>>({ key: "name", descending: false });
  const visible = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return rows.filter((r) => {
      const text = `${r.name} ${r.id} ${r.family} ${r.about} ${r.capabilities} ${r.capabilities.includes("Vision") ? "images" : ""} ${r.capabilities.includes("Reasoning") ? "thinking" : ""}`.toLowerCase();
      return terms.every((term) => text.includes(term));
    }).sort((a, b) => compareValues(a[sort.key], b[sort.key], sort.descending) || a.id.localeCompare(b.id));
  }, [rows, query, sort]);
  return <div className={s.area}>
    <div className={s.toolbar}>
      <label className={s.search} htmlFor={id}>Search models<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, family, use case, vision, reasoning…" /></label>
      <SortControl columns={COLUMNS} sort={sort} setSort={setSort} />
      <span className={s.count} aria-live="polite"><b>{visible.length}</b> of {rows.length} models</span>
    </div>
    <div className={`grid cols-2 ${s.cards}`}>
      {visible.map((r) => <article key={r.id} id={r.id} data-model={r.id} className={`rc ${s.card}`}>
        <div className={`head ${s.identity}`}><Logo family={r.family} /><Link href={`/models/${r.id}/`}>{r.name}</Link></div>
        <div className="meta"><span>{r.family || "Family not recorded"}</span><span>{r.released ? <>Released <time dateTime={r.released}>{r.released}</time></> : "Release date not recorded"}</span></div>
        {r.about && <p className="about">{r.about}</p>}
        <div className="chips">{r.capabilities.split(" · ").map((capability) => <span key={capability} className="chip on">{capability}</span>)}</div>
        <div className="meta"><span>Runs on {r.hardware} hardware {r.hardware === 1 ? "type" : "types"}</span><span>{r.configurations} configurations</span></div>
        <div className="actions"><Link className="btn" href={`/models/${r.id}/`}>View model ›</Link></div>
      </article>)}
    </div>
    {!visible.length && <p className={s.empty}>No models match “{query}”.</p>}
  </div>;
}
