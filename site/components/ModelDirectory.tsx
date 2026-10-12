"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import type { ModelDirectoryRow } from "@/lib/configurations";
import { SortControl, SortHeader, compareValues, type Column, type Sort } from "./tableSort";
import s from "./RegistryTables.module.css";

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
  return <div className={s.tableArea}>
    <div className={s.toolbar}>
      <label className={s.search} htmlFor={id}>Search models<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, family, use case, vision, reasoning…" /></label>
      <SortControl columns={COLUMNS} sort={sort} setSort={setSort} />
      <span className={s.count} aria-live="polite"><b>{visible.length}</b> of {rows.length} models</span>
    </div>
    <div className={s.frame}>
      <table className={`${s.table} ${s.directory}`}>
        <thead><tr>{COLUMNS.map((column) => <SortHeader key={column.key} column={column} sort={sort} setSort={setSort} />)}</tr></thead>
        <tbody>{visible.map((r) => <tr key={r.id} id={r.id} data-model={r.id}>
          <td className={s.identity}><Link className={s.primary} href={`/models/${r.id}/`}>{r.name}</Link>{r.about && <span className={s.meta}>{r.about}</span>}</td>
          <td data-label="Family" className={s.dimCell}>{r.family || <span className={s.unknown}>—</span>}</td>
          <td data-label="Released" className={s.dimCell}>{r.released || <span className={s.unknown}>—</span>}</td>
          <td data-label="Hardware" className={s.num}>{r.hardware}</td>
          <td data-label="Configurations" className={s.num}>{r.configurations}</td>
          <td data-label="Capabilities" className={s.dimCell}>{r.capabilities}</td>
        </tr>)}</tbody>
      </table>
      {!visible.length && <p className={s.empty}>No models match “{query}”.</p>}
    </div>
  </div>;
}
