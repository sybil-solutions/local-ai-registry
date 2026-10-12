"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import HardwarePhoto from "./HardwarePhoto";
import type { HardwareDirectoryRow } from "@/lib/configurations";
import { SortControl, SortHeader, compareValues, type Column, type Sort } from "./tableSort";
import s from "./RegistryTables.module.css";

type SortKey = "name" | "vendor" | "memory" | "bandwidth" | "models" | "configurations" | "decode";
const VENDORS: Record<string, string> = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel", apple: "Apple", cpu: "CPU" };
const COLUMNS: Column<SortKey>[] = [
  { key: "name", label: "Hardware" },
  { key: "vendor", label: "Vendor" },
  { key: "memory", label: "Memory", unit: "GB", numeric: true },
  { key: "bandwidth", label: "Bandwidth", unit: "GB/s", numeric: true },
  { key: "models", label: "Models", numeric: true },
  { key: "configurations", label: "Setups", numeric: true },
  { key: "decode", label: "Max decode", unit: "tok/s", numeric: true },
];

export default function HardwareDirectory({ rows }: { rows: HardwareDirectoryRow[] }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort<SortKey>>({ key: "name", descending: false });
  const visible = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return rows.filter((r) => terms.every((term) => `${r.name} ${r.id} ${r.vendor} ${r.memory}gb ${r.models ? "recipes" : "specs only"}`.toLowerCase().includes(term)))
      .sort((a, b) => compareValues(a[sort.key], b[sort.key], sort.descending) || a.id.localeCompare(b.id));
  }, [rows, query, sort]);
  return <div className={s.tableArea}>
    <div className={s.toolbar}>
      <label className={s.search} htmlFor={id}>Search hardware<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, manufacturer, memory, specs only…" /></label>
      <SortControl columns={COLUMNS} sort={sort} setSort={setSort} />
      <span className={s.count} aria-live="polite"><b>{visible.length}</b> of {rows.length} hardware entries</span>
    </div>
    <div className={s.frame}>
      <table className={`${s.table} ${s.directory} ${s.photos}`}>
        <thead><tr>{COLUMNS.map((column) => <SortHeader key={column.key} column={column} sort={sort} setSort={setSort} />)}</tr></thead>
        <tbody>{visible.map((r) => <tr key={r.id} data-hardware={r.id}>
          <td className={s.identity}><div className={s.photoName}><HardwarePhoto id={r.id} compact /><div><Link className={s.primary} href={`/gpu/${r.id}/`}>{r.name}</Link>{!r.models && <span className={s.meta}>Specifications only · no recorded recipe</span>}</div></div></td>
          <td data-label="Vendor" className={s.dimCell}>{VENDORS[r.vendor] ?? r.vendor}</td>
          <td data-label="Memory GB" className={s.num}>{r.memory || "System RAM"}</td>
          <td data-label="Bandwidth GB/s" className={s.num}>{r.bandwidth == null ? <span className={s.unknown}>—</span> : r.bandwidth.toLocaleString("en-US")}</td>
          <td data-label="Models" className={s.num}>{r.models}</td>
          <td data-label="Setups" className={s.num}>{r.configurations}</td>
          <td data-label="Max decode tok/s" className={s.num}>{r.decode == null ? <span className={s.unknown}>—</span> : <span className={s.strong}>{r.decode.toLocaleString("en-US", { maximumFractionDigits: 1 })}</span>}</td>
        </tr>)}</tbody>
      </table>
      {!visible.length && <p className={s.empty}>No hardware matches “{query}”.</p>}
    </div>
    <p className={s.note}>Maximum decode is the highest recorded recipe measurement for any model and topology on this hardware, not a like-for-like ranking. Open the hardware page to compare exact models, card counts, host requirements and evidence. Photos link to their original manufacturer source; representative images are labelled.</p>
  </div>;
}
