"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import VendorLogo from "./VendorLogo";
import type { HardwareDirectoryRow } from "@/lib/configurations";
import { SortControl, compareValues, type Column, type Sort } from "./cardSort";
import s from "./RegistryCards.module.css";

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
  return <div className={s.area}>
    <div className={s.toolbar}>
      <label className={s.search} htmlFor={id}>Search hardware<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, manufacturer, memory, specs only…" /></label>
      <SortControl columns={COLUMNS} sort={sort} setSort={setSort} />
      <span className={s.count} aria-live="polite"><b>{visible.length}</b> of {rows.length} hardware entries</span>
    </div>
    <div className={`grid cols-2 ${s.cards}`}>
      {visible.map((r) => <article key={r.id} data-hardware={r.id} className={`rc ${s.card}`}>
        <div className={`head ${s.identity}`}><VendorLogo vendor={r.vendor} /><Link href={`/gpu/${r.id}/`}>{r.name}</Link></div>
        <div className="meta"><span>{VENDORS[r.vendor] ?? r.vendor}</span><span>{r.models ? `${r.models} models · ${r.configurations} setups` : "Specifications only · no recorded recipe"}</span></div>
        <div className={`numbers ${s.metrics}`}>
          <div><b>{r.memory || "System RAM"}</b><span>memory GB</span></div>
          <div><b>{r.bandwidth == null ? "—" : r.bandwidth.toLocaleString("en-US")}</b><span>bandwidth GB/s</span></div>
          <div><b data-speed="decode">{r.decode == null ? "—" : r.decode.toLocaleString("en-US", { maximumFractionDigits: 1 })}</b><span>max decode tok/s</span></div>
        </div>
        <div className="actions"><Link className="btn" href={`/gpu/${r.id}/`}>{r.models ? "View models and recipes ›" : "View specifications ›"}</Link></div>
      </article>)}
    </div>
    {!visible.length && <p className={s.empty}>No hardware matches “{query}”.</p>}
    <p className={s.note}>Maximum decode is the highest recorded recipe measurement for any model and topology on this hardware, not a like-for-like ranking. Open the hardware page to compare exact models, card counts, host requirements and evidence. Manufacturer logos identify the cards; original product photos and their source attribution remain on hardware detail pages.</p>
  </div>;
}
