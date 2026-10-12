"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import HardwarePhoto from "./HardwarePhoto";
import type { HardwareDirectoryRow } from "@/lib/configurations";
import s from "./RegistryTables.module.css";

type SortKey = "name" | "vendor" | "memory" | "bandwidth" | "models" | "configurations" | "decode";
const VENDORS: Record<string, string> = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel", apple: "Apple", cpu: "CPU" };
export default function HardwareDirectory({ rows }: { rows: HardwareDirectoryRow[] }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "name", descending: false });
  const visible = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return rows.filter((r) => terms.every((term) => `${r.name} ${r.id} ${r.vendor} ${r.memory}gb ${r.models ? "recipes" : "specs only"}`.toLowerCase().includes(term))).sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      if (av == null || bv == null) return av == null ? bv == null ? a.id.localeCompare(b.id) : 1 : -1;
      const order = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
      return (sort.descending ? -order : order) || a.id.localeCompare(b.id);
    });
  }, [rows, query, sort]);
  const header = (key: SortKey, label: string) => <th scope="col" aria-sort={sort.key === key ? sort.descending ? "descending" : "ascending" : "none"}><button onClick={() => setSort((current) => ({ key, descending: current.key === key ? !current.descending : !["name", "vendor"].includes(key) }))}>{label}{sort.key === key ? sort.descending ? " ↓" : " ↑" : ""}</button></th>;
  return <div className={s.tableArea}>
    <div className={s.toolbar}><label htmlFor={id}>Search hardware<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, manufacturer, memory, specs only…" /></label><span className="dim" aria-live="polite">{visible.length} of {rows.length} hardware entries</span></div>
    <div className={s.scroll}><table className={`t ${s.directory}`}><thead><tr>{header("name", "Hardware")}{header("vendor", "Vendor")}{header("memory", "Memory GB")}{header("bandwidth", "Bandwidth GB/s")}{header("models", "Models")}{header("configurations", "Configurations")}{header("decode", "Max decode tok/s")}</tr></thead><tbody>{visible.map((r) => <tr key={r.id} data-hardware={r.id}><td><div className={s.photoName}><HardwarePhoto id={r.id} compact /><div><Link href={`/gpu/${r.id}/`}>{r.name}</Link>{!r.models && <small>Specifications only · no recorded recipe</small>}</div></div></td><td>{VENDORS[r.vendor] ?? r.vendor}</td><td>{r.memory || "System RAM"}</td><td>{r.bandwidth == null ? "—" : r.bandwidth.toLocaleString("en-US")}</td><td>{r.models}</td><td>{r.configurations}</td><td>{r.decode == null ? "—" : r.decode.toLocaleString("en-US", { maximumFractionDigits: 1 })}</td></tr>)}</tbody></table></div>
    {!visible.length && <p className="dim">No hardware matches “{query}”.</p>}
    <p className={s.note}>Maximum decode is the highest recorded recipe measurement for any model and topology on this hardware, not a like-for-like ranking. Open the hardware page to compare exact models, card counts, host requirements and evidence. Photos link to their original manufacturer source; representative images are labelled.</p>
  </div>;
}
