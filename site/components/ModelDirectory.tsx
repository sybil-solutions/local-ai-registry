"use client";
import { useId, useMemo, useState } from "react";
import Link from "next/link";
import type { ModelDirectoryRow } from "@/lib/configurations";
import s from "./RegistryTables.module.css";

type SortKey = "name" | "family" | "released" | "hardware" | "configurations" | "capabilities";
export default function ModelDirectory({ rows }: { rows: ModelDirectoryRow[] }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "name", descending: false });
  const visible = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return rows.filter((r) => {
      const text = `${r.name} ${r.id} ${r.family} ${r.about} ${r.capabilities} ${r.capabilities.includes("Vision") ? "images" : ""} ${r.capabilities.includes("Reasoning") ? "thinking" : ""}`.toLowerCase();
      return terms.every((term) => text.includes(term));
    }).sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      if (av === "" || bv === "") return av === "" ? bv === "" ? a.id.localeCompare(b.id) : 1 : -1;
      const order = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
      return (sort.descending ? -order : order) || a.id.localeCompare(b.id);
    });
  }, [rows, query, sort]);
  const header = (key: SortKey, label: string) => <th scope="col" aria-sort={sort.key === key ? sort.descending ? "descending" : "ascending" : "none"}><button onClick={() => setSort((current) => ({ key, descending: current.key === key ? !current.descending : ["released", "hardware", "configurations"].includes(key) }))}>{label}{sort.key === key ? sort.descending ? " ↓" : " ↑" : ""}</button></th>;
  return <div className={s.tableArea}>
    <div className={s.toolbar}><label htmlFor={id}>Search models<input id={id} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, family, use case, vision, reasoning…" /></label><span className="dim" aria-live="polite">{visible.length} of {rows.length} models</span></div>
    <div className={s.scroll}><table className={`t ${s.directory}`}><thead><tr>{header("name", "Model")}{header("family", "Family")}{header("released", "Released")}{header("hardware", "Hardware")}{header("configurations", "Configurations")}{header("capabilities", "Capabilities")}</tr></thead><tbody>{visible.map((r) => <tr key={r.id} id={r.id} data-model={r.id}><td><Link href={`/models/${r.id}/`}>{r.name}</Link>{r.about && <small>{r.about}</small>}</td><td>{r.family}</td><td>{r.released || "—"}</td><td>{r.hardware}</td><td>{r.configurations}</td><td>{r.capabilities}</td></tr>)}</tbody></table></div>
    {!visible.length && <p className="dim">No models match “{query}”.</p>}
  </div>;
}
