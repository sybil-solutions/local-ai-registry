"use client";

import { useId, useMemo, useState } from "react";
import Link from "next/link";
import Copy from "./Copy";
import Logo from "./Logo";
import type { ConfigurationRow, ConfigurationVariant } from "@/lib/configurations";
import { SortControl, compareValues, type Column, type Sort } from "./cardSort";
import s from "./RegistryCards.module.css";

type SortKey = "hardware" | "model" | "memory" | "label" | "context" | "decode" | "prefill" | "status";
type Entry = { row: ConfigurationRow; variants: ConfigurationVariant[]; current: ConfigurationVariant };
const COMMON: Column<SortKey>[] = [
  { key: "memory", label: "Total memory", unit: "GB", numeric: true },
  { key: "label", label: "Recipe" }, { key: "context", label: "Context", numeric: true },
  { key: "decode", label: "Decode", unit: "tok/s", numeric: true },
  { key: "prefill", label: "Prefill", unit: "tok/s", numeric: true },
  { key: "status", label: "Evidence" },
];
const HARDWARE: Column<SortKey>[] = [{ key: "hardware", label: "Hardware setup" }, ...COMMON];
const MODELS: Column<SortKey>[] = [{ key: "model", label: "Model" }, ...HARDWARE];
const ENGINE: Record<string, string> = { tabbyapi: "TabbyAPI", sglang: "SGLang", vllm: "vLLM", exllamav3: "ExLlamaV3", flm: "FastFlowLM" };
const setupName = (row: ConfigurationRow) => row.machines > 1 ? `${row.machines} × ${row.cardName}` : row.cards > 1 ? `${row.cards} × ${row.cardName}` : row.cardName;
const measurement = (value: number | null) => value == null ? "—" : value.toLocaleString("en-US", { maximumFractionDigits: 1 });

export default function ConfigurationCards({ rows, showModel = false }: { rows: ConfigurationRow[]; showModel?: boolean }) {
  const searchId = useId();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [sort, setSort] = useState<Sort<SortKey>>({ key: showModel ? "model" : "hardware", descending: false });
  const visible = useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const result: Entry[] = [];
    for (const row of rows) {
      const common = `${row.hardware} ${row.hardwareId} ${row.model} ${row.modelId} ${row.family} ${row.memory}gb ${row.cards} cards ${row.machines} machines ${row.host}`;
      const variants = row.variants.filter((v) => {
        const text = `${common} ${v.label} ${v.key} ${v.context} ${v.context.toLocaleString("en-US")} context ${v.status} ${v.requirements}`.toLowerCase();
        return terms.every((term) => text.includes(term));
      });
      if (variants.length) result.push({ row, variants, current: variants.find((v) => v.key === selected[row.id]) ?? variants[0] });
    }
    const value = (entry: Entry) => sort.key === "hardware" || sort.key === "model" || sort.key === "memory" ? entry.row[sort.key] : entry.current[sort.key];
    return result.sort((a, b) => compareValues(value(a), value(b), sort.descending) || a.row.id.localeCompare(b.row.id));
  }, [rows, query, selected, sort]);
  return <div className={s.area}>
    <div className={s.toolbar}>
      <label className={s.search} htmlFor={searchId}>Search configurations<input id={searchId} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Hardware, model, engine, format, host RAM…" /></label>
      <SortControl columns={showModel ? MODELS : HARDWARE} sort={sort} setSort={setSort} />
      <span className={s.count} aria-live="polite">{visible.length} configurations · {visible.reduce((n, entry) => n + entry.variants.length, 0)} recipes</span>
    </div>
    <div className={`grid cols-2 ${s.cards}`}>
      {visible.map(({ row, variants, current }) => {
        const expanded = !!open[row.id], name = setupName(row);
        const commandId = `${searchId}-command-${row.id.replace(/[^a-z0-9]+/gi, "-")}`;
        return <article key={row.id} className={`rc ${s.card}`} data-configuration={row.id} data-recipe={current.key} data-variant-count={variants.length}>
          <div className={`head ${s.identity}`}>
            {showModel && <Logo family={row.family} />}
            <Link href={showModel ? `/models/${row.modelId}/` : `/gpu/${row.hardwareId}/`}>{showModel ? row.model : name}</Link>
          </div>
          <div className="meta">
            {showModel && <Link href={`/gpu/${row.hardwareId}/`}>{name}</Link>}
            <span>{row.memory ? `${row.memory} GB${row.cards * row.machines > 1 ? " total" : ""}` : "System RAM"}</span>
            <span>{row.cards} card{row.cards === 1 ? "" : "s"} per machine · {row.machines} machine{row.machines === 1 ? "" : "s"}</span>
          </div>
          {row.needs.length > 0 && <dl className={s.needs}>{row.needs.map((need) => <div key={need.label}><dt>{need.label}</dt><dd>{need.value}</dd></div>)}</dl>}
          {variants.length > 1 ? <label className={s.recipeField}>Exact recipe<select aria-label={`Recipe for ${row.model} on ${row.hardware}`} value={current.key} onChange={(e) => setSelected((chosen) => ({ ...chosen, [row.id]: e.target.value }))}>{variants.map((v) => <option key={v.key} value={v.key}>{v.label}</option>)}</select></label> : <div className="meta"><span>{ENGINE[current.engine] ?? current.engine}</span><span>{current.format || "Format not recorded"}</span></div>}
          <div className={`numbers ${s.metrics}`}>
            <div><b data-speed="decode" title={current.decode == null ? "No measurement recorded" : undefined}>{measurement(current.decode)}</b><span>decode tok/s</span></div>
            <div><b data-speed="prefill" title={current.prefill == null ? "No measurement recorded" : undefined}>{measurement(current.prefill)}</b><span>prefill tok/s</span></div>
            <div><b>{current.context >= 1024 ? `${Math.round(current.context / 1024)}K` : current.context}</b><span>context</span></div>
          </div>
          <div className="proof" title={current.statusDetail}><i className={`dot ${current.statusTone === "ok" ? "" : current.statusTone === "warm" ? "warm" : "dim"}`} /><span>{current.status}{current.testedAt ? ` · ${current.testedAt}` : " · Date not recorded"}</span></div>
          <div className="actions"><button type="button" className="btn solid" aria-expanded={expanded} aria-controls={commandId} onClick={() => setOpen((state) => ({ ...state, [row.id]: !state[row.id] }))}>{expanded ? "Hide command" : "Command"}</button><Link className="btn" href={current.href}>Full recipe ›</Link></div>
          {expanded && <div className={s.command} id={commandId}>
            <div className={s.commandHead}><span>Exact command</span><Copy text={current.command} event="command_copied" props={{ recipe: current.key, from: "configuration_cards" }} /></div>
            <p>Requires the hardware above and the recipe’s engine prerequisites.{current.machines > 1 ? ` Run on each of the ${current.machines} machines with its documented rank and network settings.` : ""}{row.host ? ` Host: ${row.host}.` : ""}</p>
            {current.requirements && <p>Setup required: {current.requirements}</p>}
            <pre className="code" data-command-for={current.key}>{current.command}</pre>
          </div>}
        </article>;
      })}
    </div>
    {!visible.length && <p className={s.empty}>No configurations match “{query}”. Try another hardware name, engine or model.</p>}
    <p className={s.note}>Speeds are the selected recipe’s recorded measurements, not estimates; “—” means unmeasured. Evidence distinguishes checks on this card, sibling-card checks, earlier checks and publisher reports.</p>
  </div>;
}
