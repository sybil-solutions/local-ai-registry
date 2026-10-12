"use client";

import { useId, useMemo, useState } from "react";
import Link from "next/link";
import Copy from "./Copy";
import type { ConfigurationRow, ConfigurationVariant } from "@/lib/configurations";
import s from "./RegistryTables.module.css";

type SortKey = "hardware" | "model" | "memory" | "label" | "context" | "decode" | "prefill" | "status";
const SORT_LABELS: Record<SortKey, string> = { hardware: "Hardware", model: "Model", memory: "Memory", label: "Recipe", context: "Context", decode: "Decode speed", prefill: "Prefill speed", status: "Evidence" };
const EVIDENCE: Record<string, string> = { "Tested on this card": "Tested", "Tested on a sibling": "Sibling test", "Earlier check": "Earlier check" };

export default function ConfigurationTable({ rows, showModel = false }: { rows: ConfigurationRow[]; showModel?: boolean }) {
  const searchId = useId();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "hardware", descending: false });
  const visible = useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const result: { row: ConfigurationRow; variants: ConfigurationVariant[]; current: ConfigurationVariant }[] = [];
    for (const row of rows) {
      const common = `${row.hardware} ${row.hardwareId} ${row.model} ${row.modelId} ${row.memory}gb ${row.cards} cards ${row.machines} machines ${row.host}`;
      const variants = row.variants.filter((v) => {
        const text = `${common} ${v.label} ${v.key} ${v.context} context ${v.status} ${v.requirements}`.toLowerCase();
        return terms.every((term) => text.includes(term));
      });
      if (!variants.length) continue;
      result.push({ row, variants, current: variants.find((v) => v.key === selected[row.id]) ?? variants[0] });
    }
    return result.sort((a, b) => {
      const av = sort.key in a.row ? a.row[sort.key as "hardware" | "model" | "memory"] : a.current[sort.key as "label" | "context" | "decode" | "prefill" | "status"];
      const bv = sort.key in b.row ? b.row[sort.key as "hardware" | "model" | "memory"] : b.current[sort.key as "label" | "context" | "decode" | "prefill" | "status"];
      if (av == null || bv == null) return av == null ? bv == null ? a.row.id.localeCompare(b.row.id) : 1 : -1;
      const order = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
      return (sort.descending ? -order : order) || a.row.id.localeCompare(b.row.id);
    });
  }, [rows, query, selected, sort]);
  const header = (key: SortKey, text: string) => <th scope="col" aria-sort={sort.key === key ? sort.descending ? "descending" : "ascending" : "none"}>{text}</th>;
  const columns = showModel ? 5 : 4;
  return (
    <div className={s.tableArea}>
      <div className={`${s.toolbar} ${s.configToolbar}`}>
        <label htmlFor={searchId}>Search configurations<input id={searchId} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Hardware, engine, format…" /></label>
        <div className={s.sortControls}>
          <select aria-label="Sort configurations" value={sort.key} onChange={(e) => setSort({ key: e.target.value as SortKey, descending: ["memory", "context", "decode", "prefill"].includes(e.target.value) })}>{Object.entries(SORT_LABELS).filter(([key]) => showModel || key !== "model").map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
          <button className="btn" aria-label={`Sort ${sort.descending ? "ascending" : "descending"}`} onClick={() => setSort((current) => ({ ...current, descending: !current.descending }))}>{sort.descending ? "↓" : "↑"}</button>
        </div>
        <span className="dim" aria-live="polite">{visible.length} setups · {visible.reduce((n, entry) => n + entry.variants.length, 0)} recipes</span>
      </div>
      <div className={s.scroll}>
        <table className={`t ${s.configurations}`}>
          <thead><tr>{header("hardware", "Hardware")}{showModel && header("model", "Model")}{header("label", "Recipe")}{header("decode", "Speed")}<th scope="col"><span className={s.runLabel}>Run</span></th></tr></thead>
          <tbody>{visible.flatMap(({ row, variants, current }) => [
            <tr key={row.id} data-configuration={row.id} data-recipe={current.key} data-variant-count={variants.length}>
              <td><Link href={`/gpu/${row.hardwareId}/`}>{row.hardware}</Link><small>{row.memory ? `${row.memory} GB${row.cards * row.machines > 1 ? " total" : ""}` : "System RAM"}{row.machines > 1 ? ` · ${row.machines} nodes` : ""}</small>{row.host && <small className={s.host} title="Minimum free host RAM, local weight storage and logical CPU requirements">{row.host}</small>}</td>
              {showModel && <td><Link href={`/models/${row.modelId}/`}>{row.model}</Link></td>}
              <td>{variants.length > 1 ? <select aria-label={`Recipe for ${row.model} on ${row.hardware}`} value={current.key} onChange={(e) => setSelected((chosen) => ({ ...chosen, [row.id]: e.target.value }))}>{variants.map((v) => <option key={v.key} value={v.key}>{v.label}</option>)}</select> : <span className={s.recipeLabel} title={current.label}>{current.label}</span>}<small title={current.statusDetail}>{EVIDENCE[current.status] ?? (current.status.startsWith("Reported by") ? "Reported" : current.status)}</small></td>
              <td className={s.speed}><strong data-speed="decode">{current.decode == null ? "—" : current.decode.toLocaleString("en-US", { maximumFractionDigits: 1 })}</strong><span className="dim"> tok/s</span>{current.prefill != null && <small><span data-speed="prefill">{current.prefill.toLocaleString("en-US", { maximumFractionDigits: 1 })}</span> prefill</small>}</td>
              <td><button className="btn" aria-expanded={!!open[row.id]} onClick={() => setOpen((expanded) => ({ ...expanded, [row.id]: !expanded[row.id] }))}>{open[row.id] ? "Hide" : "Command"}</button></td>
            </tr>,
            ...(open[row.id] ? [<tr key={`${row.id}/command`} className={s.commandRow}><td colSpan={columns}>
              <div className={s.commandHead}><span>Exact command · {row.model} · {row.hardware}</span><Copy text={current.command} event="command_copied" props={{ recipe: current.key, from: "configuration_table" }} /></div>
              <p className="dim">{current.status}{current.testedAt ? ` · ${current.testedAt}` : ""}. {current.statusDetail}</p>
              <p className="dim">{current.context.toLocaleString("en-US")} tokens context · {current.format || "Format not recorded"} · {row.cards} card{row.cards === 1 ? "" : "s"} per machine · {row.machines} machine{row.machines === 1 ? "" : "s"}{row.host ? ` · ${row.host}` : ""}</p>
              <p className="dim">Requires the hardware above and the recipe’s engine prerequisites. {current.machines > 1 ? `Run on each of the ${current.machines} machines with its documented node rank and network settings. ` : ""}<Link href={current.href}>Full recipe, setup and evidence →</Link></p>
              {current.requirements && <p className="dim">Setup required: {current.requirements}</p>}
              <pre className="code" data-command-for={current.key}>{current.command}</pre>
            </td></tr>] : []),
          ])}</tbody>
        </table>
      </div>
      {!visible.length && <p className="dim">No configurations match “{query}”. Try another hardware name, engine or model.</p>}
      <p className={s.note}>Recorded decode speed; prefill shown where measured. “—” means unmeasured. Open Command for setup requirements and full evidence.</p>
    </div>
  );
}
