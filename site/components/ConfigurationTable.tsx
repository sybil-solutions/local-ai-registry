"use client";

import { Fragment, useId, useMemo, useState } from "react";
import Link from "next/link";
import Copy from "./Copy";
import type { ConfigurationRow, ConfigurationVariant } from "@/lib/configurations";
import { SortControl, SortHeader, compareValues, type Column, type Sort } from "./tableSort";
import s from "./RegistryTables.module.css";

type SortKey = "hardware" | "model" | "memory" | "label" | "context" | "decode" | "prefill" | "status";
type Entry = { row: ConfigurationRow; variants: ConfigurationVariant[]; current: ConfigurationVariant };

/** "RTX 3090", "4 × RTX 3090", "2 × DGX Spark GB10": the multiplier is the device count you need. */
const setupName = (row: ConfigurationRow) => row.machines > 1 ? `${row.machines} × ${row.cardName}` : row.cards > 1 ? `${row.cards} × ${row.cardName}` : row.cardName;
/** Only multi-device setups need spelling out; a bare name is one card in one machine. */
const topology = (row: ConfigurationRow) =>
  row.machines > 1 ? `${row.machines} machines${row.cards > 1 ? `, ${row.cards} cards each` : ""}` : row.cards > 1 ? `${row.cards} cards in one machine` : "";

function Measure({ value, strong = false }: { value: number | null; strong?: boolean }) {
  if (value == null) return <span className={s.unknown} title="No measurement recorded">—</span>;
  return <span className={strong ? s.strong : undefined}>{value.toLocaleString("en-US", { maximumFractionDigits: 1 })}</span>;
}

export default function ConfigurationTable({ rows, showModel = false }: { rows: ConfigurationRow[]; showModel?: boolean }) {
  const searchId = useId();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [sort, setSort] = useState<Sort<SortKey>>({ key: showModel ? "model" : "hardware", descending: false });
  // On a hardware page every setup is the same card, so its setups order by total memory (= device count)
  // and memory sits in the setup cell; a model page compares different hardware, so memory gets a column.
  const columns: Column<SortKey>[] = [
    ...(showModel
      ? [{ key: "model", label: "Model" }, { key: "memory", label: "Setup", unit: "by total memory", numeric: true, leftAligned: true }] as const
      : [{ key: "hardware", label: "Hardware" }, { key: "memory", label: "Memory", unit: "GB", numeric: true }] as const),
    { key: "label", label: "Recipe" },
    { key: "context", label: "Context", numeric: true },
    { key: "decode", label: "Decode", unit: "tok/s", numeric: true },
    { key: "prefill", label: "Prefill", unit: "tok/s", numeric: true },
    { key: "status", label: "Evidence" },
  ];
  const visible = useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const result: Entry[] = [];
    for (const row of rows) {
      const common = `${row.hardware} ${row.hardwareId} ${row.model} ${row.modelId} ${row.family} ${row.memory}gb ${row.cards} cards ${row.machines} machines ${row.host}`;
      const variants = row.variants.filter((v) => {
        const text = `${common} ${v.label} ${v.key} ${v.context} ${v.context.toLocaleString("en-US")} context ${v.status} ${v.requirements}`.toLowerCase();
        return terms.every((term) => text.includes(term));
      });
      if (!variants.length) continue;
      result.push({ row, variants, current: variants.find((v) => v.key === selected[row.id]) ?? variants[0] });
    }
    const value = (entry: Entry) => sort.key === "hardware" || sort.key === "model" || sort.key === "memory" ? entry.row[sort.key] : entry.current[sort.key];
    return result.sort((a, b) => compareValues(value(a), value(b), sort.descending) || a.row.id.localeCompare(b.row.id));
  }, [rows, query, selected, sort]);
  const recipeCount = visible.reduce((n, entry) => n + entry.variants.length, 0);
  return (
    <div className={s.tableArea}>
      <div className={s.toolbar}>
        <label className={s.search} htmlFor={searchId}>Search configurations<input id={searchId} type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Hardware, model, engine, format, host RAM…" /></label>
        <SortControl columns={columns} sort={sort} setSort={setSort} />
        <span className={s.count} aria-live="polite"><b>{visible.length}</b> configurations · <b>{recipeCount}</b> recipes</span>
      </div>
      <div className={s.frame}>
        <table className={`${s.table} ${s.configurations} ${showModel ? s.byModel : s.byHardware}`}>
          <thead><tr>{columns.map((column) => <SortHeader key={column.key} column={column} sort={sort} setSort={setSort} />)}</tr></thead>
          <tbody>{visible.map(({ row, variants, current }) => {
            const expanded = !!open[row.id];
            const commandId = `${searchId}-command-${row.id.replace(/[^a-z0-9]+/gi, "-")}`;
            const name = setupName(row);
            const detail = [topology(row), showModel ? row.memory ? `${row.memory} GB total` : "System RAM" : ""].filter(Boolean).join(" · ");
            return <Fragment key={row.id}>
              <tr className={expanded ? s.expanded : undefined} data-configuration={row.id} data-recipe={current.key}>
                {showModel && <td className={s.identity}><Link className={s.primary} href={`/models/${row.modelId}/`}>{row.model}</Link>{row.family && <span className={s.meta}>{row.family}</span>}</td>}
                <td className={s.identity}>
                  {showModel ? <span className={s.primary}>{name}</span> : <Link className={s.primary} href={`/gpu/${row.hardwareId}/`}>{name}</Link>}
                  {detail && <span className={s.meta}>{detail}</span>}
                  {row.needs.length > 0 && <dl className={s.needs}>{row.needs.map((need) => <div key={need.label}><dt>{need.label}</dt><dd>{need.value}</dd></div>)}</dl>}
                </td>
                {!showModel && <td className={s.num} data-label="Memory GB"><span className={s.primary}>{row.memory || "System RAM"}</span></td>}
                <td className={s.recipe}><div className={s.recipeLine}>
                  <div className={variants.length > 1 ? `${s.picker} ${s.choice}` : s.picker} title={current.label}>
                    <span className={s.engine}><span>{current.engine}</span>{variants.length > 1 && <span className={s.options}>{variants.length} recipes</span>}</span>
                    <span className={current.format ? s.format : `${s.format} ${s.unknown}`}>{current.format || "format not recorded"}</span>
                    {variants.length > 1 && <select aria-label={`Recipe for ${row.model} on ${row.hardware}`} value={current.key} onChange={(e) => setSelected((chosen) => ({ ...chosen, [row.id]: e.target.value }))}>{variants.map((v) => <option key={v.key} value={v.key}>{v.label}</option>)}</select>}
                  </div>
                  <button type="button" className={s.toggle} aria-expanded={expanded} aria-controls={commandId} aria-label={`${expanded ? "Hide" : "Show"} command for ${current.label}`} onClick={() => setOpen((state) => ({ ...state, [row.id]: !state[row.id] }))}>Command<span aria-hidden="true">{expanded ? "▴" : "▾"}</span></button>
                </div></td>
                <td className={s.num} data-label="Context"><span className={s.primary}>{current.context >= 1024 ? `${Math.round(current.context / 1024)}K` : current.context}</span></td>
                <td className={s.num} data-label="Decode tok/s" data-speed="decode"><span className={s.primary}><Measure value={current.decode} strong /></span></td>
                <td className={s.num} data-label="Prefill tok/s" data-speed="prefill"><span className={s.primary}><Measure value={current.prefill} /></span></td>
                <td className={s.evidence} title={current.statusDetail}>
                  <span className={s.primary}><i className={`dot ${current.statusTone === "ok" ? "" : current.statusTone === "warm" ? "warm" : "dim"}`} />{current.status}</span>
                  <span className={s.meta}>{current.testedAt || "Date not recorded"}</span>
                </td>
              </tr>
              {expanded && <tr className={s.commandRow} id={commandId}><td colSpan={columns.length}>
                <div className={s.commandHead}>
                  <div><span className={s.commandTitle}>Exact command</span><span className={s.meta}>{row.model} · {row.hardware} · {current.label}</span></div>
                  <div className={s.commandActions}><Link href={current.href}>Full recipe, setup and evidence →</Link><Copy text={current.command} event="command_copied" props={{ recipe: current.key, from: "configuration_table" }} /></div>
                </div>
                <p>Requires the hardware above and the recipe’s engine prerequisites.{current.machines > 1 ? ` Run on each of the ${current.machines} machines with its documented node rank and network settings.` : ""}{row.host ? ` Host: ${row.host}.` : ""}</p>
                {current.requirements && <p>Setup required: {current.requirements}</p>}
                <pre className="code" data-command-for={current.key}>{current.command}</pre>
              </td></tr>}
            </Fragment>;
          })}</tbody>
        </table>
        {!visible.length && <p className={s.empty}>No configurations match “{query}”. Try another hardware name, engine or model.</p>}
      </div>
      <p className={s.note}>Speeds are the selected recipe’s recorded measurements, not estimates; “—” means no measurement was recorded. Evidence: <i className="dot" /> passed all checks on this card, <i className="dot warm" /> tested on a sibling card, <i className="dot dim" /> an earlier check or a publisher’s report. Hover an evidence label for its source.</p>
    </div>
  );
}
