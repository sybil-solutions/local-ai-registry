"use client";
import { useId, type Dispatch, type SetStateAction } from "react";
import s from "./RegistryCards.module.css";

export type Sort<K extends string> = { key: K; descending: boolean };
export type Column<K extends string> = { key: K; label: string; unit?: string; numeric?: boolean };
type Value = string | number | null | undefined;

/** Unknown values remain last in either direction. */
export function compareValues(av: Value, bv: Value, descending: boolean): number {
  const aUnknown = av == null || av === "", bUnknown = bv == null || bv === "";
  if (aUnknown || bUnknown) return aUnknown === bUnknown ? 0 : aUnknown ? 1 : -1;
  const order = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
  return descending ? -order : order;
}

export function SortControl<K extends string>({ columns, sort, setSort }: { columns: Column<K>[]; sort: Sort<K>; setSort: Dispatch<SetStateAction<Sort<K>>> }) {
  const id = useId();
  const numeric = !!columns.find((c) => c.key === sort.key)?.numeric;
  const direction = sort.descending ? numeric ? "↓ High–low" : "↓ Z–A" : numeric ? "↑ Low–high" : "↑ A–Z";
  return <div className={s.sortControl}>
    <label htmlFor={id}>Sort by</label>
    <select id={id} value={sort.key} onChange={(e) => { const column = columns.find((c) => c.key === e.target.value)!; setSort({ key: column.key, descending: !!column.numeric }); }}>
      {columns.map((c) => <option key={c.key} value={c.key}>{c.label}{c.unit ? ` (${c.unit})` : ""}</option>)}
    </select>
    <button type="button" onClick={() => setSort((current) => ({ ...current, descending: !current.descending }))} aria-label={`Sort direction: ${sort.descending ? "descending" : "ascending"}. Activate to reverse.`}>{direction}</button>
  </div>;
}
