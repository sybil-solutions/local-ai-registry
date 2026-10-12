"use client";
// Sorting shared by the registry tables: sortable column headings on wide screens and an
// equivalent "Sort by" control for the stacked narrow-screen layout, where headings are hidden.
import { useId, type Dispatch, type SetStateAction } from "react";
import s from "./RegistryTables.module.css";

export type Sort<K extends string> = { key: K; descending: boolean };
/** `numeric`: sorts as numbers, largest first, and right-aligns unless `leftAligned` (a named setup ordered by memory). */
export type Column<K extends string> = { key: K; label: string; unit?: string; numeric?: boolean; leftAligned?: boolean };
type Value = string | number | null | undefined;

/** Orders two values; unknown values (null, undefined, "") always sort last, whichever the direction. */
export function compareValues(av: Value, bv: Value, descending: boolean): number {
  const aUnknown = av == null || av === "", bUnknown = bv == null || bv === "";
  if (aUnknown || bUnknown) return aUnknown === bUnknown ? 0 : aUnknown ? 1 : -1;
  const order = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
  return descending ? -order : order;
}

/** Numbers start largest-first; names start A–Z. A second click reverses. */
const next = <K extends string>(column: Column<K>, current: Sort<K>): Sort<K> =>
  ({ key: column.key, descending: current.key === column.key ? !current.descending : !!column.numeric });

export function SortHeader<K extends string>({ column, sort, setSort }: { column: Column<K>; sort: Sort<K>; setSort: Dispatch<SetStateAction<Sort<K>>> }) {
  const active = sort.key === column.key;
  return (
    <th scope="col" className={column.numeric && !column.leftAligned ? s.num : undefined} aria-sort={active ? sort.descending ? "descending" : "ascending" : "none"}>
      <button type="button" className={active ? s.sorted : undefined} onClick={() => setSort((current) => next(column, current))}>
        <span>{column.label}{column.unit && <span className={s.unit}> {column.unit}</span>}</span>
        <span className={s.arrow} aria-hidden="true">{active ? sort.descending ? "↓" : "↑" : "↕"}</span>
      </button>
    </th>
  );
}

/** The narrow-screen sort: the same columns and directions as the headings. */
export function SortControl<K extends string>({ columns, sort, setSort }: { columns: Column<K>[]; sort: Sort<K>; setSort: Dispatch<SetStateAction<Sort<K>>> }) {
  const id = useId();
  const numeric = !!columns.find((c) => c.key === sort.key)?.numeric;
  const direction = sort.descending ? numeric ? "↓ High–low" : "↓ Z–A" : numeric ? "↑ Low–high" : "↑ A–Z";
  return (
    <div className={s.sortControl}>
      <label htmlFor={id}>Sort by</label>
      <select id={id} value={sort.key} onChange={(e) => { const column = columns.find((c) => c.key === e.target.value)!; setSort({ key: column.key, descending: !!column.numeric }); }}>
        {columns.map((c) => <option key={c.key} value={c.key}>{c.label}{c.unit ? ` (${c.unit})` : ""}</option>)}
      </select>
      <button type="button" onClick={() => setSort((current) => ({ ...current, descending: !current.descending }))} aria-label={`Sort direction: ${sort.descending ? "descending" : "ascending"}. Activate to reverse.`}>
        {direction}
      </button>
    </div>
  );
}
