"use client";
import Link from "next/link";
import { useState } from "react";
import s from "./Find.module.css";

// A searchable list: the most common entries first, the first few shown, every entry reachable by typing.
export type Item = { name: string; meta: string; href: string; terms: string };

export default function Find({ title, placeholder, items, all, show = 8 }: { title: string; placeholder: string; items: Item[]; all: { label: string; href: string }; show?: number }) {
  const [q, setQ] = useState("");
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const hits = words.length ? items.filter((i) => words.every((w) => i.terms.includes(w))) : items;
  return (
    <div className={s.find}>
      <div className={s.top}><span className={s.title}>{title}</span><span className={s.count}>{items.length}</span></div>
      <input className={s.search} value={q} onChange={(e) => setQ(e.target.value)} placeholder={placeholder} aria-label={placeholder} />
      <ul className={s.list}>
        {hits.slice(0, words.length ? 50 : show).map((i) => (
          <li key={i.href}><Link href={i.href}><span>{i.name}</span><span className={s.meta}>{i.meta}</span></Link></li>
        ))}
        {!hits.length && <li className={s.none}>Nothing yet. <Link href="/#contribute">Add it →</Link></li>}
      </ul>
      <Link href={all.href} className={s.all}>{all.label} →</Link>
    </div>
  );
}
