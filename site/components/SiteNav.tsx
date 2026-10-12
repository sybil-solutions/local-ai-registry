import Link from "next/link";
import copy from "@/lib/landing/copy.json";
import s from "@/app/landing.module.css";

// The header and footer of every page: the landing, the videos page and the registry pages (see (registry)/layout.tsx).
export const X = "https://x.com/0xSero";
export const PLUGIN = "https://github.com/sybil-solutions/omarchy-local-ai";
export const REGISTRY = "https://github.com/sybil-solutions/local-ai-registry";
const NAV: Record<string, string> = { "what it is": "/#what", models: "/models/", hardware: "/hardware/", videos: "/videos/", "how it works": "/#how", contribute: "/#contribute", github: REGISTRY, docs: "/docs/", api: "/docs/#api" };

export const list = (v: string | null | undefined) => String(v ?? "").split(",").map((x) => x.trim()).filter(Boolean);

export function XIcon() {
  return (
    <svg viewBox="0 0 24 24" width="1em" height="1em" aria-hidden="true" className={s.xicon}>
      <path fill="currentColor" d="M18.24 2.25h3.31l-7.23 8.26 8.5 11.24h-6.66l-5.21-6.82-5.97 6.82H1.67l7.73-8.84L1.25 2.25h6.83l4.71 6.23 5.45-6.23Zm-1.16 17.52h1.83L7.08 4.13H5.12l11.96 15.64Z" />
    </svg>
  );
}

export function SiteNav() {
  return (
    <header className={s.nav}>
      <Link href="/" className={s.word}>{copy.word}</Link>
      <nav>
        {list(copy.sNav).map((label) => {
          const href = NAV[label.toLowerCase()] ?? "#";
          return href.startsWith("http") ? <a key={label} href={href}>{label}</a> : <Link key={label} href={href}>{label}</Link>;
        })}
        {copy.sFollow && <a href={X} className={s.btn}><XIcon />{copy.sFollow}</a>}
      </nav>
    </header>
  );
}

export function SiteFoot() {
  return <footer className={s.foot}><a href={X}>{copy.sFooter}</a><Link href="/videos/">Videos</Link><Link href="/hardware/">Hardware</Link><a href={REGISTRY}>Registry</a></footer>;
}
