import type { Metadata } from "next";
import { SiteFoot, SiteNav, X } from "@/components/SiteNav";
import copy from "@/lib/landing/copy.json";
import s from "../landing.module.css";

// Every video, moved off the landing page. The titles, dates, links and poster frames come from lib/landing/copy.json (v1…v6).
export const metadata: Metadata = { title: "Videos", description: copy.vLede ?? undefined };

function Play() {
  return (
    <span className={s.play} aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M7 4.5v15l13-7.5z" /></svg>
    </span>
  );
}

export default function Videos() {
  const C = copy as Record<string, string | null>;
  const videos = [1, 2, 3, 4, 5, 6].map((i) => ({ title: C[`v${i}t`], meta: C[`v${i}m`], href: C[`v${i}u`] ?? X, img: C[`v${i}i`] })).filter((v) => v.title);
  return (
    <div className={s.page}>
      <SiteNav />
      <section className={`${s.section} ${s.top}`}>
        <div className={s.head}><span className={s.label}>{C.vHead}</span><a href={X} className={s.more}>{copy.sWatchLink}</a></div>
        {C.vLede && <p className={s.lede}>{C.vLede}</p>}
        <div className={s.videos}>
          {videos.map((v) => (
            <a key={v.href} href={v.href} className={s.thumb} target="_blank" rel="noreferrer">
              <span className={s.frame}>
                {v.img ? <img src={`/landing/videos/${v.img}.jpg`} alt="" className={s.poster} loading="lazy" /> : <img src="/landing/hands.svg" alt="" />}
                <Play />
                {v.meta && <span className={s.meta}>{v.meta}</span>}
              </span>
              <span className={s.thumbTitle}>{v.title}</span>
            </a>
          ))}
        </div>
      </section>
      <SiteFoot />
    </div>
  );
}
