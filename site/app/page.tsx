import type { Metadata } from "next";
import Link from "next/link";
import Copy from "@/components/Copy";
import Dots, { DotNumber } from "@/components/Dots";
import Seen from "@/components/Seen";
import { PLUGIN, REGISTRY, SiteFoot, SiteNav, X, XIcon, list } from "@/components/SiteNav";
import copy from "@/lib/landing/copy.json";
import logos from "@/lib/landing/logos.json";
import { GPU, STEPS, TERMINAL, WEIGHTS } from "@/lib/landing/pictures";
import { stats } from "@/lib/registry";
import s from "./landing.module.css";

// The landing page. Every word comes from lib/landing/copy.json, the text written in the brand page's form;
// the links are fixed here. The pictures are dot art in the style of the hands (lib/landing/pictures.ts); the videos
// have their own page at /videos/.
const INSTALL = `omarchy plugin add ${PLUGIN} --enable`;
const GROUP_LINK: Record<string, string> = { gpuLogos: "/hardware/", modLogos: "/models/" };

type Logo = { name: string; wordmark: boolean; body: string };
const LOGOS = logos as Record<string, Logo>;

export const metadata: Metadata = {
  title: { absolute: `${copy.word}: ${copy.sHead}` },
  description: copy.sSub ?? undefined,
};

function BrandLogo({ k }: { k: string }) {
  const L = LOGOS[k];
  if (!L) return null;
  // The Intel and AMD glyphs are wordmarks with empty space above and below, so they are drawn larger and show no name.
  const scale = L.wordmark ? (k === "amd" ? 3.6 : 2.4) : 1;
  return (
    <span className={s.logo} title={L.name}>
      <svg viewBox="0 0 24 24" fill="currentColor" fillRule="evenodd" style={{ height: `${scale}em`, margin: `${-(scale - 1) / 2}em 0` }} dangerouslySetInnerHTML={{ __html: L.body }} />
      {!L.wordmark && <span>{L.name}</span>}
    </span>
  );
}

/** The dotted line between two pictures, with a crimson dot travelling along it. */
function Wire() {
  return <span className={s.wire} aria-hidden="true"><i /></span>;
}

export default function Landing() {
  const groups = ([["gpuTitle", "gpuLogos"], ["harTitle", "harLogos"], ["modTitle", "modLogos"]] as const)
    .map(([t, l]) => ({ title: copy[t], keys: list(copy[l]).filter((k) => LOGOS[k]), href: GROUP_LINK[l] }))
    .filter((g) => g.keys.length);
  const C = copy as Record<string, string | null>;
  const steps = [1, 2, 3, 4, 5, 6].map((i) => [C[`s${i}t`], C[`s${i}d`]]).filter(([t]) => t);
  const ways = [1, 2, 3].map((i) => ({ t: C[`c${i}t`], d: C[`c${i}d`], l: C[`c${i}l`], u: C[`c${i}u`] })).filter((w) => w.t);
  const what = [GPU, WEIGHTS, TERMINAL].map((art, i) => ({ art, t: C[`w${i + 1}t`], d: C[`w${i + 1}d`] }));
  const numbers = [[stats.gpus, C.nGpus], [stats.recipes, C.nRecipes], [stats.tested, C.nTested]] as const;

  return (
    <div className={s.page}>
      <SiteNav />

      <section className={s.hero}>
        <h1>{copy.sHead}</h1>
        {copy.sSub && <p>{copy.sSub}</p>}
        <div className={s.cmd}>
          <span className={s.prompt}>$</span>
          <code title={INSTALL}>{copy.sCmd}</code>
          <Copy text={INSTALL} event="install_copied" props={{ from: "landing" }} />
        </div>
      </section>

      <div className={s.art}>
        <img src="/landing/hands.svg" alt="Two hands drawn in dots, reaching for each other" />
        {copy.credit && <span className={s.credit}>{copy.credit}</span>}
      </div>

      <section id="what" className={s.section}>
        <div className={s.head}><span className={s.label}>{C.wHead}</span></div>
        {C.wLede && <p className={s.big}>{C.wLede}</p>}
        <div className={s.flow}>
          {what.map((w, i) => (
            <div key={i} className={s.flowItem}>
              <div className={s.pic}><Dots art={w.art} /></div>
              {i < what.length - 1 && <Wire />}
              <span className={s.num}>0{i + 1}</span><h3>{w.t}</h3><p>{w.d}</p>
            </div>
          ))}
        </div>
        <Link href="/hardware/" className={s.numbers}>
          {numbers.map(([n, l]) => <span key={l} className={s.number}><DotNumber n={n} label={String(n)} /><span>{l}</span></span>)}
        </Link>
      </section>

      <section className={s.supported}>
        {copy.supLabel && <div className={s.supHead}><span className={s.label}>{copy.supLabel}</span></div>}
        <div className={s.supBar}>
          {groups.map((g) => (
            <div key={g.title ?? g.keys[0]} className={s.group}>
              {g.href ? <Link href={g.href} className={s.groupTitle}>{g.title}</Link> : <span className={s.groupTitle}>{g.title}</span>}
              {g.keys.map((k) => <BrandLogo key={k} k={k} />)}
            </div>
          ))}
        </div>
      </section>

      <section id="how" className={s.section}>
        <div className={s.head}><span className={s.label}>{copy.sHow}</span><Link href="/docs/" className={s.more}>The details →</Link></div>
        <div className={s.steps}>
          {steps.map(([t, d], i) => (
            <div key={i} className={s.step}><div className={s.stepPic}><Dots art={STEPS[i]} /></div><span className={s.num}>0{i + 1}</span><h3>{t}</h3><p>{d}</p></div>
          ))}
        </div>
      </section>

      {C.cHead && (
        <section id="contribute" className={s.section}>
          <div className={s.head}><span className={s.label}>{C.cHead}</span><a href={REGISTRY} className={s.more}>{C.cBtn} →</a></div>
          {C.cText && <p className={s.lede}>{C.cText}</p>}
          <div className={s.ways}>
            {ways.map((w) => (
              <a key={w.t} href={w.u ?? REGISTRY} className={s.way}><h3>{w.t}</h3><p>{w.d}</p>{w.l && <span>{w.l}</span>}</a>
            ))}
          </div>
        </section>
      )}

      <section className={`${s.section} ${s.follow}`}>
        <span className={s.label}>{copy.sXHead}</span>
        {copy.sXText && <p>{copy.sXText}</p>}
        <div className={s.followBtns}>
          <a href={X} className={s.btn}><XIcon />{copy.sXBtn}</a>
          <Link href="/videos/" className={s.btn}>{C.vHead} →</Link>
        </div>
      </section>

      <SiteFoot />
      <Seen />
    </div>
  );
}
