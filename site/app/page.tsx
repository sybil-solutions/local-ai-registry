import type { Metadata } from "next";
import Copy from "@/components/Copy";
import Dots from "@/components/Dots";
import Find, { type Item } from "@/components/Find";
import Seen from "@/components/Seen";
import { PLUGIN, REGISTRY, SiteFoot, SiteNav } from "@/components/SiteNav";
import copy from "@/lib/landing/copy.json";
import { GPU, NET, TERMINAL } from "@/lib/landing/pictures";
import { cards, models, recipes, short } from "@/lib/registry";
import s from "./landing.module.css";

// The landing page: the promise, the hands, three dot pictures saying what it is, then every GPU and model, searchable.
// Every word comes from lib/landing/copy.json; the details live in /docs/ and the videos in /videos/.
const INSTALL = `omarchy plugin add ${PLUGIN} --enable`;

// The cards people most often run models on at home, first; every other card follows by how many recipes it has.
const COMMON = ["rtx-3090-24gb", "rtx-4090-24gb", "rtx-5090-32gb", "rtx-3060-12gb", "rtx-4060-ti-16gb", "rtx-5070-12gb", "rtx-4070-super-12gb", "rtx-5060-ti-16gb",
  "rtx-4080-super-16gb", "rtx-5080-16gb", "rx-7900-xtx-24gb", "rtx-4060-8gb", "rtx-3080-10gb", "rtx-3070-8gb", "dgx-spark-gb10-128gb", "apple-m4-max-128gb", "x86-64-avx2-cpu"];

export const metadata: Metadata = {
  title: { absolute: `${copy.word}: ${copy.sHead}` },
  description: copy.sSub ?? undefined,
};

const count = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;

function gpuItems(): Item[] {
  const n = (id: string) => recipes.filter((r) => r.card === id).length;
  const rank = (id: string) => (COMMON.includes(id) ? COMMON.indexOf(id) : COMMON.length);
  return [...cards]
    .sort((a, b) => rank(a.id) - rank(b.id) || n(b.id) - n(a.id) || a.name.localeCompare(b.name))
    .map((c) => ({ name: short(c.name), meta: `${c.vendor === "cpu" ? "RAM" : `${c.vram_gb} GB`} · ${count(n(c.id), "recipe")}`, href: `/gpu/${c.id}/`, terms: `${c.name} ${c.id} ${c.vendor} ${c.vram_gb}gb`.toLowerCase() }));
}

function modelItems(): Item[] {
  return Object.entries(models)
    .map(([id, m]) => ({ id, m, n: new Set(recipes.filter((r) => r.model === id).map((r) => r.card)).size }))
    .filter((x) => x.n)
    .sort((a, b) => b.n - a.n || a.m.name.localeCompare(b.m.name))
    .map(({ id, m, n }) => ({ name: m.name, meta: count(n, "GPU"), href: `/models/#${id}`, terms: `${m.name} ${id} ${m.family}`.toLowerCase() }));
}

export default function Landing() {
  const C = copy as Record<string, string | null>;
  const what = [GPU, NET, TERMINAL].map((art, i) => ({ art, t: C[`w${i + 1}t`], d: C[`w${i + 1}d`] }));

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
        <div className={s.flow}>
          {what.map((w, i) => (
            <div key={i} className={s.flowItem}>
              <div className={s.pic}><Dots art={w.art} label={w.t ?? undefined} /></div>
              {i < what.length - 1 && <span className={s.wire} aria-hidden="true"><i /></span>}
              <h3>{w.t}</h3>{w.d && <p>{w.d}</p>}
            </div>
          ))}
        </div>
      </section>

      <section id="find" className={`${s.section} ${s.finds}`}>
        <Find title={C.fGpus ?? "GPUs"} placeholder={C.fGpusQ ?? ""} items={gpuItems()} all={{ label: C.fGpusAll ?? "", href: "/hardware/" }} />
        <Find title={C.fModels ?? "Models"} placeholder={C.fModelsQ ?? ""} items={modelItems()} all={{ label: C.fModelsAll ?? "", href: "/models/" }} />
      </section>

      {C.cText && (
        <section id="contribute" className={`${s.section} ${s.follow}`}>
          <p>{C.cText}</p>
          <a href={C.cUrl ?? REGISTRY} className={s.btn}>{C.cBtn} →</a>
        </section>
      )}

      <SiteFoot />
      <Seen />
    </div>
  );
}
