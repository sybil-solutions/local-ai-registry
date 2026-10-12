import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Copy from "@/components/Copy";
import Logo from "@/components/Logo";
import Viewed from "@/components/Viewed";
import { recipes, recipe, card, model, format, ctxLabel, engineKind, gates, status, steps, weightsList, fmtDate, cardName, setupOf } from "@/lib/registry";

type P = { params: Promise<{ card: string; recipe: string }> };
export const generateStaticParams = () => recipes.map((r) => ({ card: r.card, recipe: r.slug }));
const GATES: [string, string][] = [["load", "Loads and serves within an hour"], ["chat", "Answers a plain question and stops on its own"], ["reasoning", "Thinks separately and gets 17 × 23 right"], ["tools", "Calls a tool with the right arguments and uses the result"], ["context", "Recalls a code buried in a prompt that fills 85% of the window"], ["speed", "Decodes at 15 tok/s or more"]];

export async function generateMetadata({ params }: P): Promise<Metadata> {
  const { card: c, recipe: s } = await params;
  const r = recipe(c, s);
  if (!r) return {};
  const on = setupOf(r).name;
  return { title: `${model(r).name} on ${on}`, description: `${format(r)}, ${ctxLabel(r.launch.ctx)} context, ${Math.round(r.proof[0].tps ?? 0)} tok/s on ${setupOf(r).cards * setupOf(r).machines > 1 ? on : `the ${on}`}. ${status(r).detail}` };
}

export default async function RecipePage({ params }: P) {
  const { card: cid, recipe: slug } = await params;
  const r = recipe(cid, slug);
  const c = card(cid);
  if (!r || !c) notFound();
  const m = model(r);
  const g = gates(r);
  const s = status(r);
  const p = r.proof[0];
  const w = weightsList(r.launch)[0];
  const run = steps(r);
  const endpoint = `http://localhost:${r.launch.kind === "host" ? r.launch.port : 8000}/v1`;
  const source = r.launch.source ?? (p.reported && p.src ? `https://github.com/${p.src.replace("@", "/tree/")}` : undefined);
  return (
    <main>
      <Viewed event="recipe_viewed" props={{ gpu: cid, recipe: r.key, model: r.model }} />
      <Link href={`/gpu/${cid}`} className="back">‹ {cardName(cid)}</Link>
      <h1 className="title"><Logo family={m.logo ?? m.family} size={28} /><Link href={`/models/${r.model}/`}>{m.name}</Link></h1>
      <div className="dim">{format(r)} · {engineKind(r)} · {ctxLabel(r.launch.ctx)} context · {setupOf(r).name} · {c.vendor === "cpu" ? "System RAM" : `${setupOf(r).vram_gb} GB`}</div>
      {m.about && <p style={{ maxWidth: "70ch", marginTop: 22 }}>{m.about}</p>}

      <section className="grid cols-4">
        <div className="stat"><b>{p.tps ? Math.round(p.tps) : "–"} tok/s</b><span>decode</span></div>
        <div className="stat"><b>{p.prefill ? Math.round(p.prefill).toLocaleString("en-US") : "–"} tok/s</b><span>prefill</span></div>
        <div className="stat"><b>{ctxLabel(r.launch.ctx)}</b><span>context window</span></div>
        <div className="stat"><b className={s.tone}>{s.label}</b><span>{p.at ? fmtDate(p.at) : "earlier"}</span></div>
      </section>

      <section>
        <span className="label">Run it</span>
        {p.reported ? <p className="dim" style={{ marginTop: 0 }}>This is the launch as {s.label.replace("Reported by ", "")} publishes it. Their repository has the full setup: <a href={source} style={{ textDecoration: "underline" }}>{p.src} ›</a> The server then answers on <code>{endpoint}</code>.</p> :
        <p className="dim" style={{ marginTop: 0 }}>Follow the steps below to download and run this recipe. <a href="https://github.com/sybil-solutions/omarchy-local-ai" style={{ textDecoration: "underline" }}>Omarchy Local AI</a> lists the recipes it can launch on your hardware. The server then answers on <code>{endpoint}</code>.</p>}
        <ol className="steps">
          {run.map((st, i) => (
            <li key={i}>
              <div className="step-head"><span><span className="faint">{i + 1}</span> {st.title}</span><Copy text={st.code} event="command_copied" props={{ gpu: cid, recipe: r.key, step: i + 1 }} /></div>
              <pre className="code">{st.code}</pre>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <span className="label">What it passed</span>
        <div className="grid cols-2">
          {GATES.map(([k, l]) => <div key={k} className="stat"><b className={g.has(k) ? "ok" : "faint"}>{g.has(k) ? "✓" : "–"} {k}</b><span>{l}</span></div>)}
        </div>
        <p className="dim">{s.detail}</p>
      </section>

      <section>
        <span className="label">Details</span>
        <dl className="kv">
          <dt>Weights</dt><dd>{!w ? "inside the image" : <a href={`https://huggingface.co/${w.repo}/tree/${w.revision}`}>{w.repo} @ {w.revision.slice(0, 10)} ›</a>}</dd>
          <dt>Image</dt><dd>{r.launch.build ? <a href={`https://github.com/${r.launch.build.repo}/tree/${r.launch.build.commit}`}>built from {r.launch.build.repo} @ {r.launch.build.commit.slice(0, 10)} ›</a> : r.launch.image ?? "none: a program on the host"}</dd>
          {source && <><dt>Source</dt><dd><a href={source}>{source.replace("https://github.com/", "").replace("/tree/", " @ ").slice(0, 60)} ›</a></dd></>}
          <dt>{r.profile.startsWith("registry/launches/") ? "Launch" : "Engine profile"}</dt><dd><a href={`https://github.com/sybil-solutions/local-ai-registry/blob/main/${r.profile}`}>{r.engine.split("@")[0]} ›</a></dd>
          <dt>Recipe file</dt><dd><a href={`https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/${r.key}.json`}>registry/recipes/{r.key}.json ›</a></dd>
          {m.released && <><dt>Model released</dt><dd>{fmtDate(m.released)}</dd></>}
          <dt>Tested</dt><dd>{p.reported ? `not yet by us; reported by ${p.on}` : p.on === "legacy" ? "earlier acceptance" : `${p.on}${p.gpu ? `, ${p.gpu}` : ""}`}{p.proxy ? ` (sibling: ${cardName(p.proxy)})` : ""}</dd>
        </dl>
      </section>
    </main>
  );
}
