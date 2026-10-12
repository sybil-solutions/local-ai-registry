import Link from "next/link";
import Logo from "@/components/Logo";
import PageHead from "@/components/PageHead";
import { NET } from "@/lib/landing/pictures";
import { models, recipes, cardName, fmtDate } from "@/lib/registry";

export const metadata = { title: "Models", description: "Every model in the registry, what it is good for, and the GPUs it runs on." };

export default function Models() {
  const used = Object.entries(models).filter(([id]) => recipes.some((r) => r.model === id));
  return (
    <main>
      <PageHead title="Models" sub="Open models from the last eight months, thinking on, each with the GPUs it runs on." art={NET} />
      <section className="grid cols-2">
        {used.map(([id, m]) => {
          // one link per card: a card with several setups (1×, 2×, two machines) has a recipe for each
          const rs = recipes.filter((r) => r.model === id).filter((r, i, all) => all.findIndex((x) => x.card === r.card) === i);
          return (
            <div key={id} id={id} className="rc">
              <div className="head"><Logo family={m.logo ?? m.family} />{m.name}</div>
              <div className="meta"><span>released {fmtDate(m.released)}</span>{m.vision && <span>vision</span>}{m.reasoning && <span>thinks</span>}</div>
              {m.about && <p className="about">{m.about}</p>}
              <div className="dim" style={{ fontSize: 13 }}>Runs on {rs.length} GPU{rs.length === 1 ? "" : "s"}: {rs.slice(0, 8).map((r, i) => <span key={r.key}>{i ? ", " : ""}<Link href={`/gpu/${r.card}/${r.slug}`} style={{ color: "var(--text)" }}>{cardName(r.card)}</Link></span>)}{rs.length > 8 ? `, and ${rs.length - 8} more` : ""}</div>
            </div>
          );
        })}
      </section>
    </main>
  );
}
