import Link from "next/link";
import Logo from "@/components/Logo";
import { models, recipes, cardName, fmtDate } from "@/lib/registry";

export const metadata = { title: "Models", description: "Every model in the registry, what it is good for, and the GPUs it runs on." };

export default function Models() {
  const used = Object.entries(models).filter(([id]) => recipes.some((r) => r.model === id));
  return (
    <main>
      <div className="hero" style={{ paddingBottom: 0 }}><h1>Models</h1><p>Only models released in the last eight months, from Qwen, Gemma, DeepSeek, GLM, Step, Kimi and MiniMax, with thinking on.</p></div>
      <section className="grid cols-2">
        {used.map(([id, m]) => {
          const rs = recipes.filter((r) => r.model === id);
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
