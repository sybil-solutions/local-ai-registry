import Link from "next/link";
import Picker from "@/components/Picker";
import Logo from "@/components/Logo";
import PageHead from "@/components/PageHead";
import { GPU } from "@/lib/landing/pictures";
import { cards, picks, lead, model, format, ctxLabel, stats, VENDOR, short, setupTag, bare, memoryLabel } from "@/lib/registry";

export const metadata = { title: "Hardware" };

export default function Hardware() {
  const vendors = ["nvidia", "amd", "intel", "apple", "cpu"].filter((v) => cards.some((c) => c.vendor === v) || bare.some((h) => h.vendor === v));
  return (
    <main>
      <PageHead title="Hardware" sub="Pick your card: the best models it runs, how fast, and the command." art={GPU}>
        <Picker options={[...cards, ...bare].map((c) => ({ id: c.id, name: short(c.name), vendor: c.vendor, vram: c.vram_gb }))} />
        <div className="facts">
          <span><b>{stats.gpus}</b> with recipes</span><span><b>{stats.recipes}</b> recipes</span><span><b>{stats.tested}</b> tested on the card</span>
        </div>
      </PageHead>

      <section id="gpus">
        {vendors.map((v) => (
          <div key={v} style={{ marginTop: v === "nvidia" ? 0 : 40 }}>
            <span className="label" style={{ display: "block", marginBottom: 4 }}>{VENDOR[v]}</span>
            <div className="gpus">
              {cards.filter((c) => c.vendor === v).map((c) => {
                const top = picks(lead(c))[0];
                const m = top && model(top);
                const tps = top?.proof[0].tps ?? 0;
                return (
                  <Link key={c.id} href={`/gpu/${c.id}`} className="row">
                    <span className="gpu-name">{short(c.name)}{!c.picks.length && <span className="faint"> · {setupTag(c.setups[0])}</span>}{c.setups.length > 1 && <span className="faint"> · also {c.setups.slice(1).map(setupTag).join(", ")}</span>}</span>
                    <span className="gpu-mem dim">{memoryLabel(c)}</span>
                    {top && <span className="gpu-model"><Logo family={m.logo ?? m.family} size={16} />{m.name}<span className="faint">{format(top)}</span></span>}
                    <span className="gpu-speed"><span className="bar"><i style={{ width: `${Math.min(100, tps / 1.8)}%` }} /></span><span>{tps ? Math.round(tps) : "–"}<span className="faint"> tok/s</span></span></span>
                    <span className="dim">›</span>
                  </Link>
                );
              })}
            </div>
            {bare.some((h) => h.vendor === v) && (
              <p className="dim" style={{ fontSize: 13, marginTop: 12, lineHeight: 1.8 }}>
                No recipe yet: {bare.filter((h) => h.vendor === v).map((h, i) => <span key={h.id}>{i ? " · " : ""}<Link href={`/gpu/${h.id}`}>{short(h.name)}</Link></span>)}
              </p>
            )}
          </div>
        ))}
      </section>
    </main>
  );
}
