import { og, size } from "@/lib/og";
import { cards, card, picks, lead, model, format, ctxLabel, short, setupTag, specs, bare } from "@/lib/registry";
import { isEstimated, tpsValue } from "@/lib/speed";
export { size };
export const contentType = "image/png";
export const dynamic = "force-static";
export const generateStaticParams = () => [...cards, ...bare].map((c) => ({ card: c.id }));
export default async function Image({ params }: { params: Promise<{ card: string }> }) {
  const id = (await params).card, c = card(id);
  if (!c) {
    const h = specs(id)!;
    const top = h.tflops.find((t) => t[0] === "FP16" || t[0] === "BF16");
    return og({ kicker: `${h.vram_gb} GB`, title: short(h.name), sub: "No recipe yet · spec sheet", logo: false,
      stats: [[h.bandwidth_gb_s ? h.bandwidth_gb_s.toLocaleString("en-US") : "–", "GB/s"], [top?.[1] ? `${Math.round(top[1])}` : "–", "FP16 TFLOPS"], [h.price.find((p) => p[0] !== "Availability")?.[1] ?? "–", "price"]] });
  }
  const s = lead(c);
  const top = picks(s)[0];
  const m = model(top);
  const others = c.setups.filter((x) => x.picks !== s.picks);
  return og({ kicker: c.vendor === "cpu" ? "System RAM" : `${c.picks.length ? `${c.vram_gb} GB` : `${c.setups[0].label} · ${c.setups[0].vram_gb} GB`}${others.length ? ` · also ${others.map(setupTag).join(", ")}` : ""}`, title: short(c.name), sub: `Runs ${m.name} · ${format(top)} · ${ctxLabel(top.launch.ctx)} context`, logo: m.family === "qwen",
    stats: [[tpsValue(top.proof[0]), isEstimated(top.proof[0]) ? "tok/s decode (estimate)" : "tok/s decode"], [top.proof[0].prefill ? `${Math.round(top.proof[0].prefill).toLocaleString("en-US")}` : "–", "tok/s prefill"], [`${picks(s).length}`, picks(s).length === 1 ? "recipe" : "recipes"]] });
}
