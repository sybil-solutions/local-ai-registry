import { og, size } from "@/lib/og";
import { recipes, recipe, model, format, ctxLabel, status, setupOf } from "@/lib/registry";
import { isEstimated, tpsValue } from "@/lib/speed";
export { size };
export const contentType = "image/png";
export const dynamic = "force-static";
export const generateStaticParams = () => recipes.map((r) => ({ card: r.card, recipe: r.slug }));
export default async function Image({ params }: { params: Promise<{ card: string; recipe: string }> }) {
  const { card: c, recipe: s } = await params;
  const r = recipe(c, s)!;
  const m = model(r);
  return og({ kicker: setupOf(r).name, title: m.name, sub: `${format(r)} · ${ctxLabel(r.launch.ctx)} context · ${status(r).label.toLowerCase()}`, logo: m.family === "qwen",
    stats: [[tpsValue(r.proof[0]), isEstimated(r.proof[0]) ? "tok/s decode (estimate)" : "tok/s decode"], [r.proof[0].prefill ? `${Math.round(r.proof[0].prefill).toLocaleString("en-US")}` : "–", "tok/s prefill"], [ctxLabel(r.launch.ctx), "context"]] });
}
