import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import HardwarePhoto from "@/components/HardwarePhoto";
import HardwareSpecs from "@/components/HardwareSpecs";
import ConfigurationTable from "@/components/ConfigurationTable";
import { configurationRows } from "@/lib/configurations";
import { cards, card, recipes, VENDOR, short, specs, bare, memoryLabel } from "@/lib/registry";

type P = { params: Promise<{ card: string }> };
export const generateStaticParams = () => [...cards, ...bare].map((c) => ({ card: c.id }));

export async function generateMetadata({ params }: P): Promise<Metadata> {
  const id = (await params).card, c = card(id), h = specs(id);
  if (!c && !h) return {};
  const name = short((c ?? h)!.name);
  return {
    title: `${name} · ${c ? memoryLabel(c) : `${h!.vram_gb} GB`}`,
    description: `${name}: original hardware photos, specifications, every recorded model configuration, measured decode and prefill speeds, and exact launch commands.`,
  };
}

export default async function Gpu({ params }: P) {
  const id = (await params).card, c = card(id), h = specs(id);
  if (!c && !h) notFound();
  const hardware = (c ?? h)!;
  const rs = recipes.filter((r) => r.card === id);
  const rows = configurationRows(rs);
  return (
    <main>
      <Link href="/hardware/" className="back">‹ all hardware</Link>
      <div className="phead">
        <div className="phead-text">
          <h1>{short(hardware.name)}</h1>
          <p>{VENDOR[hardware.vendor]} · {c ? memoryLabel(c) : `${h!.vram_gb} GB`}{hardware.bandwidth_gb_s != null ? ` · ${hardware.bandwidth_gb_s} GB/s` : ""}</p>
          <div className="facts"><span><b>{new Set(rs.map((r) => r.model)).size}</b> models</span><span><b>{rows.length}</b> model configurations</span><span><b>{rs.length}</b> exact recipes</span></div>
        </div>
        <HardwarePhoto id={id} />
      </div>
      <section id="configurations">
        <h2>Model configurations</h2>
        {rs.length ? <><p className="dim">Each row is a model and hardware setup, including any documented host RAM, storage or CPU requirements. Select a recipe to update its speeds, context and exact command.</p><ConfigurationTable rows={rows} showModel /></> : <p className="dim">No recorded recipe for this hardware yet. Specifications below are sourced separately; no speed or launch command is inferred.</p>}
      </section>
      {h && <HardwareSpecs h={h} />}
    </main>
  );
}
