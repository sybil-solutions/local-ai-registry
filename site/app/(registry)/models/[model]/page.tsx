import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Logo from "@/components/Logo";
import ModelCard from "@/components/ModelCard";
import ConfigurationTable from "@/components/ConfigurationTable";
import { configurationRows, modelDirectoryRows } from "@/lib/configurations";
import { models, recipes, fmtDate } from "@/lib/registry";
import s from "./model.module.css";

type P = { params: Promise<{ model: string }> };
export const generateStaticParams = () => modelDirectoryRows().map((m) => ({ model: m.id }));

export async function generateMetadata({ params }: P): Promise<Metadata> {
  const m = models[(await params).model];
  return m ? { title: m.name, description: `${m.name}: Hugging Face model card, recorded hardware configurations, decode and prefill speeds, and exact launch commands.` } : {};
}

export default async function Model({ params }: P) {
  const id = (await params).model;
  const m = models[id];
  const rs = recipes.filter((r) => r.model === id);
  if (!m || !rs.length) notFound();
  const rows = configurationRows(rs);
  return <main>
    <Link className="back" href="/models/">‹ all models</Link>
    <h1 className={s.title}><Logo family={m.logo ?? m.family} size={30} />{m.name}</h1>
    <div className={s.metadata}><span>{m.family}</span>{m.released && <span>Released {fmtDate(m.released)}</span>}{m.reasoning && <span>Reasoning</span>}{m.vision && <span>Vision</span>}</div>
    {m.about && <p className={s.intro}>{m.about}</p>}
    <div className="facts"><span><b>{new Set(rs.map((r) => r.card)).size}</b> hardware entries</span><span><b>{rows.length}</b> hardware configurations</span><span><b>{rs.length}</b> exact recipes</span></div>
    <nav className={s.links}><a href="#configurations">Hardware configurations</a><a href="#model-card">Hugging Face model card</a></nav>
    <section id="configurations"><h2>Hardware configurations</h2><p className="dim">Choose a hardware setup and recipe. Command opens the pinned launch and setup details.</p><ConfigurationTable rows={rows} /></section>
    <section id="model-card"><h2>Hugging Face model card</h2><ModelCard modelId={id} /></section>
  </main>;
}
