// Everything the site shows comes from ../dist/catalog.json, built by lab/catalog.py.
import catalog from "../../dist/catalog.json";
import hardwareFile from "../../dist/hardware.json";
import { steps as sdkSteps } from "../../sdk/js/index.js";

export type Proof = {
  at: string; on: string; gpu?: string | null; gates: string; tps: number | null; prefill?: number | null;
  proxy?: string; legacy?: boolean; log?: string; reported?: boolean; src?: string; claims?: string; note?: string;
};
export type Launch = {
  image: string; entrypoint: string | null; args: string[]; env: Record<string, string>; port: number; shm: string | null;
  weights: { repo: string; revision: string; at: string; layout?: string; files?: string | string[] | null } | { repo: string; revision: string; at: string; layout?: string; files?: string | string[] | null }[];
  config: { at: string; text: string } | null; ctx: number; seqs: number; vision: boolean; cards?: number; backend?: string | null;
  prepare?: { at: string; args: string[] }; kind?: string; machines?: number; build?: { repo: string; commit: string }; setup?: string; source?: string; install?: string;
  needs?: { host_ram_gb?: number; disk_gb?: number; fast_storage?: string; cpus?: number };
};
export type Recipe = { key: string; slug: string; model: string; weights: string; engine: string; profile: string; card: string; proof: Proof[]; launch: Launch };
export type Model = { family: string; name: string; released: string; reasoning: boolean; vision: boolean; about?: string; good_for?: string; logo?: string; hf?: string };
/** One way to run a card: one card, several in one machine, or several machines. Picks and more are recipe keys. */
export type Setup = { cards: number; machines: number; label: string; vram_gb: number; picks: string[]; more: string[] };
export type Card = { id: string; name: string; vendor: string; backend: string; vram_gb: number; bandwidth_gb_s: number | null; picks: string[]; more?: string[]; setups: Setup[] };

type Raw = { models: Record<string, Model>; builds: Record<string, { format: string; size_gb: number }>; cards: Record<string, Omit<Card, "id">>; recipes: Record<string, Omit<Recipe, "key" | "slug">> };
const raw = catalog as unknown as Raw;

export const models = raw.models;
export const builds = raw.builds ?? {};
export const cards: Card[] = Object.entries(raw.cards)
  .map(([id, c]) => ({ id, ...c }))
  .sort((a, b) => vendorRank(a.vendor) - vendorRank(b.vendor) || b.vram_gb - a.vram_gb || a.name.localeCompare(b.name));
export const recipes: Recipe[] = Object.entries(raw.recipes).map(([key, r]) => ({ key, slug: key.split("/").pop()!, ...r }));

function vendorRank(v: string) { return ["nvidia", "amd", "intel", "apple", "cpu"].indexOf(v); }

/** A spec sheet from data/registry/hardware, built into ../dist/hardware.json by lab/hardware.py. */
export type Specs = {
  name: string; vendor: string; kind: string; family: string; vram_gb: number; bandwidth_gb_s: number | null; products: string[];
  memory: [string, string][]; compute: [string, string][]; tflops: [string, number | null, number | null][]; price: [string, string][];
  listings: { what: string; price: string; at: string; publisher: string; url: string }[]; sources: { publisher: string; url: string; at: string }[];
};
const specsById = (hardwareFile as unknown as { hardware: Record<string, Specs> }).hardware;
export const specs = (id: string): Specs | undefined => specsById[id];
/** Hardware with a spec sheet but no recipe yet, sorted like the cards. */
export const bare = Object.entries(specsById).filter(([id]) => !raw.cards[id]).map(([id, h]) => ({ id, ...h }))
  .sort((a, b) => vendorRank(a.vendor) - vendorRank(b.vendor) || b.vram_gb - a.vram_gb || a.name.localeCompare(b.name));

export const VENDOR: Record<string, string> = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel", apple: "Apple", cpu: "CPU" };
export const memoryLabel = (c: { vendor: string; vram_gb: number }) => c.vendor === "cpu" ? "System RAM" : `${c.vram_gb} GB`;
export const card = (id: string) => cards.find((c) => c.id === id);
export const recipe = (cardId: string, slug: string) => recipes.find((r) => r.card === cardId && r.slug === slug);
const byKey = (keys: string[]) => keys.map((k) => recipes.find((r) => r.key === k)!).filter(Boolean);
export const picks = (c: Card | Setup) => byKey(c.picks);
export const more = (c: Card | Setup) => byKey(c.more ?? []);
/** The setup a card leads with: one card, or its smallest group when it only runs in one (4 × CMP 170HX). */
export const lead = (c: Card): Card | Setup => (c.picks.length ? c : c.setups[0]);
/** "RTX PRO 6000 Blackwell", "4 × RTX PRO 6000 Blackwell in one machine", "2 × DGX Spark GB10". */
export function setupName(name: string, cards = 1, machines = 1) {
  if (machines > 1) return `${machines} × ${name}${cards > 1 ? `, ${cards} cards each` : ""}`;
  return cards > 1 ? `${cards} × ${name} in one machine` : name;
}
/** The setup a recipe runs on, named, with its total memory. */
export function setupOf(r: Recipe) {
  const c = card(r.card)!, n = r.launch.cards ?? 1, m = r.launch.machines ?? 1;
  return { cards: n, machines: m, name: setupName(short(c.name), n, m), vram_gb: c.vram_gb * n * m };
}
/** The setups beyond one card, as "4×" or "2 machines". */
export const setupTag = (s: Setup) => (s.machines > 1 ? `${s.machines} machines` : `${s.cards}×`);
export const model = (r: Recipe): Model => models[r.model] ?? { family: "", name: r.model, released: "", reasoning: false, vision: false };
export const engineKind = (r: Recipe) => r.slug.slice(r.model.length + 1).replace(/\.\d+k(\.\d+x)?$/, "");
export const weightsList = (l: Launch) => (Array.isArray(l.weights) ? l.weights : [l.weights]).filter((w) => w && w.repo);

/** What a recipe proved, in plain words. */
export function status(r: Recipe) {
  const p = r.proof[0];
  if (p.reported) return { label: `Reported by ${p.on === "miaai-lab" ? "MiaAI-Lab" : p.on}`, tone: "dim", detail: `Published by ${p.on === "miaai-lab" ? "MiaAI-Lab" : p.on} in ${p.src}${p.tps ? `, where it reports ${p.tps} tok/s` : ""}. Our six checks have not run on it yet.${p.note ? ` ${p.note}` : ""}` };
  if (p.legacy) return { label: "Earlier check", tone: "dim", detail: "Passed the older check (loads and chats); a full six-check run is pending." };
  if (p.proxy) return { label: "Tested on a sibling", tone: "warm", detail: `No ${cardName(r.card)} is rentable; this ran on the ${cardName(p.proxy)}, the same chip family.` };
  return { label: "Tested on this card", tone: "ok", detail: `Passed all six checks on a real ${p.gpu ?? cardName(r.card)} on ${fmtDate(p.at)}.` };
}
export const gates = (r: Recipe) => new Set(r.proof[0].gates.split(" "));
/** What a recipe can do: what our checks proved, or for a reported one what its publisher says. */
export const claims = (r: Recipe) => new Set(r.proof[0].reported ? (r.proof[0].claims ?? "").split(" ") : r.proof[0].gates.split(" "));
export const short = (name: string) => name.replace(/^(NVIDIA|AMD|Intel)\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+(Generation|GPU)\b/gi, "").replace(/Laptop$/, "Laptop").trim();
export const cardName = (id: string) => (card(id) ? short(card(id)!.name) : id);
export const ctxLabel = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}K` : `${n}`);
export const fmtDate = (d: string) => (d ? new Date(d + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }) : "");

/** The format as people say it: "EXL3 3 bpw", "GGUF Q4_K_M", "FP8". */
export function format(r: Recipe) {
  const b = builds[r.weights]?.format;
  if (b) { const m = b.match(/EXL3 · (?:SC )?(\d+(?:\.\d+)?)\s*bpw/); if (m) return `EXL3 ${Number(m[1])} bpw`; }
  if (b?.startsWith("MLX · ")) return b.replace(" · ", " ");
  const w = weightsList(r.launch)[0];
  const s = `${w?.repo ?? ""} ${w?.at ?? ""} ${r.engine} ${r.slug}`.toLowerCase();
  const bpw = s.match(/(\d(?:\.\d+)?)\s*-?bpw/) ?? s.match(/(\d)bpw/);
  if (s.includes("exl3")) return `EXL3${bpw ? ` ${Number(bpw[1])} bpw` : ""}`;
  const q = s.replace(/-/g, "_").match(/\b(iq\d_[a-z]+|q\d_k(?:_[msl])?|q\d_0)\b/);
  if (q) return `GGUF ${q[1].toUpperCase()}`;
  if (s.includes("ptq1")) return "Ternary";
  if (s.includes("fp8")) return "FP8";
  if (s.includes("nvfp4")) return "NVFP4";
  return "";
}

/** Why a pick is in the top three on its card: the first is recommended, the others say what they are best at. */
export function role(r: Recipe, all: Recipe[]) {
  if (all[0]?.key === r.key) return "Recommended";
  const fastest = [...all].sort((a, b) => (b.proof[0].tps ?? 0) - (a.proof[0].tps ?? 0))[0];
  if (fastest.key === r.key) return "Fastest";
  const longest = [...all].sort((a, b) => b.launch.ctx - a.launch.ctx)[0];
  if (longest.key === r.key && longest.launch.ctx > all[0].launch.ctx) return "Longest context";
  if (model(r).family !== model(all[0]).family) return "Another family";
  return "Alternative";
}

/** The exact steps to run a recipe by hand: the SDK's, so the site and every client say the same thing. */
export const steps = (r: Recipe) => sdkSteps(r);

export const stats = {
  gpus: cards.length,
  hardware: cards.length + bare.length,
  recipes: recipes.length,
  tested: recipes.filter((r) => !r.proof[0].legacy && !r.proof[0].proxy && !r.proof[0].reported).length,
  reported: recipes.filter((r) => r.proof[0].reported).length,
  models: new Set(recipes.map((r) => r.model)).size,
};
