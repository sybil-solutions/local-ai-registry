import { command as sdkCommand } from "../../sdk/js/index.js";
import { cards, models, recipes, short, setupOf, format, engineKind, status, fmtDate, ctxLabel, type Recipe } from "./registry";

export type ConfigurationVariant = {
  key: string; label: string; href: string; engine: string; format: string;
  context: number; decode: number | null; prefill: number | null;
  status: string; statusDetail: string; testedAt: string; command: string;
  requirements: string; machines: number;
};
export type ConfigurationRow = {
  id: string; hardwareId: string; hardware: string; modelId: string; model: string;
  memory: number; cards: number; machines: number; host: string; variants: ConfigurationVariant[];
};

const ENGINES: Record<string, string> = { vllm: "vLLM", sglang: "SGLang", tabbyapi: "TabbyAPI", exllamav3: "ExLlamaV3" };

/** A single shell invocation containing the SDK's exact, pinned launch steps. */
function oneCommand(r: Recipe): string {
  const script = sdkCommand(r) as string;
  let delimiter = "LOCAL_AI_RECIPE";
  while (script.split("\n").includes(delimiter)) delimiter += "_";
  return `bash -e <<'${delimiter}'\n${script}\n${delimiter}`;
}

const hardwareConfigId = (r: Recipe) => `${r.card}/${r.launch.cards ?? 1}/${r.launch.machines ?? 1}/${r.launch.needs?.host_ram_gb ?? ""}/${r.launch.needs?.disk_gb ?? ""}/${r.launch.needs?.fast_storage ?? ""}/${r.launch.needs?.cpus ?? ""}`;

function hostRequirements(r: Recipe): string {
  const needs = r.launch.needs;
  if (!needs) return "";
  return [
    needs.host_ram_gb != null ? `RAM ≥ ${needs.host_ram_gb} GB` : "",
    needs.disk_gb != null ? `${needs.fast_storage?.toUpperCase() ?? "Disk"} ≥ ${needs.disk_gb} GB` : "",
    needs.cpus != null ? `≥ ${needs.cpus} CPUs` : "",
  ].filter(Boolean).join(" · ");
}

export function configurationRows(selected: Recipe[]): ConfigurationRow[] {
  const groups = new Map<string, ConfigurationRow>();
  for (const r of selected) {
    const c = cards.find((c) => c.id === r.card);
    if (!c) continue;
    const setup = setupOf(r);
    const id = `${r.model}/${hardwareConfigId(r)}`;
    let row = groups.get(id);
    if (!row) {
      row = { id, hardwareId: r.card, hardware: setup.name.replace(/ in one machine$/, ""), modelId: r.model,
        model: models[r.model]?.name ?? r.model, memory: setup.vram_gb,
        cards: setup.cards, machines: setup.machines, host: hostRequirements(r), variants: [] };
      groups.set(id, row);
    }
    const proof = r.proof[0];
    const evidence = status(r);
    const engine = engineKind(r);
    const quant = format(r);
    row.variants.push({ key: r.key, label: `${ENGINES[engine] ?? engine} · ${ctxLabel(r.launch.ctx)} ctx${quant ? ` · ${quant}` : ""}`,
      href: `/gpu/${r.card}/${r.slug}/`, engine, format: quant, context: r.launch.ctx,
      decode: proof.tps ?? null, prefill: proof.prefill ?? null, status: evidence.label,
      statusDetail: evidence.detail, testedAt: fmtDate(proof.at), command: oneCommand(r),
      requirements: r.launch.setup ?? "", machines: setup.machines });
  }
  return [...groups.values()].sort((a, b) => a.hardware.localeCompare(b.hardware) || a.model.localeCompare(b.model));
}

export type ModelDirectoryRow = {
  id: string; name: string; family: string; about: string; released: string;
  capabilities: string; hardware: number; configurations: number;
};
export function modelDirectoryRows(): ModelDirectoryRow[] {
  const used = new Map<string, { hardware: Set<string>; configurations: Set<string> }>();
  for (const r of recipes) {
    let counts = used.get(r.model);
    if (!counts) { counts = { hardware: new Set(), configurations: new Set() }; used.set(r.model, counts); }
    counts.hardware.add(r.card);
    counts.configurations.add(hardwareConfigId(r));
  }
  return [...used].map(([id, counts]) => {
    const m = models[id];
    return { id, name: m?.name ?? id, family: m?.family ?? "", about: m?.about ?? "",
      released: m?.released ?? "", capabilities: [m?.reasoning ? "Reasoning" : "", m?.vision ? "Vision" : ""].filter(Boolean).join(" · ") || "Text",
      hardware: counts.hardware.size, configurations: counts.configurations.size };
  });
}

export type HardwareDirectoryRow = {
  id: string; name: string; vendor: string; memory: number; bandwidth: number | null;
  models: number; configurations: number; decode: number | null;
};
export function hardwareDirectoryRows(extra: { id: string; name: string; vendor: string; vram_gb: number; bandwidth_gb_s: number | null }[]): HardwareDirectoryRow[] {
  const byHardware = new Map<string, Recipe[]>();
  for (const r of recipes) {
    const rs = byHardware.get(r.card) ?? [];
    rs.push(r); byHardware.set(r.card, rs);
  }
  return [...cards, ...extra].map((c) => {
    const rs = byHardware.get(c.id) ?? [];
    const speeds = rs.map((r) => r.proof[0].tps).filter((n): n is number => n != null);
    return { id: c.id, name: short(c.name), vendor: c.vendor, memory: c.vram_gb,
      bandwidth: c.bandwidth_gb_s, models: new Set(rs.map((r) => r.model)).size,
      configurations: new Set(rs.map(hardwareConfigId)).size,
      decode: speeds.length ? Math.max(...speeds) : null };
  });
}
