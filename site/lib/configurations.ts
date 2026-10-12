import { command as sdkCommand } from "../../sdk/js/index.js";
import { cards, models, recipes, short, setupOf, format, engineKind, status, fmtDate, ctxLabel, type Recipe } from "./registry";

export type ConfigurationVariant = {
  key: string; label: string; href: string; engine: string; format: string;
  context: number; decode: number | null; prefill: number | null;
  status: string; statusTone: string; statusDetail: string; testedAt: string; command: string;
  requirements: string; machines: number;
};
/** A documented host requirement, split for display: "Host RAM" / "≥ 59.1 GB free". */
export type HostNeed = { label: string; value: string };
export type ConfigurationRow = {
  id: string; hardwareId: string; hardware: string; cardName: string; modelId: string; model: string; family: string;
  memory: number; cards: number; machines: number; host: string; needs: HostNeed[]; variants: ConfigurationVariant[];
};

/** A single shell invocation containing the SDK's exact, pinned launch steps. */
function oneCommand(r: Recipe): string {
  const script = sdkCommand(r) as string;
  let delimiter = "LOCAL_AI_RECIPE";
  while (script.split("\n").includes(delimiter)) delimiter += "_";
  return `bash -e <<'${delimiter}'\n${script}\n${delimiter}`;
}

const hardwareConfigId = (r: Recipe) => `${r.card}/${r.launch.cards ?? 1}/${r.launch.machines ?? 1}/${r.launch.needs?.host_ram_gb ?? ""}/${r.launch.needs?.disk_gb ?? ""}/${r.launch.needs?.fast_storage ?? ""}/${r.launch.needs?.cpus ?? ""}`;

function hostNeeds(r: Recipe): HostNeed[] {
  const needs = r.launch.needs;
  if (!needs) return [];
  const storage = needs.fast_storage === "nvme" ? "NVMe" : needs.fast_storage ?? "";
  return [
    needs.host_ram_gb != null ? { label: "Host RAM", value: `≥ ${needs.host_ram_gb} GB free` } : null,
    needs.disk_gb != null ? { label: "Disk", value: `≥ ${needs.disk_gb} GB${storage ? ` ${storage}` : ""}` } : null,
    needs.cpus != null ? { label: "CPUs", value: `≥ ${needs.cpus} logical` } : null,
  ].filter((n): n is HostNeed => n != null);
}

function hostRequirements(r: Recipe): string {
  const needs = r.launch.needs;
  if (!needs) return "";
  return [
    needs.host_ram_gb != null ? `≥ ${needs.host_ram_gb} GB free host RAM` : "",
    needs.disk_gb != null ? `≥ ${needs.disk_gb} GB ${needs.fast_storage ?? ""} disk` : "",
    needs.cpus != null ? `≥ ${needs.cpus} logical CPUs` : "",
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
      row = { id, hardwareId: r.card, hardware: setup.name, cardName: short(c.name), modelId: r.model,
        model: models[r.model]?.name ?? r.model, family: models[r.model]?.family ?? "", memory: setup.vram_gb,
        cards: setup.cards, machines: setup.machines, host: hostRequirements(r), needs: hostNeeds(r), variants: [] };
      groups.set(id, row);
    }
    const proof = r.proof[0];
    const evidence = status(r);
    const engine = engineKind(r);
    const quant = format(r);
    row.variants.push({ key: r.key, label: [engine, quant, `${ctxLabel(r.launch.ctx)} context`].filter(Boolean).join(" · "),
      href: `/gpu/${r.card}/${r.slug}/`, engine, format: quant, context: r.launch.ctx,
      decode: proof.tps ?? null, prefill: proof.prefill ?? null, status: evidence.label, statusTone: evidence.tone,
      statusDetail: evidence.detail, testedAt: fmtDate(proof.at), command: oneCommand(r),
      requirements: r.launch.setup ?? "", machines: setup.machines });
  }
  // Short labels can collide (same engine, format and rounded context); fall back to the recipe slug there.
  for (const row of groups.values()) {
    const seen = new Map<string, number>();
    for (const v of row.variants) seen.set(v.label, (seen.get(v.label) ?? 0) + 1);
    for (const v of row.variants) if (seen.get(v.label)! > 1) v.label += ` · ${v.key.slice(v.key.lastIndexOf("/") + 1)}`;
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
