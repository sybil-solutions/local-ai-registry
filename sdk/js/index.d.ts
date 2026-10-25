export type Launch = { image: string; entrypoint: string | null; args: string[]; env: Record<string, string>; port: number; shm: string | null;
  prepare?: { at: string; args: string[]; verifyArgs?: string[]; gpu?: boolean; sizeGb?: number }; resources?: { memoryBytes?: number; memorySwapBytes?: number; memlockUnlimited?: boolean; ipcLock?: boolean };
  weights: Weights | Weights[]; config: { at: string; text: string } | null; ctx: number; seqs: number; vision: boolean; backend?: string | null; cards?: number;
  kind?: string; command?: string[]; install?: string; pip?: string[]; runtime?: Runtime };
/** A host `layout: "hub"` weights entry is a Hugging Face cache under `at`; `ref` (e.g. "main") is then written to name `revision`. */
export type Weights = { repo: string; revision: string; at: string; layout?: string; files?: string | string[] | null; ref?: string };
/** A pinned host runtime (host launches only, never with `pip`): an interpreter archive, its interpreter (a relative path
 *  inside `archive.into`) and a hash-locked offline wheel set; `dir` and `target` must not overlap. A launch with weights
 *  needs `layout: "hub"` weights and a `huggingface-hub` wheel. `steps` throws on anything else (docs/design.md). */
export type Runtime = {
  archive?: { url: string; sha256: string; into: string };
  python?: string;
  /** The lowest macOS major version (11 to 99): the first step stops on an older macOS, before any download. */
  macos?: number;
  packages?: { dir: string; target: string; wheels: { name: string; version: string; url: string; sha256: string }[] };
};
export type Proof = {
  at: string; on: string; gpu?: string | null; gates: string; tps: number | null; prefill?: number | null; proxy?: string; legacy?: boolean;
  /** Free text the lab or a publisher attached to the proof. */
  note?: string;
  /** true when the lab could only estimate tps (characters/4: the server had no tokenize route); the catalog writes only
   *  true, never false. Typed boolean so a consumer's own boolean proof type (the site's) stays assignable. */
  tps_estimated?: boolean;
};
export type Recipe = { key?: string; model: string; weights: string; engine: string; card: string; proof: Proof[]; launch: Launch };
export type Card = { name: string; vendor: string; backend: string; vram_gb: number; bandwidth_gb_s: number | null; picks: string[] };
export type Catalog = { schema: string; cards: Record<string, Card>; recipes: Record<string, Recipe>; models: Record<string, unknown> };
export declare const URL: string;
export declare function catalog(url?: string): Promise<Catalog>;
export declare function detect(cat: Catalog, gpu: string, vram?: number): string | null;
export declare function pick(q: { gpu: string; vram?: number; all?: false }, cat?: Catalog): Promise<Recipe | null>;
export declare function pick(q: { gpu: string; vram?: number; all: true }, cat?: Catalog): Promise<Recipe[]>;
export declare function steps(r: Recipe): { title: string; code: string }[];
export declare function command(r: Recipe): string;
