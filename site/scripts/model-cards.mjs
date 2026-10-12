// Prefetches each used model's real Hugging Face model card (README) and caches the raw
// Markdown into lib/model-cards.data.json, so the static build renders real text without
// depending on the Hub at build time. Run by `npm run build` before `next build`.
//
// A model with a canonical `hf` repo uses it directly. The many models whose `hf` is absent
// resolve their original repository from the catalog name plus the weight upstreams and their
// Hugging Face `cardData.base_model` chain: we gather candidate repos (the weight repos and
// their base models), score them against the model's name, and prefer the non-quantized repo
// whose name actually matches. Every candidate is source-verified against the live API; no
// repo name is invented. Gated/missing cards are reported honestly against the real repo.
//
// Network is bounded: short timeouts, limited concurrency, a per-model API-call budget, and
// the cache is incremental — an existing successful entry for the same repo is reused, so
// repeat builds do not depend on rate limits. Pass `--refresh` to re-fetch everything.
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const HF = "https://huggingface.co";
const TIMEOUT_MS = 20000;
const CONCURRENCY = 6;
const API_BUDGET = 16; // max Hub API calls spent resolving one model
const MAX_DEPTH = 2; // how far to follow the base_model chain
const FORCE = process.argv.includes("--refresh");

const catURL = new URL("../../dist/catalog.json", import.meta.url);
const outURL = new URL("../lib/model-cards.data.json", import.meta.url);
const cat = JSON.parse(readFileSync(catURL));
const models = cat.models ?? {};
const recipes = cat.recipes ?? {};

// Used models and their weight upstreams, in catalog order.
const used = new Map(); // modelId -> [{repo, revision}]
for (const r of Object.values(recipes)) {
  const id = r.model;
  if (!used.has(id)) used.set(id, []);
  const w = r.launch?.weights;
  const arr = Array.isArray(w) ? w : w ? [w] : [];
  for (const it of arr) if (it?.repo && !used.get(id).some((x) => x.repo === it.repo)) used.get(id).push({ repo: it.repo, revision: it.revision || "main" });
}

const prior = !FORCE && existsSync(outURL) ? JSON.parse(readFileSync(outURL)) : { cards: {} };
const priorCards = prior.cards ?? {};

async function hget(url, json) {
  const res = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_MS), headers: { "user-agent": "sybil-registry-site" } });
  return { ok: res.ok, status: res.status, body: res.ok ? (json ? await res.json() : await res.text()) : null };
}

const QUANT = /(NVFP4|FP4|GGUF|EXL3|EXL2|AWQ|GPTQ|FP8|INT4|INT8|W4A16|W8A|TR3|DFlash|MTP|AQLM|GSQ|RCO|MLX|4bit|8bit|bf16|hybrid|Slotpack|Bare-?Speed|MTPLX|Uncensored|-Spark\b|-pi\b)/gi;
const base = (repo) => repo.split("/").pop();
const norm = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
const normNoQuant = (repo) => norm(base(repo).replace(QUANT, ""));
const isQuant = (repo) => { QUANT.lastIndex = 0; return QUANT.test(repo); };

async function getInfo(repo, state) {
  if (state.info.has(repo)) return state.info.get(repo);
  if (state.calls >= API_BUDGET) return undefined;
  state.calls++;
  const r = await hget(`${HF}/api/models/${repo}`, true).catch(() => ({ ok: false, status: 0, body: null }));
  const info = r.ok ? r.body : { __status: r.status };
  state.info.set(repo, info);
  return info;
}

// Fetch a repo's README pinned to its current commit; classify failures honestly.
async function fetchCard(repo, state) {
  const info = await getInfo(repo, state);
  if (!info || info.__status) {
    const s = info?.__status;
    const gated = s === 401 || s === 403;
    return { repo, repoUrl: `${HF}/${repo}`,
      status: gated ? "gated" : s === 404 ? "not-found" : "error",
      note: gated ? "Source repository is gated or private on Hugging Face." :
        s === 404 ? "Source repository not found on Hugging Face." : `Hugging Face API returned HTTP ${s ?? "error"}.` };
  }
  const sha = info.sha || "main";
  const r = await hget(`${HF}/${repo}/raw/${sha}/README.md`, false).catch(() => ({ ok: false, status: 0, body: null }));
  if (!r.ok || !r.body?.trim()) {
    return { repo, repoUrl: `${HF}/${repo}`, revision: sha, gated: !!info.gated,
      status: r.ok || r.status === 404 ? "no-card" : "error",
      note: r.ok || r.status === 404 ? "Source repository has no README model card." : `Could not fetch README (HTTP ${r.status}).` };
  }
  return { repo, repoUrl: `${HF}/${repo}`, revision: sha, gated: !!info.gated, status: "ok",
    fetchedAt: new Date().toISOString(), markdown: r.body };
}

async function resolveCard(id, weights) {
  const m = models[id] ?? {};
  const canonical = m.hf && m.hf.includes("/") ? m.hf : null;
  const state = { info: new Map(), calls: 0 };

  // Candidate repos: canonical first, then weight repos, then their base_model chain.
  const cands = [];
  const add = (repo, depth) => { if (repo?.includes("/") && !cands.some((c) => c.repo === repo)) cands.push({ repo, depth }); };
  if (canonical) add(canonical, -1);
  for (const w of weights) add(w.repo, 0);
  for (let i = 0; i < cands.length; i++) {
    const c = cands[i];
    if (c.depth >= MAX_DEPTH || state.calls >= API_BUDGET) continue;
    const info = await getInfo(c.repo, state);
    if (!info || info.__status) continue;
    let bases = info.cardData?.base_model ?? [];
    bases = (Array.isArray(bases) ? bases : [bases]).filter((b) => typeof b === "string" && b.includes("/"));
    for (const b of bases) add(b, c.depth + 1);
  }

  // No catalog repo: find a real public card for the same model by Hub search (bounded, 1 call).
  const searched = new Set();
  if (!canonical && m.name && state.calls < API_BUDGET) {
    state.calls++;
    const r = await hget(`${HF}/api/models?search=${encodeURIComponent(m.name)}&limit=10`, true).catch(() => ({ ok: false, body: null }));
    for (const hit of (r.ok && Array.isArray(r.body) ? r.body : [])) { if (!cands.some((c) => c.repo === hit.id)) searched.add(hit.id); add(hit.id, 1); }
  }

  if (!cands.length) {
    return { modelId: id, name: m.name ?? id, repo: null, resolvedVia: null, status: "no-source",
      note: "No catalog repo, weight upstream, or matching public Hugging Face card found for this model." };
  }

  // Score by name match to the catalog model name; prefer the canonical, then exact/substring
  // name matches, then non-quantized repos, then shallower depth.
  const target = norm(m.name) || normNoQuant(weights[0]?.repo || "");
  const score = (c) => {
    if (c.depth < 0) return 100; // catalog-declared canonical
    const cn = normNoQuant(c.repo);
    const match = cn === target ? 3 : cn && (cn.includes(target) || target.includes(cn)) ? 2 : 0;
    return match * 10 + (isQuant(c.repo) ? 0 : 3) + (MAX_DEPTH - c.depth);
  };
  cands.sort((a, b) => score(b) - score(a));

  // Try candidates best-first; return the first readable card, else the best honest failure.
  let firstFail = null;
  for (const c of cands) {
    const res = await fetchCard(c.repo, state);
    const resolvedVia = c.depth < 0 ? "catalog.hf" : c.depth === 0 ? "weight upstream" : searched.has(c.repo) ? "Hugging Face search" : "base_model chain";
    if (res.status === "ok") return { modelId: id, name: m.name ?? id, resolvedVia, ...res };
    if (!firstFail) firstFail = { modelId: id, name: m.name ?? id, resolvedVia, ...res };
  }
  return firstFail;
}

async function run() {
  const ids = [...used.keys()].sort();
  const results = {};
  let i = 0;
  async function worker() {
    while (i < ids.length) {
      const id = ids[i++];
      const p = priorCards[id];
      const m = models[id] ?? {};
      if (!FORCE && p && p.status === "ok" && p.markdown && (!m.hf || p.repo === m.hf)) {
        results[id] = p; console.log(`cache     ${id} -> ${p.repo}`); continue;
      }
      try {
        const card = await resolveCard(id, used.get(id));
        results[id] = card;
        console.log(`${card.status.padEnd(9)} ${id} -> ${card.repo ?? "(none)"}`);
      } catch (e) {
        results[id] = { modelId: id, name: m.name ?? id, repo: null, status: "error", note: String(e?.message || e) };
        console.log(`error     ${id}: ${e?.message || e}`);
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, ids.length) }, worker));

  const ordered = {};
  for (const id of ids) ordered[id] = results[id];
  const ok = ids.filter((id) => results[id].status === "ok").length;
  if (JSON.stringify(ordered) !== JSON.stringify(priorCards)) {
    writeFileSync(outURL, JSON.stringify({ generatedAt: new Date().toISOString(), cards: ordered }, null, 2) + "\n");
    console.log(`model-cards: ${ok}/${ids.length} cards fetched; wrote ${outURL.pathname}`);
  } else console.log(`model-cards: ${ok}/${ids.length} cards cached; source unchanged`);
  const bad = ids.filter((id) => results[id].status !== "ok");
  if (bad.length) console.log("unavailable:", bad.map((id) => `${id} (${results[id].status}: ${results[id].repo ?? "no repo"})`).join(", "));
}

run().catch((e) => { console.error(e); process.exit(1); });
