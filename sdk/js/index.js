// Local AI registry, JavaScript. No dependencies; works in Node 18+, Deno, Bun and browsers.
//   const r = await pick({ gpu: "NVIDIA GeForce RTX 4090", vram: 24 });
//   console.log(command(r));
export const URL = "https://local.sybilsolutions.ai/api/v2/catalog.json";

let cached;
/** The whole catalog: cards (with their picks), recipes (with rendered launches), models. */
export async function catalog(url = URL) {
  return (cached ??= fetch(url).then((r) => { if (!r.ok) throw new Error(`catalog: ${r.status}`); return r.json(); }));
}

const norm = (s) => s.toLowerCase().replace(/\((r|tm)\)/g, "").replace(/geforce|nvidia|amd|radeon|intel|graphics|[^a-z0-9]/g, "");
const names = (id, c) => [norm(c.name), norm(id), ...(c.match?.names ?? [])];
/** The card id for a GPU as the driver names it (nvidia-smi, amd-smi, lspci) and its memory in GB. */
export function detect(cat, gpu, vram) {
  const n = norm(gpu);
  const hits = Object.entries(cat.cards).filter(([id, c]) => names(id, c).some((x) => x && n.includes(x)) && (!vram || Math.abs(c.vram_gb - vram) <= 2));
  const best = ([id, c]) => Math.max(...names(id, c).filter((x) => n.includes(x)).map((x) => x.length));
  return hits.sort((a, b) => best(b) - best(a) ||
    (vram ? Math.abs(a[1].vram_gb - vram) - Math.abs(b[1].vram_gb - vram) : 0))[0]?.[0] ?? null;
}

/** The recommended recipe for a GPU (or null); `all: true` returns the card's top three. */
export async function pick({ gpu, vram, all = false }, cat) {
  cat ??= await catalog();
  const id = detect(cat, gpu, vram);
  if (!id) return all ? [] : null;
  const rs = cat.cards[id].picks.map((k) => ({ key: k, ...cat.recipes[k] }));
  return all ? rs : rs[0];
}

const GPU = { cpu: "", nvidia: "--gpus all", "amd-rocm": "--device /dev/kfd --device /dev/dri", "amd-vulkan": "--device /dev/dri", "intel-xpu": "--device /dev/dri" };
const q = (s) => (/^[\w@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`);
const value = (s) => /^\$\{[A-Z_][A-Z0-9_]*\}$/.test(s) ? `"${s}"` : q(s);
const files = (w) => (Array.isArray(w.files) ? w.files : w.files ? [w.files] : []).map((f) => ` ${q(f)}`).join("");
const weights = (l) => (Array.isArray(l.weights) ? l.weights : [l.weights]).filter((w) => w && w.repo);
const resources = (l) => {
  const r = l.resources ?? {};
  return [...(r.memoryBytes ? [`--memory ${r.memoryBytes}`] : []), ...(r.memorySwapBytes ? [`--memory-swap ${r.memorySwapBytes}`] : []),
    ...(r.memlockUnlimited ? ['--ulimit memlock=-1:-1'] : []), ...(r.ipcLock ? ['--cap-add IPC_LOCK'] : [])];
};

const pairs = (a) => {
  const args = [];
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = a[i + 1];
    if (x.startsWith("-") && y !== undefined && !y.startsWith("-")) { args.push(`${x} ${q(y)}`); i++; } else args.push(q(x));
  }
  return args;
};

/** A program on the host. One with pinned packages or weights (MLX, oMLX or llama.cpp on Apple silicon) also downloads
 *  its weights, writes its launcher and raises the GPU wired-memory limit before it starts. */
function hostSteps(l) {
  const env = Object.entries(l.env ?? {}).map(([k, v]) => `${k}=${value(v)}`);
  if (!l.pip && !l.weights?.length) return [{ title: "Install", code: install(l.install) }, { title: "Start the server", code: [...env, ...l.command.map(q)].join(" ") }];
  const out = [l.pip ? { title: "Install the engine", code: `python3 -m pip install ${l.pip.map(q).join(" ")}` } : { title: "Install the engine", code: install(l.install) }];
  if (l.pip && /^https:\/\/\S+\.tar\.gz#sha256=[0-9a-f]{64}$/.test(l.install ?? "")) out.push({ title: "Install the native engine", code: install(l.install) });
  if (l.weights?.length) out.push({ title: "Download the weights", code: l.weights.map((w) => `hf download ${w.repo}${files(w)} \\\n  --revision ${w.revision} \\\n  --local-dir ${w.at}`).join("\n\n") });
  if (l.config) out.push({ title: "Write the launcher", code: `${l.config.at.includes("/") ? `mkdir -p ${l.config.at.replace(/\/[^/]*$/, "")}\n` : ""}cat > ${l.config.at} <<'EOF'\n${l.config.text.trimEnd()}\nEOF` });
  if (l.sysctl) out.push({ title: "Raise the GPU memory limit (resets on reboot)", code: Object.entries(l.sysctl).map(([k, v]) => `sudo sysctl ${k}=${v}`).join("\n") });
  const run = [...env, l.command[0], ...pairs(l.command.slice(1))];
  out.push({ title: "Start the server", code: run.filter(Boolean).join(" \\\n  ") });
  return out;
}

/** The steps to run a recipe by hand: download the weights, write the config, start the container. */
export function steps(r) {
  const l = r.launch, name = (r.key ?? r.model).split("/").pop();
  const out = l.setup ? [{ title: "Set up first", code: `# ${l.setup}\n# Full instructions: ${l.source}` }] : [];
  if (l.kind === "host") return [...out, ...hostSteps(l)];
  let image = l.image;
  if (l.build) {  // an image built from the source's Dockerfile at a pinned commit
    image = `local-ai/${name}`;
    const value = (s) => /^\$\{[A-Z_][A-Z0-9_]*\}$/.test(s) ? `"${s}"` : q(s);
const files = (l.build.dockerfile ?? "Dockerfile").split(" + ");
    out.push({ title: "Build the image", code: [install(`https://github.com/${l.build.repo}/tree/${l.build.commit}`), ...files.map((d) => `docker build -t ${image} -f ${d} .`)].join("\n") });
  }
  const prepared = `~/models/${name}-prepared-${image.slice(-12)}`;
  const mounts = [], dl = [];
  for (const w of weights(l)) {
    const dir = `~/models/${w.repo.split("/")[1]}-${w.revision.slice(0, 8)}`;
    if (w.layout === "hub") { dl.push(`hf download ${w.repo}${files(w)} \\\n  --revision ${w.revision}`); if (!mounts.includes("-v ~/.cache/huggingface:/root/.cache/huggingface")) mounts.push("-v ~/.cache/huggingface:/root/.cache/huggingface"); }  // several hub weights share one cache mount
    else { dl.push(`hf download ${w.repo}${files(w)} \\\n  --revision ${w.revision} \\\n  --local-dir ${dir}`); mounts.push(`-v ${dir}:${w.at}:ro`); }
  }
  if (dl.length) out.push({ title: "Download the weights", code: dl.join("\n\n") });
  if (l.config) {
    out.push({ title: "Write the server config", code: `cat > ${name}.yml <<'EOF'\n${l.config.text.trimEnd()}\nEOF` });
    mounts.push(`-v $PWD/${name}.yml:${l.config.at}:ro`);
  }
  const ep = Array.isArray(l.entrypoint) ? l.entrypoint : l.entrypoint ? [l.entrypoint] : [];
  const all = [...ep.slice(1), ...l.args], args = [];
  for (let i = 0; i < all.length; i++) {
    const a = all[i], b = all[i + 1];
    if (a.startsWith("-") && b !== undefined && !b.startsWith("-")) { args.push(`${a} ${value(b)}`); i++; } else args.push(value(a));
  }
  if (l.prepare) {
    const options = [...(l.flags ?? []), ...resources(l), ...(l.shm ? [`--shm-size ${l.shm}`] : []),
      ...Object.entries(l.env ?? {}).filter(([k]) => k !== 'NVIDIA_VISIBLE_DEVICES').map(([k, v]) => `-e ${k}=${value(v)}`), ...(ep.length ? [`--entrypoint ${q(ep[0])}`] : [])];
    const prep = ["docker run --rm", l.prepare.gpu === false ? "" : GPU[l.backend ?? "nvidia"] ?? GPU.nvidia, '--user "$(id -u):$(id -g)"', "-e HOME=/tmp",
      ...options, `-v ${prepared}:${l.prepare.at}`, ...mounts, image, ...[...ep.slice(1), ...l.prepare.args].map(q)];
    out.push({ title: "Prepare the model (first start)", code: `mkdir -p ${prepared}\n` + prep.filter(Boolean).join(" \\\n  ") });
    mounts.unshift(`-v ${prepared}:${l.prepare.at}:ro`);
    if (l.prepare.verifyArgs) out.push({ title: "Verify the prepared model", code: ['docker run --rm', '--user "$(id -u):$(id -g)"', '-e HOME=/tmp',
      ...options, ...mounts, image, ...[...ep.slice(1), ...l.prepare.verifyArgs].map(q)].join(' ') });
  }
  const run = ["docker run --rm", GPU[l.backend ?? "nvidia"] ?? GPU.nvidia, `-p 8000:${l.port}`, ...(l.flags ?? []), ...resources(l), ...(l.shm ? [`--shm-size ${l.shm}`] : []),
    ...Object.entries(l.env ?? {}).map(([k, v]) => `-e ${k}=${value(v)}`), ...mounts, ...(ep.length ? [`--entrypoint ${q(ep[0])}`] : []), image, ...args];
  out.push({ title: l.machines ? `Start the server on each of the ${l.machines} machines (NODE_RANK 0 to ${l.machines - 1})` : "Start the server", code: run.filter(Boolean).join(" \\\n  ") });
  return out;
}

/** Getting a source: a pinned GitHub tree becomes a clone at that commit. */
function install(url) {
  const m = url.match(/^https:\/\/github\.com\/([\w.-]+)\/([\w.-]+)\/tree\/([0-9a-f]{40})$/);
  if (m) return `git clone https://github.com/${m[1]}/${m[2]} && cd ${m[2]} && git checkout ${m[3]}`;
  const t = url.match(/^(https:\/\/\S+\/([\w.-]+\.tar\.gz))#sha256=([0-9a-f]{64})$/);  // a release archive, checked
  return t ? `curl -L -o ${t[2]} ${t[1]}\necho '${t[3]}  ${t[2]}' | shasum -a 256 -c\ntar xzf ${t[2]}` : `# ${url}`;
}

/** All the steps as one shell script. */
export const command = (r) => steps(r).map((s) => `# ${s.title}\n${s.code}`).join("\n\n");
