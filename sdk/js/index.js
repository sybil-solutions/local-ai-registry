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

// A pinned host runtime and Hugging Face cache weights: the same rules as lab/lab.py check_host_runtime and the Python SDK,
// checked again here so that nothing malformed is ever rendered into a shell command (fail closed).
const RT = {
  sha: /^[0-9a-f]{64}$/, rev: /^[0-9a-f]{40}$/,
  rel: /^(?!.*(?:^|\/)\.\.?(?:\/|$))[A-Za-z0-9_][A-Za-z0-9._+-]*(?:\/[A-Za-z0-9_][A-Za-z0-9._+-]*)*$/,
  url: /^https:\/\/[A-Za-z0-9.-]+(?:\/[A-Za-z0-9._~%+-]+)+$/,
  archive: /^[A-Za-z0-9][A-Za-z0-9._+-]*\.(?:tar\.gz|tgz)$/, wheel: /^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$/,
  name: /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/, version: /^[A-Za-z0-9][A-Za-z0-9.+!_-]*$/,
  ref: /^[A-Za-z0-9][A-Za-z0-9._-]*$/, repo: /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/,
};
const ok = (cond, msg) => { if (!cond) throw new Error(`invalid host runtime: ${msg}`); };
const is = (re, v) => typeof v === "string" && re.test(v);
const keys = (o) => (o && typeof o === "object" && !Array.isArray(o) ? Object.keys(o).sort().join(",") : null);
const canon = (n) => n.replace(/[-_.]+/g, "-").toLowerCase();
const urlFile = (u) => decodeURIComponent(u.slice(u.lastIndexOf("/") + 1));
/** Equal or one inside the other, ignoring case (macOS file systems do). */
const nested = (a, b) => { a = a.toLowerCase(); b = b.toLowerCase(); return a === b || a.startsWith(`${b}/`) || b.startsWith(`${a}/`); };
function checkHost(l) {
  const ws = weights(l), hubs = new Set();
  ok(new Set(ws.map((w) => w.layout === "hub")).size <= 1, "hub and directory weights mixed in one launch");
  for (const w of ws) {
    if ("ref" in w) ok(w.layout === "hub" && is(RT.ref, w.ref), `weights.ref ${w.ref}`);
    if (w.layout === "hub") {
      const key = JSON.stringify([String(w.at).toLowerCase(), String(w.repo).toLowerCase()]);
      ok(!hubs.has(key), `weights ${w.repo} listed twice in the cache ${w.at}`);
      hubs.add(key);
      ok(is(RT.repo, w.repo) && is(RT.rev, w.revision) && is(RT.rel, w.at), "hub weights repo, revision or cache dir");
      ok((Array.isArray(w.files) ? w.files : w.files ? [w.files] : []).every((f) => is(RT.rel, f)), "hub weights files");
    }
  }
  if (l.config && confined(l)) ok(is(RT.rel, l.config.at), "launcher path");  // a launcher the setup steps write: a plain relative path
  const rt = l.runtime;
  if (rt === undefined || rt === null) return checkPaths(l);
  ok(keys(rt) && Object.keys(rt).length && Object.keys(rt).every((k) => ["archive", "python", "packages"].includes(k)) && !l.pip?.length, "runtime keys");
  if ("archive" in rt) {
    const a = rt.archive;
    ok(keys(a) === "into,sha256,url", "runtime.archive keys");
    ok(is(RT.url, a.url) && is(RT.archive, urlFile(a.url)), `runtime.archive.url ${a.url}`);
    ok(is(RT.sha, a.sha256) && is(RT.rel, a.into), "runtime.archive sha256 or into");
  }
  if ("python" in rt) ok(is(RT.rel, rt.python) && "archive" in rt && rt.python.startsWith(`${rt.archive.into}/`), `runtime.python ${rt.python} is not inside runtime.archive.into`);
  if ("packages" in rt) {
    const pk = rt.packages;
    ok("python" in rt && keys(pk) === "dir,target,wheels", "runtime.packages keys");
    ok(is(RT.rel, pk.dir) && is(RT.rel, pk.target), "runtime.packages dir or target");
    ok(!nested(pk.dir, pk.target), "runtime.packages dir and target overlap");
    ok(Array.isArray(pk.wheels) && pk.wheels.length, "runtime.packages.wheels");
    const seen = new Set();
    for (const w of pk.wheels) {
      ok(keys(w) === "name,sha256,url,version", "wheel keys");
      ok(is(RT.name, w.name) && is(RT.version, w.version) && is(RT.url, w.url) && is(RT.sha, w.sha256), `wheel ${w.name}`);
      const f = urlFile(w.url), [dist = "", ver = ""] = f.split("-");
      ok(is(RT.wheel, f) && canon(dist) === canon(w.name) && ver === w.version, `wheel file ${f}`);
      ok(!seen.has(canon(w.name)), `wheel ${w.name} listed twice`);
      seen.add(canon(w.name));
    }
  }
  if (ws.length) ok("packages" in rt && rt.packages.wheels.some((w) => canon(w.name) === "huggingface-hub") && ws[0].layout === "hub",
    "weights with a runtime need hub layout and a pinned huggingface_hub wheel");
  checkPaths(l);
}

/** A host launch whose setup steps are confined (strict, staged, no links): one with a runtime or hub weights. */
const confined = (l) => (l.runtime !== undefined && l.runtime !== null) || weights(l).some((w) => w.layout === "hub");
/** What the setup steps publish besides the archive's entries: packages target and dir, host hub caches, launcher. */
function owned(l) {
  const pk = l.runtime?.packages, hub = weights(l).filter((w) => w.layout === "hub");
  return [...(pk ? [["packages.target", pk.target], ["packages.dir", pk.dir]] : []), ...hub.map((w) => ["weights.at", w.at]),
    ...(l.config && (l.runtime || hub.length) ? [["launcher", l.config.at]] : [])];
}
/** No step may publish over another's output: no two destinations equal or nest (hub caches may be shared), none is or
 *  holds runtime.archive.into, and none lies in the archive entry that holds the interpreter. */
function checkPaths(l) {
  const own = owned(l);
  own.forEach(([na, a], i) => own.slice(i + 1).forEach(([nb, b]) => ok((na === "weights.at" && nb === "weights.at" && a.toLowerCase() === b.toLowerCase()) || !nested(a, b), `${na} ${a} and ${nb} ${b} overlap`)));
  const a = l.runtime?.archive;
  if (a) {
    const py = l.runtime.python, entry = py ? `${a.into}/${py.slice(a.into.length + 1).split("/")[0]}` : null;
    for (const [n, x] of own) {
      ok(!(x.toLowerCase() === a.into.toLowerCase() || a.into.toLowerCase().startsWith(`${x.toLowerCase()}/`)), `${n} ${x} would replace runtime.archive.into`);
      ok(!entry || !nested(x, entry), `${n} ${x} overlaps the interpreter's archive entry ${entry}`);
    }
  }
}

const NOLINK = 'nolink() { for p; do if [ -L "$p" ]; then echo "local-ai: $p is a symlink; refusing to write through it" >&2; exit 1; fi; done; }';
const HUB = "import sys; sys.path.insert(0, sys.argv[1]); from huggingface_hub import snapshot_download; " +
  "snapshot_download(sys.argv[2], revision=sys.argv[3], cache_dir=sys.argv[4], allow_patterns=sys.argv[5:] or None, token=False)";
// Every link under a staged tree resolves inside it (relative, not dangling), every hard link stays inside it, and it holds
// no device, fifo, socket or set-id file. The same text in both SDKs.
const CONFINE = "confine() {\n  b=$(cd -P \"$1\" && pwd -P)\n  if [ -n \"$(find \"$1\" ! -type f ! -type d ! -type l)\" ]; then echo \"local-ai: $1 holds a device, fifo or socket\" >&2; exit 1; fi\n  if [ -n \"$(find \"$1\" -type f \\( -perm -4000 -o -perm -2000 \\))\" ]; then echo \"local-ai: $1 holds a setuid or setgid file\" >&2; exit 1; fi\n  find \"$1\" -type l | while IFS= read -r l; do\n    if [ ! -L \"$l\" ]; then echo \"local-ai: unexpected name under $1\" >&2; exit 1; fi\n    k=$(readlink \"$l\")\n    case $k in /*) echo \"local-ai: link $l is absolute\" >&2; exit 1;; esac\n    if [ ! -e \"$l\" ]; then echo \"local-ai: link $l is dangling\" >&2; exit 1; fi\n    if [ -d \"$l\" ]; then r=$(cd -P \"$l\" && pwd -P); else\n      case ${k##*/} in .|..) echo \"local-ai: link $l leaves $1\" >&2; exit 1;; esac\n      r=$(cd -P \"$(dirname \"$l\")\" && cd -P \"$(dirname \"$k\")\" && pwd -P)/${k##*/}\n    fi\n    case $r/ in \"$b\"/*) ;; *) echo \"local-ai: link $l leaves $1\" >&2; exit 1;; esac\n  done\n  find \"$1\" -type f -links +1 | while IFS= read -r f; do\n    if [ \"$(ls -ld \"$f\" | awk '{print $2}')\" != \"$(find \"$1\" -samefile \"$f\" | wc -l | tr -d ' ')\" ]; then echo \"local-ai: $f is hard-linked outside $1\" >&2; exit 1; fi\n  done\n}";
// The download sees no ambient Hugging Face token, config, cache or .netrc: only directories inside its staging cache.
const HUBENV = "hubenv() { for v in $(env | sed -n 's/^\\(HF_[A-Za-z0-9_]*\\)=.*/\\1/p; s/^\\(HUGGING_FACE_[A-Za-z0-9_]*\\)=.*/\\1/p'); do unset \"$v\"; done; unset NETRC; h=$(cd -P \"$1\" && pwd -P); HOME=$h/.home; HF_HOME=$h/.hf-home; HF_XET_CACHE=$h/.xet; TMPDIR=$h/.tmp; TMP=$h/.tmp; TEMP=$h/.tmp; HF_HUB_DISABLE_TELEMETRY=1; export HOME HF_HOME HF_XET_CACHE TMPDIR TMP TEMP HF_HUB_DISABLE_TELEMETRY; mkdir -p \"$HOME\" \"$TMPDIR\"; }";
// pip sees no config file (PIP_CONFIG_FILE=/dev/null turns off every one, which --isolated does not), and its temp files,
// home and cache are inside the packages staging directory.
const PIPENV = "pipenv() { h=$(cd -P \"$1\" && pwd -P)/.pip; mkdir -p \"$h/tmp\" \"$h/home\" \"$h/cache\"; PIP_CONFIG_FILE=/dev/null; TMPDIR=$h/tmp; TMP=$h/tmp; TEMP=$h/tmp; HOME=$h/home; XDG_CACHE_HOME=$h/cache; XDG_CONFIG_HOME=$h/home/.config; export PIP_CONFIG_FILE TMPDIR TMP TEMP HOME XDG_CACHE_HOME XDG_CONFIG_HOME; }";
// Every step's exit, by success, failure or a signal (TERM, INT, HUP), goes through fin: while a publication is unfinished
// (pub set) the step's back() puts the previous installation back first, and if that fails the backup is kept (named on
// stderr) and the step fails; only then are the staging directories removed. fin ignores TERM, INT and HUP first (the
// rollback's own children inherit that), so a repeated signal cannot cut the rollback short. The same text in both SDKs.
const TXN = "pub= c= o= s= t= w=\nfin() { rc=${1:-$?}; trap '' TERM INT HUP; trap - EXIT; if [ -n \"$pub\" ] && ! back; then for k in ${s:+\"$s/old\"} ${o:+\"$o/old\"}; do if [ -e \"$k\" ]; then echo \"local-ai: could not restore it; the previous copy is kept in $k\" >&2; fi; done; exit 1; fi; for x in \"$c\" \"$o\" \"$s\" \"$t\" \"$w\"; do if [ -n \"$x\" ]; then rm -rf \"$x\"; fi; done; exit \"$rc\"; }\ntrap fin EXIT; trap 'fin 143' TERM; trap 'fin 130' INT; trap 'fin 129' HUP";
const HUBUNDO = 'hubundo() { echo "local-ai: could not publish $1; restoring the previous one" >&2; exit 1; }';
// the hub model directory being published: m (live) and ms (staged); the new one is placed once ms is gone
const HUBBACK = "back() { ok=0; if [ ! -e \"$ms\" ]; then rm -rf \"$m\" || ok=1; fi; if [ -e \"$o/old\" ]; then mv \"$o/old\" \"$m\" || ok=1; fi; return $ok; }";
/** `a/b/c` -> `a a/b a/b/c`: every directory a write to `path` passes through, quoted. */
const prefixes = (path) => path.split("/").map((_, i, p) => q(p.slice(0, i + 1).join("/"))).join(" ");
const parent = (path) => (path.includes("/") ? path.replace(/\/[^/]*$/, "") : ".");
/** One step as one subshell that stops at its first failure and returns non-zero (pasted alone or in a script). */
const strict = (lines) => `(\nset -e\n${lines.join("\n")}\n)`;

/** The pinned interpreter archive, then the hash-locked offline wheel install. Each is staged in a fresh directory
 *  beside its destination and checked in full (hashes, entry names, links, the interpreter) before anything live is
 *  touched; it is then published by rename, and a failed rename puts back everything it had moved. */
function runtimeSteps(l) {
  const rt = l.runtime, out = [];
  if (rt.archive) {
    const a = rt.archive, into = a.into, f = urlFile(a.url), d = q(into);
    const keep = owned(l).map(([, x]) => x).filter((x) => x.toLowerCase().startsWith(`${into.toLowerCase()}/`));  // ignoring case, as macOS does
    out.push({ title: "Install the runtime", code: strict([
      NOLINK,
      TXN,
      `nolink ${prefixes(into)}`,
      `mkdir -p ${d}`,
      `s=$(mktemp -d ${d}/.local-ai-runtime.XXXXXX)`,
      `curl -q -fL -o "$s"/${q(f)} ${q(a.url)}`,
      `(cd "$s" && echo '${a.sha256}  ${f}' | shasum -a 256 -c -)`,
      'mkdir "$s/x" "$s/old"',
      `tar -tzf "$s"/${q(f)} | while IFS= read -r m; do case $m in /*|..|../*|*/..|*/../*) echo "local-ai: archive member $m leaves the archive" >&2; exit 1;; esac; done`,
      `tar -xzf "$s"/${q(f)} -C "$s/x"`,
      'test -n "$(ls -A "$s/x")"',
      'for n in $(ls -A "$s/x"); do',
      '  case $n in .local-ai-*|*[!A-Za-z0-9._+-]*) echo "local-ai: unexpected archive entry $n" >&2; exit 1;; esac',
      `  nolink ${d}/"$n"`,
      "  ln=$(printf '%s' \"$n\" | tr '[:upper:]' '[:lower:]')",
      ...keep.map((x) => `  case "$ln" in ${q(x.slice(into.length + 1).split("/")[0].toLowerCase())}) echo "local-ai: archive entry $n would replace ${x}" >&2; exit 1;; esac`),
      "done",
      CONFINE,
      'confine "$s/x"',
      ...(rt.python ? [`test -f "$s"/x/${q(rt.python.slice(into.length + 1))}`, `"$s"/x/${q(rt.python.slice(into.length + 1))} -I -c pass`] : []),
      'entries=$(ls -A "$s/x")',
      // an entry gone from $s/x was placed (rename is atomic): remove it; then put every backed-up entry back
      `back() { ok=0; for n in $entries; do if [ ! -e "$s/x/$n" ] && [ ! -L "$s/x/$n" ]; then rm -rf ${d}/"$n" || ok=1; fi; done; ` +
      `for n in $(ls -A "$s/old"); do rm -rf ${d}/"$n" || ok=1; mv "$s/old/$n" ${d}/"$n" || ok=1; done; return $ok; }`,
      'undo() { echo "local-ai: could not publish the runtime; restoring the previous one" >&2; exit 1; }',
      "pub=1",
      'for n in $entries; do',
      `  if [ -e ${d}/"$n" ]; then mv ${d}/"$n" "$s/old/$n" || undo; fi`,
      `  mv "$s/x/$n" ${d}/"$n" || undo`,
      "done",
      "pub="]) });
  }
  if (rt.packages) {
    const pk = rt.packages, d = q(pk.dir), t = q(pk.target), fs = pk.wheels.map((w) => urlFile(w.url));
    out.push({ title: "Install the packages", code: strict([
      NOLINK,
      TXN,
      `nolink ${prefixes(pk.dir)} ${prefixes(pk.target)}`,
      `mkdir -p ${q(parent(pk.dir))} ${q(parent(pk.target))}`,
      `w=$(mktemp -d ${q(parent(pk.dir))}/.local-ai-wheels.XXXXXX)`,
      `o=$(mktemp -d ${q(parent(pk.dir))}/.local-ai-old.XXXXXX)`,
      `s=$(mktemp -d ${q(parent(pk.target))}/.local-ai-packages.XXXXXX)`,
      ...pk.wheels.map((w, i) => `curl -q -fL -o "$w"/${q(fs[i])} ${q(w.url)}`),
      `printf '%s\\n' ${pk.wheels.map((w, i) => q(`${w.sha256}  ${fs[i]}`)).join(" ")} > "$w/SHA256SUMS"`,
      '(cd "$w" && shasum -a 256 -c SHA256SUMS)',
      `printf '%s\\n' ${pk.wheels.map((w) => q(`${w.name}==${w.version} --hash=sha256:${w.sha256}`)).join(" ")} > "$w/requirements.lock.txt"`,
      PIPENV,
      `(pipenv "$s"; exec ${q(`./${rt.python}`)} -I -m pip --isolated install --no-index --find-links "$w" --only-binary :all: --require-hashes ` +
      '-r "$w/requirements.lock.txt" --target "$s/t" --no-compile --no-cache-dir --disable-pip-version-check)',
      `nolink ${t} ${d}`,
      // the new target and wheel dir are placed once "$s/t" and "$w" are gone; the old ones wait in $s/old and $o/old
      `back() { ok=0; if [ ! -e "$s/t" ]; then rm -rf ${t} || ok=1; fi; if [ -e "$s/old" ]; then mv "$s/old" ${t} || ok=1; fi; ` +
      `if [ ! -e "$w" ]; then rm -rf ${d} || ok=1; fi; if [ -e "$o/old" ]; then mv "$o/old" ${d} || ok=1; fi; return $ok; }`,
      'undo() { echo "local-ai: could not publish the packages; restoring the previous ones" >&2; exit 1; }',
      "pub=1",
      `if [ -e ${t} ]; then mv ${t} "$s/old" || undo; fi`,
      `mv "$s/t" ${t} || undo`,
      `if [ -e ${d} ]; then mv ${d} "$o/old" || undo; fi`,
      `mv "$w" ${d} || undo`,
      "pub="]) });
  }
  return out;
}

/** Write a file through an exclusive temp file in its (checked) directory `dir`, renamed over `path` only if that is
 *  neither a link nor a directory; the temp file is a fresh regular file that no one else could have created. `dir` and
 *  `path` are shell words. */
const write = (dir, path, label, body) => [`t=$(mktemp ${dir}/.local-ai-tmp.XXXXXX)`, 'test -f "$t"', 'test ! -L "$t"', ...body('"$t"'),
  'chmod 644 "$t"', `nolink ${path}`, `if [ -d ${path} ]; then echo "local-ai: ${label} is a directory" >&2; exit 1; fi`, `mv -f "$t" ${path}`];

/** Host weights: a directory (--local-dir), or a Hugging Face cache (layout hub). A hub download goes into a fresh
 *  staging cache beside `at`, with no ambient token or cache (token=False and HUBENV); its pinned snapshot must exist and
 *  every link in it must stay inside it; `ref` then names the revision; and only then does the staged model directory
 *  replace the one in `at` by rename (put back if that fails). Nothing in `at` is written through. With a pinned runtime
 *  the download runs on its own interpreter and huggingface_hub, never an `hf` from PATH. */
function hostDownload(w, rt) {
  if (w.layout !== "hub") return `hf download ${w.repo}${files(w)} \\\n  --revision ${w.revision} \\\n  --local-dir ${w.at}`;
  const at = w.at, rev = w.revision, name = `models--${w.repo.replace("/", "--")}`;
  const m = q(`${at}/${name}`), staged = `"$c"/${q(name)}`;
  const fs = Array.isArray(w.files) ? w.files : w.files ? [w.files] : [];
  const get = rt ? `${q(`./${rt.python}`)} -I -c ${q(HUB)} ${q(rt.packages.target)} ${w.repo} ${rev} "$c"${fs.map((f) => ` ${q(f)}`).join("")}`
    : `hf download ${w.repo}${files(w)} --revision ${rev} --cache-dir "$c"`;
  const lines = [`nolink ${prefixes(at)} ${m}`, `mkdir -p ${q(at)}`,
    `c=$(mktemp -d ${q(at)}/.local-ai-hub.XXXXXX)`, `o=$(mktemp -d ${q(at)}/.local-ai-old.XXXXXX)`,
    `(hubenv "$c"; exec ${get})`,
    `test -d ${staged}/snapshots/${rev}`, `confine ${staged}`];
  if (w.ref) lines.push(`mkdir -p ${staged}/refs`, `nolink ${staged}/refs`,
    ...write(`${staged}/refs`, `${staged}/refs/${q(w.ref)}`, `refs/${w.ref}`, (t) => [`printf '%s' ${rev} > ${t}`]));
  return [...lines, `nolink ${prefixes(at)} ${m}`, `m=${m}`, `ms=${staged}`, "pub=1",
    'if [ -e "$m" ]; then mv "$m" "$o/old" || hubundo "$m"; fi', 'mv "$ms" "$m" || hubundo "$m"',
    "pub=", 'rm -rf "$c" "$o"', "c= o="];
}

/** A program on the host. One with pinned packages or weights (MLX, oMLX or llama.cpp on Apple silicon) also downloads
 *  its weights, writes its launcher and raises the GPU wired-memory limit before it starts. One with a pinned `runtime`
 *  installs that runtime first (in place of the free-text `install`). */
function hostSteps(l) {
  checkHost(l);
  const env = Object.entries(l.env ?? {}).map(([k, v]) => `${k}=${value(v)}`);
  const rt = l.runtime;
  let out;
  if (rt) out = runtimeSteps(l);
  else {
    if (!l.pip && !l.weights?.length) return [{ title: "Install", code: install(l.install) }, { title: "Start the server", code: [...env, ...l.command.map(q)].join(" ") }];
    out = [l.pip ? { title: "Install the engine", code: `python3 -m pip install ${l.pip.map(q).join(" ")}` } : { title: "Install the engine", code: install(l.install) }];
    if (l.pip && /^https:\/\/\S+\.tar\.gz#sha256=[0-9a-f]{64}$/.test(l.install ?? "")) out.push({ title: "Install the native engine", code: install(l.install) });
  }
  if (l.weights?.length) {
    if (l.weights.some((w) => w.layout === "hub")) out.push({ title: "Download the weights", code: strict([NOLINK, TXN, HUBBACK, HUBUNDO, CONFINE, HUBENV, ...l.weights.flatMap((w) => hostDownload(w, rt))]) });
    else out.push({ title: "Download the weights", code: l.weights.map((w) => hostDownload(w)).join("\n\n") });
  }
  if (l.config) {
    const at = l.config.at;
    if (confined(l)) out.push({ title: "Write the launcher", code: strict([NOLINK, TXN, `nolink ${prefixes(at)}`, `mkdir -p ${q(parent(at))}`,
      ...write(q(parent(at)), q(at), at, (t) => [`cat > ${t} <<'EOF'\n${l.config.text.trimEnd()}\nEOF`])]) });
    else out.push({ title: "Write the launcher", code: `${at.includes("/") ? `mkdir -p ${at.replace(/\/[^/]*$/, "")}\n` : ""}cat > ${at} <<'EOF'\n${l.config.text.trimEnd()}\nEOF` });
  }
  if (l.sysctl) out.push({ title: "Raise the GPU memory limit (resets on reboot)", code: Object.entries(l.sysctl).map(([k, v]) => `sudo sysctl ${k}=${v}`).join("\n") });
  const run = [...env, l.command[0], ...pairs(l.command.slice(1))];
  out.push({ title: "Start the server", code: run.filter(Boolean).join(" \\\n  ") });
  return out;
}

/** The steps to run a recipe by hand: download the weights, write the config, start the container. */
export function steps(r) {
  const l = r.launch, name = (r.key ?? r.model).split("/").pop();
  const src = l.source ? `\n# Full instructions: ${l.source}` : "";
  const out = l.setup ? [{ title: "Set up first", code: `# ${l.setup}${src}` }] : [];
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
/** All the steps as one shell script. One with a pinned runtime or a host Hugging Face cache stops at the first step
 *  that fails, however it is run. */
export const command = (r) => {
  const l = r.launch, strictRun = l.kind === "host" && (l.runtime || weights(l).some((w) => w.layout === "hub"));
  return (strictRun ? "set -e\n\n" : "") + steps(r).map((s) => `# ${s.title}\n${s.code}`).join("\n\n");
};
