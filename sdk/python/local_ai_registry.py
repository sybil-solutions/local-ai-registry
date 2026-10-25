"""Local AI registry, Python. Standard library only.

    from local_ai_registry import pick, command
    print(command(pick("NVIDIA GeForce RTX 4090", vram=24)))

    python3 local_ai_registry.py "RTX 4090" 24      # prints the command for your GPU
"""
import json
import re
import sys
import urllib.parse
import urllib.request

URL = "https://local.sybilsolutions.ai/api/v2/catalog.json"
_cache = {}


def catalog(url=URL):
    """The whole catalog: cards (with their picks), recipes (with rendered launches), models."""
    if url not in _cache:
        with urllib.request.urlopen(url, timeout=30) as r:
            _cache[url] = json.load(r)
    return _cache[url]


def _norm(s):
    s = re.sub(r"\((r|tm)\)", "", s.lower())
    return re.sub(r"geforce|nvidia|amd|radeon|intel|graphics|[^a-z0-9]", "", s)


def _names(i, c):
    return [_norm(c["name"]), _norm(i)] + list((c.get("match") or {}).get("names") or [])


def detect(cat, gpu, vram=None):
    """The card id for a GPU as the driver names it, and its memory in GB."""
    n = _norm(gpu)
    hits = [(i, c) for i, c in cat["cards"].items()
            if any(x and x in n for x in _names(i, c)) and (not vram or abs(c["vram_gb"] - vram) <= 2)]
    hits.sort(key=lambda x: (-max(len(y) for y in _names(*x) if y and y in n),
                             abs(x[1]["vram_gb"] - vram) if vram else 0))
    return hits[0][0] if hits else None


def pick(gpu, vram=None, all=False, cat=None):
    """The recommended recipe for a GPU (or None); all=True returns the card's top three."""
    cat = cat or catalog()
    i = detect(cat, gpu, vram)
    if not i:
        return [] if all else None
    rs = [{"key": k, **cat["recipes"][k]} for k in cat["cards"][i]["picks"]]
    return rs if all else rs[0]


GPU = {"cpu": "", "nvidia": "--gpus all", "amd-rocm": "--device /dev/kfd --device /dev/dri", "amd-vulkan": "--device /dev/dri", "intel-xpu": "--device /dev/dri"}


def _q(s):
    return s if re.fullmatch(r"[\w@%+=:,./-]+", s) else "'" + s.replace("'", "'\\''") + "'"  # the same quoting as the JS SDK


def _value(s):
    return '"' + s + '"' if re.fullmatch(r"\$\{[A-Z_][A-Z0-9_]*\}", s) else _q(s)


def _files(w):
    fs = w.get("files") or []
    return "".join(" " + _q(f) for f in (fs if isinstance(fs, list) else [fs]))


def _resources(l):
    r = l.get('resources') or {}
    return ([f"--memory {r['memoryBytes']}"] if r.get('memoryBytes') else []) + ([f"--memory-swap {r['memorySwapBytes']}"] if r.get('memorySwapBytes') else []) + \
        (['--ulimit memlock=-1:-1'] if r.get('memlockUnlimited') else []) + (['--cap-add IPC_LOCK'] if r.get('ipcLock') else [])


# A pinned host runtime and Hugging Face cache weights: the same rules as lab/lab.py check_host_runtime, checked again here so
# that nothing malformed is ever rendered into a shell command (fail closed).
_RT_SHA = r"[0-9a-f]{64}"
_RT_REL = r"(?!.*(?:^|/)\.\.?(?:/|$))[A-Za-z0-9_][A-Za-z0-9._+-]*(?:/[A-Za-z0-9_][A-Za-z0-9._+-]*)*"
_RT_URL = r"https://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~%+-]+)+"
_RT_ARCHIVE = r"[A-Za-z0-9][A-Za-z0-9._+-]*\.(?:tar\.gz|tgz)"
_RT_WHEEL = r"[A-Za-z0-9][A-Za-z0-9._+-]*\.whl"
_RT_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_RT_VERSION = r"[A-Za-z0-9][A-Za-z0-9.+!_-]*"
_RT_REF = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RT_REPO = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"


def _ok(cond, msg):
    if not cond:
        raise ValueError(f"invalid host runtime: {msg}")


def _is(pattern, v):
    return isinstance(v, str) and re.fullmatch(pattern, v) is not None


def _canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _url_file(url):
    return urllib.parse.unquote(url.rsplit("/", 1)[1])


def _check_host(l):
    """Refuse to render a host runtime or hub weights that lab.py check would reject."""
    ws, hubs = _weights(l), set()
    _ok(len({w.get("layout") == "hub" for w in ws}) <= 1, "hub and directory weights mixed in one launch")
    for w in ws:
        if "ref" in w:
            _ok(w.get("layout") == "hub" and _is(_RT_REF, w["ref"]), f"weights.ref {w.get('ref')!r}")
        if w.get("layout") == "hub":
            key = (str(w.get("at")).lower(), str(w.get("repo")).lower())
            _ok(key not in hubs, f"weights {w.get('repo')} listed twice in the cache {w.get('at')}")
            hubs.add(key)
            _ok(_is(_RT_REPO, w.get("repo")) and _is(r"[0-9a-f]{40}", w.get("revision")) and _is(_RT_REL, w.get("at")), "hub weights repo, revision or cache dir")
            fs = w.get("files") or []
            _ok(all(_is(_RT_REL, f) for f in (fs if isinstance(fs, list) else [fs])), "hub weights files")
    if l.get("config") and _confined(l):  # a launcher the setup steps write: a plain relative path
        _ok(_is(_RT_REL, l["config"].get("at")), "launcher path")
    rt = l.get("runtime")
    if rt is None:
        return _check_paths(l)
    _ok(isinstance(rt, dict) and rt and set(rt) <= {"archive", "python", "packages"} and not l.get("pip"), "runtime keys")
    if "archive" in rt:
        a = rt["archive"]
        _ok(isinstance(a, dict) and set(a) == {"url", "sha256", "into"}, "runtime.archive keys")
        _ok(_is(_RT_URL, a["url"]) and _is(_RT_ARCHIVE, _url_file(a["url"])), f"runtime.archive.url {a.get('url')!r}")
        _ok(_is(_RT_SHA, a["sha256"]) and _is(_RT_REL, a["into"]), "runtime.archive sha256 or into")
    if "python" in rt:
        _ok(_is(_RT_REL, rt["python"]) and "archive" in rt and rt["python"].startswith(rt["archive"]["into"] + "/"),
            f"runtime.python {rt.get('python')!r} is not inside runtime.archive.into")
    if "packages" in rt:
        pk = rt["packages"]
        _ok("python" in rt and isinstance(pk, dict) and set(pk) == {"dir", "target", "wheels"}, "runtime.packages keys")
        _ok(_is(_RT_REL, pk["dir"]) and _is(_RT_REL, pk["target"]), "runtime.packages dir or target")
        _ok(not _nested(pk["dir"], pk["target"]), "runtime.packages dir and target overlap")
        _ok(isinstance(pk["wheels"], list) and pk["wheels"], "runtime.packages.wheels")
        seen = set()
        for w in pk["wheels"]:
            _ok(isinstance(w, dict) and set(w) == {"name", "version", "url", "sha256"}, "wheel keys")
            _ok(_is(_RT_NAME, w["name"]) and _is(_RT_VERSION, w["version"]) and _is(_RT_URL, w["url"]) and _is(_RT_SHA, w["sha256"]), f"wheel {w.get('name')!r}")
            f = _url_file(w["url"])
            dist, ver = (f.split("-") + ["", ""])[:2]
            _ok(_is(_RT_WHEEL, f) and _canon(dist) == _canon(w["name"]) and ver == w["version"], f"wheel file {f!r}")
            _ok(_canon(w["name"]) not in seen, f"wheel {w['name']} listed twice")
            seen.add(_canon(w["name"]))
    if ws:
        _ok("packages" in rt and "huggingface-hub" in {_canon(w["name"]) for w in rt["packages"]["wheels"]} and ws[0].get("layout") == "hub",
            "weights with a runtime need hub layout and a pinned huggingface_hub wheel")
    _check_paths(l)


def _confined(l):
    """A host launch whose setup steps are confined (strict, staged, no links): one with a runtime or hub weights."""
    return l.get("runtime") is not None or any(w.get("layout") == "hub" for w in _weights(l))


def _owned(l):
    """What the setup steps publish besides the archive's entries: packages target and dir, host hub caches, launcher."""
    pk = (l.get("runtime") or {}).get("packages") or {}
    return ([("packages.target", pk["target"]), ("packages.dir", pk["dir"])] if pk else []) + \
        [("weights.at", w["at"]) for w in _weights(l) if w.get("layout") == "hub"] + \
        ([("launcher", l["config"]["at"])] if l.get("config") and (l.get("runtime") or any(w.get("layout") == "hub" for w in _weights(l))) else [])


def _check_paths(l):
    """No step may publish over another's output: no two destinations equal or nest (hub caches may be shared), none is or
    holds runtime.archive.into, and none lies in the archive entry that holds the interpreter."""
    owned = _owned(l)
    for i, (na, a) in enumerate(owned):
        for nb, b in owned[i + 1:]:
            _ok((na == nb == "weights.at" and a.lower() == b.lower()) or not _nested(a, b), f"{na} {a} and {nb} {b} overlap")
    a = (l.get("runtime") or {}).get("archive")
    if a:
        py = l["runtime"].get("python")
        entry = a["into"] + "/" + py[len(a["into"]) + 1:].split("/")[0] if py else None
        for n, x in owned:
            _ok(not (x.lower() == a["into"].lower() or a["into"].lower().startswith(x.lower() + "/")), f"{n} {x} would replace runtime.archive.into")
            _ok(not entry or not _nested(x, entry), f"{n} {x} overlaps the interpreter's archive entry {entry}")


def _nested(a, b):
    """Equal or one inside the other, ignoring case (macOS file systems do)."""
    a, b = a.lower(), b.lower()
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _weights(l):
    ws = l.get("weights") or []
    return [w for w in (ws if isinstance(ws, list) else [ws]) if w and w.get("repo")]


_NOLINK = ('nolink() { for p; do if [ -L "$p" ]; then echo "local-ai: $p is a symlink; refusing to write through it" >&2; exit 1; fi; done; }')
_HUB = ("import sys; sys.path.insert(0, sys.argv[1]); from huggingface_hub import snapshot_download; "
        "snapshot_download(sys.argv[2], revision=sys.argv[3], cache_dir=sys.argv[4], allow_patterns=sys.argv[5:] or None, token=False)")
# Every link under a staged tree resolves inside it (relative, not dangling), every hard link stays inside it, and it holds
# no device, fifo, socket or set-id file. The same text in both SDKs.
_CONFINE = "confine() {\n  b=$(cd -P \"$1\" && pwd -P)\n  if [ -n \"$(find \"$1\" ! -type f ! -type d ! -type l)\" ]; then echo \"local-ai: $1 holds a device, fifo or socket\" >&2; exit 1; fi\n  if [ -n \"$(find \"$1\" -type f \\( -perm -4000 -o -perm -2000 \\))\" ]; then echo \"local-ai: $1 holds a setuid or setgid file\" >&2; exit 1; fi\n  find \"$1\" -type l | while IFS= read -r l; do\n    if [ ! -L \"$l\" ]; then echo \"local-ai: unexpected name under $1\" >&2; exit 1; fi\n    k=$(readlink \"$l\")\n    case $k in /*) echo \"local-ai: link $l is absolute\" >&2; exit 1;; esac\n    if [ ! -e \"$l\" ]; then echo \"local-ai: link $l is dangling\" >&2; exit 1; fi\n    if [ -d \"$l\" ]; then r=$(cd -P \"$l\" && pwd -P); else\n      case ${k##*/} in .|..) echo \"local-ai: link $l leaves $1\" >&2; exit 1;; esac\n      r=$(cd -P \"$(dirname \"$l\")\" && cd -P \"$(dirname \"$k\")\" && pwd -P)/${k##*/}\n    fi\n    case $r/ in \"$b\"/*) ;; *) echo \"local-ai: link $l leaves $1\" >&2; exit 1;; esac\n  done\n  find \"$1\" -type f -links +1 | while IFS= read -r f; do\n    if [ \"$(ls -ld \"$f\" | awk '{print $2}')\" != \"$(find \"$1\" -samefile \"$f\" | wc -l | tr -d ' ')\" ]; then echo \"local-ai: $f is hard-linked outside $1\" >&2; exit 1; fi\n  done\n}"
# The download sees no ambient Hugging Face token, config, cache or .netrc: only directories inside its staging cache.
_HUBENV = "hubenv() { for v in $(env | sed -n 's/^\\(HF_[A-Za-z0-9_]*\\)=.*/\\1/p; s/^\\(HUGGING_FACE_[A-Za-z0-9_]*\\)=.*/\\1/p'); do unset \"$v\"; done; unset NETRC; h=$(cd -P \"$1\" && pwd -P); HOME=$h/.home; HF_HOME=$h/.hf-home; HF_XET_CACHE=$h/.xet; TMPDIR=$h/.tmp; TMP=$h/.tmp; TEMP=$h/.tmp; HF_HUB_DISABLE_TELEMETRY=1; export HOME HF_HOME HF_XET_CACHE TMPDIR TMP TEMP HF_HUB_DISABLE_TELEMETRY; mkdir -p \"$HOME\" \"$TMPDIR\"; }"
# pip sees no config file (PIP_CONFIG_FILE=/dev/null turns off every one, which --isolated does not), and its temp files,
# home and cache are inside the packages staging directory.
_PIPENV = "pipenv() { h=$(cd -P \"$1\" && pwd -P)/.pip; mkdir -p \"$h/tmp\" \"$h/home\" \"$h/cache\"; PIP_CONFIG_FILE=/dev/null; TMPDIR=$h/tmp; TMP=$h/tmp; TEMP=$h/tmp; HOME=$h/home; XDG_CACHE_HOME=$h/cache; XDG_CONFIG_HOME=$h/home/.config; export PIP_CONFIG_FILE TMPDIR TMP TEMP HOME XDG_CACHE_HOME XDG_CONFIG_HOME; }"


# Every step's exit, by success, failure or a signal (TERM, INT, HUP), goes through fin: while a publication is unfinished
# (pub set) the step's back() puts the previous installation back first, and if that fails the backup is kept (named on
# stderr) and the step fails; only then are the staging directories removed. fin ignores TERM, INT and HUP first (the
# rollback's own children inherit that), so a repeated signal cannot cut the rollback short. The same text in both SDKs.
_TXN = "pub= c= o= s= t= w=\nfin() { rc=${1:-$?}; trap '' TERM INT HUP; trap - EXIT; if [ -n \"$pub\" ] && ! back; then for k in ${s:+\"$s/old\"} ${o:+\"$o/old\"}; do if [ -e \"$k\" ]; then echo \"local-ai: could not restore it; the previous copy is kept in $k\" >&2; fi; done; exit 1; fi; for x in \"$c\" \"$o\" \"$s\" \"$t\" \"$w\"; do if [ -n \"$x\" ]; then rm -rf \"$x\"; fi; done; exit \"$rc\"; }\ntrap fin EXIT; trap 'fin 143' TERM; trap 'fin 130' INT; trap 'fin 129' HUP"
_HUBUNDO = 'hubundo() { echo "local-ai: could not publish $1; restoring the previous one" >&2; exit 1; }'
# the hub model directory being published: m (live) and ms (staged); the new one is placed once ms is gone
_HUBBACK = "back() { ok=0; if [ ! -e \"$ms\" ]; then rm -rf \"$m\" || ok=1; fi; if [ -e \"$o/old\" ]; then mv \"$o/old\" \"$m\" || ok=1; fi; return $ok; }"


def _prefixes(path):
    """`a/b/c` -> `a a/b a/b/c`: every directory a write to `path` passes through, quoted."""
    parts = path.split("/")
    return " ".join(_q("/".join(parts[:i + 1])) for i in range(len(parts)))


def _parent(path):
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _strict(lines):
    """One step as one subshell that stops at its first failure and returns non-zero (pasted alone or in a script)."""
    return "(\nset -e\n" + "\n".join(lines) + "\n)"


def _runtime_steps(l):
    """The pinned interpreter archive, then the hash-locked offline wheel install. Each is staged in a fresh directory
    beside its destination and checked in full (hashes, entry names, links, the interpreter) before anything live is
    touched; it is then published by rename, and a failed rename puts back everything it had moved."""
    rt, out = l["runtime"], []
    if "archive" in rt:
        a, into = rt["archive"], rt["archive"]["into"]
        f, d = _url_file(a["url"]), _q(into)
        keep = [x for _, x in _owned(l) if x.lower().startswith(into.lower() + "/")]  # ignoring case, as macOS does
        out.append({"title": "Install the runtime", "code": _strict([
            _NOLINK,
            _TXN,
            f"nolink {_prefixes(into)}",
            f"mkdir -p {d}",
            f"s=$(mktemp -d {d}/.local-ai-runtime.XXXXXX)",
            f'curl -q -fL -o "$s"/{_q(f)} {_q(a["url"])}',
            f"(cd \"$s\" && echo '{a['sha256']}  {f}' | shasum -a 256 -c -)",
            'mkdir "$s/x" "$s/old"',
            f'tar -tzf "$s"/{_q(f)} | while IFS= read -r m; do case $m in /*|..|../*|*/..|*/../*) echo "local-ai: archive member $m leaves the archive" >&2; exit 1;; esac; done',
            f'tar -xzf "$s"/{_q(f)} -C "$s/x"',
            'test -n "$(ls -A "$s/x")"',
            'for n in $(ls -A "$s/x"); do',
            '  case $n in .local-ai-*|*[!A-Za-z0-9._+-]*) echo "local-ai: unexpected archive entry $n" >&2; exit 1;; esac',
            f'  nolink {d}/"$n"',
            '  ln=$(printf \'%s\' "$n" | tr \'[:upper:]\' \'[:lower:]\')',
            *[f'  case "$ln" in {_q(x[len(into) + 1:].split("/")[0].lower())}) echo "local-ai: archive entry $n would replace {x}" >&2; exit 1;; esac' for x in keep],
            "done",
            _CONFINE,
            'confine "$s/x"',
            *([f'test -f "$s"/x/{_q(rt["python"][len(into) + 1:])}', f'"$s"/x/{_q(rt["python"][len(into) + 1:])} -I -c pass'] if rt.get("python") else []),
            'entries=$(ls -A "$s/x")',
            # an entry gone from $s/x was placed (rename is atomic): remove it; then put every backed-up entry back
            f'back() {{ ok=0; for n in $entries; do if [ ! -e "$s/x/$n" ] && [ ! -L "$s/x/$n" ]; then rm -rf {d}/"$n" || ok=1; fi; done; '
            f'for n in $(ls -A "$s/old"); do rm -rf {d}/"$n" || ok=1; mv "$s/old/$n" {d}/"$n" || ok=1; done; return $ok; }}',
            'undo() { echo "local-ai: could not publish the runtime; restoring the previous one" >&2; exit 1; }',
            "pub=1",
            'for n in $entries; do',
            f'  if [ -e {d}/"$n" ]; then mv {d}/"$n" "$s/old/$n" || undo; fi',
            f'  mv "$s/x/$n" {d}/"$n" || undo',
            "done",
            "pub="])})
    if "packages" in rt:
        pk = rt["packages"]
        d, t = _q(pk["dir"]), _q(pk["target"])
        files = [_url_file(w["url"]) for w in pk["wheels"]]
        out.append({"title": "Install the packages", "code": _strict([
            _NOLINK,
            _TXN,
            f"nolink {_prefixes(pk['dir'])} {_prefixes(pk['target'])}",
            f"mkdir -p {_q(_parent(pk['dir']))} {_q(_parent(pk['target']))}",
            f"w=$(mktemp -d {_q(_parent(pk['dir']))}/.local-ai-wheels.XXXXXX)",
            f"o=$(mktemp -d {_q(_parent(pk['dir']))}/.local-ai-old.XXXXXX)",
            f"s=$(mktemp -d {_q(_parent(pk['target']))}/.local-ai-packages.XXXXXX)",
            *[f'curl -q -fL -o "$w"/{_q(f)} {_q(w["url"])}' for w, f in zip(pk["wheels"], files)],
            "printf '%s\\n' " + " ".join(_q(f"{w['sha256']}  {f}") for w, f in zip(pk["wheels"], files)) + ' > "$w/SHA256SUMS"',
            '(cd "$w" && shasum -a 256 -c SHA256SUMS)',
            "printf '%s\\n' " + " ".join(_q(f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}") for w in pk["wheels"]) + ' > "$w/requirements.lock.txt"',
            _PIPENV,
            f"(pipenv \"$s\"; exec {_q('./' + rt['python'])} -I -m pip --isolated install --no-index --find-links \"$w\" --only-binary :all: --require-hashes "
            '-r "$w/requirements.lock.txt" --target "$s/t" --no-compile --no-cache-dir --disable-pip-version-check)',
            f"nolink {t} {d}",
            # the new target and wheel dir are placed once "$s/t" and "$w" are gone; the old ones wait in $s/old and $o/old
            f'back() {{ ok=0; if [ ! -e "$s/t" ]; then rm -rf {t} || ok=1; fi; if [ -e "$s/old" ]; then mv "$s/old" {t} || ok=1; fi; '
            f'if [ ! -e "$w" ]; then rm -rf {d} || ok=1; fi; if [ -e "$o/old" ]; then mv "$o/old" {d} || ok=1; fi; return $ok; }}',
            'undo() { echo "local-ai: could not publish the packages; restoring the previous ones" >&2; exit 1; }',
            "pub=1",
            f'if [ -e {t} ]; then mv {t} "$s/old" || undo; fi',
            f'mv "$s/t" {t} || undo',
            f'if [ -e {d} ]; then mv {d} "$o/old" || undo; fi',
            f'mv "$w" {d} || undo',
            "pub="])})
    return out


def _write(dir_, path, label, body):
    """Write a file through an exclusive temp file in its (checked) directory `dir_`, renamed over `path` only if that is
    neither a link nor a directory; the temp file is a fresh regular file that no one else could have created. `dir_` and
    `path` are shell words."""
    return [f"t=$(mktemp {dir_}/.local-ai-tmp.XXXXXX)",
            'test -f "$t"',
            'test ! -L "$t"',
            *body('"$t"'),
            'chmod 644 "$t"',
            f"nolink {path}",
            f"if [ -d {path} ]; then echo \"local-ai: {label} is a directory\" >&2; exit 1; fi",
            f'mv -f "$t" {path}']


def _host_download(w, rt=None):
    """Host weights: a directory (--local-dir), or a Hugging Face cache (layout hub). A hub download goes into a fresh
    staging cache beside `at`, with no ambient token or cache (token=False and _HUBENV); its pinned snapshot must exist and
    every link in it must stay inside it; `ref` then names the revision; and only then does the staged model directory
    replace the one in `at` by rename (put back if that fails). Nothing in `at` is written through. With a pinned runtime
    the download runs on its own interpreter and huggingface_hub, never an `hf` from PATH."""
    if w.get("layout") != "hub":
        return f"hf download {w['repo']}{_files(w)} \\\n  --revision {w['revision']} \\\n  --local-dir {w['at']}"
    at, rev = w["at"], w["revision"]
    name = f"models--{w['repo'].replace('/', '--')}"
    m, staged = _q(f"{at}/{name}"), f'"$c"/{_q(name)}'
    fs = w.get("files") or []
    fs = fs if isinstance(fs, list) else [fs]
    if rt:
        get = f"{_q('./' + rt['python'])} -I -c {_q(_HUB)} {_q(rt['packages']['target'])} {w['repo']} {rev} \"$c\"" + "".join(f" {_q(f)}" for f in fs)
    else:
        get = f"hf download {w['repo']}{_files(w)} --revision {rev} --cache-dir \"$c\""
    lines = [f"nolink {_prefixes(at)} {m}", f"mkdir -p {_q(at)}",
             f"c=$(mktemp -d {_q(at)}/.local-ai-hub.XXXXXX)", f"o=$(mktemp -d {_q(at)}/.local-ai-old.XXXXXX)",
             f'(hubenv "$c"; exec {get})',
             f"test -d {staged}/snapshots/{rev}", f"confine {staged}"]
    if w.get("ref"):
        lines += [f"mkdir -p {staged}/refs", f"nolink {staged}/refs",
                  *_write(f"{staged}/refs", f"{staged}/refs/{_q(w['ref'])}", f"refs/{w['ref']}", lambda t: [f"printf '%s' {rev} > {t}"])]
    return lines + [f"nolink {_prefixes(at)} {m}",
                    f"m={m}", f"ms={staged}", "pub=1",
                    'if [ -e "$m" ]; then mv "$m" "$o/old" || hubundo "$m"; fi',
                    'mv "$ms" "$m" || hubundo "$m"',
                    "pub=", 'rm -rf "$c" "$o"', "c= o="]


def _host_steps(l):
    """A program on the host. One with pinned packages or weights (MLX, oMLX or llama.cpp on Apple silicon) also downloads
    its weights, writes its launcher and raises the GPU wired-memory limit before it starts. One with a pinned `runtime`
    installs that runtime first (in place of the free-text `install`)."""
    _check_host(l)
    env = [f"{k}={_value(v)}" for k, v in (l.get("env") or {}).items()]
    rt = l.get("runtime")
    if rt:
        out = _runtime_steps(l)
    else:
        if not l.get("pip") and not l.get("weights"):
            return [{"title": "Install", "code": _install(l["install"])}, {"title": "Start the server", "code": " ".join(env + [_q(c) for c in l["command"]])}]
        out = [{"title": "Install the engine", "code": "python3 -m pip install " + " ".join(_q(x) for x in l["pip"]) if l.get("pip") else _install(l["install"])}]
        if l.get("pip") and re.fullmatch(r"https://\S+\.tar\.gz#sha256=[0-9a-f]{64}", l.get("install") or ""):
            out.append({"title": "Install the native engine", "code": _install(l["install"])})
    if l.get("weights"):
        if any(w.get("layout") == "hub" for w in l["weights"]):  # a cache: checked, confined and stopped at its first failure
            out.append({"title": "Download the weights", "code": _strict([_NOLINK, _TXN, _HUBBACK, _HUBUNDO, _CONFINE, _HUBENV,
                                                                         *[x for w in l["weights"] for x in _host_download(w, rt)]])})
        else:
            out.append({"title": "Download the weights", "code": "\n\n".join(_host_download(w) for w in l["weights"])})
    if l.get("config"):
        at = l["config"]["at"]
        if _confined(l):
            out.append({"title": "Write the launcher", "code": _strict([_NOLINK, _TXN, f"nolink {_prefixes(at)}", f"mkdir -p {_q(_parent(at))}",
                                                                         *_write(_q(_parent(at)), _q(at), at, lambda t: [f"cat > {t} <<'EOF'\n{l['config']['text'].rstrip()}\nEOF"])])})
        else:
            mk = f"mkdir -p {at.rsplit('/', 1)[0]}\n" if "/" in at else ""
            out.append({"title": "Write the launcher", "code": f"{mk}cat > {at} <<'EOF'\n{l['config']['text'].rstrip()}\nEOF"})
    if l.get("sysctl"):
        out.append({"title": "Raise the GPU memory limit (resets on reboot)",
                    "code": "\n".join(f"sudo sysctl {k}={v}" for k, v in l["sysctl"].items())})
    run = env + [l["command"][0]] + _pairs(l["command"][1:])
    out.append({"title": "Start the server", "code": " \\\n  ".join(run)})
    return out


def _pairs(a):
    """Arguments as `--flag value` pairs, one per line."""
    args, i = [], 0
    while i < len(a):
        if a[i].startswith("-") and i + 1 < len(a) and not a[i + 1].startswith("-"):
            args.append(f"{a[i]} {_q(a[i + 1])}"); i += 2
        else:
            args.append(_q(a[i])); i += 1
    return args


def steps(r):
    """The steps to run a recipe by hand: download the weights, write the config, start the container."""
    l = r["launch"]
    name = (r.get("key") or r["model"]).split("/")[-1]
    src = f"\n# Full instructions: {l['source']}" if l.get("source") else ""
    out = [{"title": "Set up first", "code": f"# {l['setup']}{src}"}] if l.get("setup") else []
    if l.get("kind") == "host":
        return out + _host_steps(l)
    image = l.get("image")
    if l.get("build"):  # an image built from the source's Dockerfile at a pinned commit
        image = f"local-ai/{name}"
        files = (l["build"].get("dockerfile") or "Dockerfile").split(" + ")
        out.append({"title": "Build the image", "code": "\n".join([_install(f"https://github.com/{l['build']['repo']}/tree/{l['build']['commit']}")] + [f"docker build -t {image} -f {d} ." for d in files])})
    ws = [w for w in (l["weights"] if isinstance(l["weights"], list) else [l["weights"]]) if w and w.get("repo")]
    prepared = f"~/models/{name}-prepared-{image[-12:]}"
    mounts, dl = [], []
    for w in ws:
        d = f"~/models/{w['repo'].split('/')[1]}-{w['revision'][:8]}"
        if w.get("layout") == "hub":
            dl.append(f"hf download {w['repo']}{_files(w)} \\\n  --revision {w['revision']}")
            if "-v ~/.cache/huggingface:/root/.cache/huggingface" not in mounts:  # several hub weights share one cache mount
                mounts.append("-v ~/.cache/huggingface:/root/.cache/huggingface")
        else:
            dl.append(f"hf download {w['repo']}{_files(w)} \\\n  --revision {w['revision']} \\\n  --local-dir {d}")
            mounts.append(f"-v {d}:{w['at']}:ro")
    if dl:
        out.append({"title": "Download the weights", "code": "\n\n".join(dl)})
    if l.get("config"):
        out.append({"title": "Write the server config", "code": f"cat > {name}.yml <<'EOF'\n{l['config']['text'].rstrip()}\nEOF"})
        mounts.append(f"-v $PWD/{name}.yml:{l['config']['at']}:ro")
    e = l.get("entrypoint")
    ep = e if isinstance(e, list) else [e] if e else []
    args, a = [], ep[1:] + l["args"]
    i = 0
    while i < len(a):
        if a[i].startswith("-") and i + 1 < len(a) and not a[i + 1].startswith("-"):
            args.append(f"{a[i]} {_value(a[i + 1])}"); i += 2
        else:
            args.append(_value(a[i])); i += 1
    if l.get("prepare"):
        options = list(l.get('flags') or []) + _resources(l) + ([f"--shm-size {l['shm']}"] if l.get('shm') else [])
        options += [f"-e {k}={_value(v)}" for k, v in (l.get('env') or {}).items() if k != 'NVIDIA_VISIBLE_DEVICES']
        options += [f"--entrypoint {_q(ep[0])}"] if ep else []
        prep = ["docker run --rm", "" if l['prepare'].get('gpu') is False else GPU.get(l.get("backend") or "nvidia", GPU["nvidia"]), '--user "$(id -u):$(id -g)"', "-e HOME=/tmp"]
        prep += options + [f"-v {prepared}:{l['prepare']['at']}"] + mounts + [image] + [_q(x) for x in ep[1:] + l["prepare"]["args"]]
        out.append({"title": "Prepare the model (first start)", "code": f"mkdir -p {prepared}\n" + " \\\n  ".join(x for x in prep if x)})
        mounts.insert(0, f"-v {prepared}:{l['prepare']['at']}:ro")
        if l['prepare'].get('verifyArgs'):
            out.append({'title': 'Verify the prepared model', 'code': ' '.join(['docker run --rm', '--user "$(id -u):$(id -g)"', '-e HOME=/tmp'] + options + mounts + [image] + [_q(x) for x in ep[1:] + l['prepare']['verifyArgs']])})
    run = ["docker run --rm", GPU.get(l.get("backend") or "nvidia", GPU["nvidia"]), f"-p 8000:{l['port']}"] + list(l.get("flags") or []) + _resources(l)
    run += [f"--shm-size {l['shm']}"] if l.get("shm") else []
    run += [f"-e {k}={_value(v)}" for k, v in (l.get("env") or {}).items()] + mounts
    run += [f"--entrypoint {_q(ep[0])}"] if ep else []
    m = l.get("machines")
    title = f"Start the server on each of the {m} machines (NODE_RANK 0 to {m - 1})" if m else "Start the server"
    out.append({"title": title, "code": " \\\n  ".join(x for x in run + [image] + args if x)})
    return out


def _install(url):
    """Getting a source: a pinned GitHub tree becomes a clone at that commit."""
    m = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+)/tree/([0-9a-f]{40})", url)
    if m:
        return f"git clone https://github.com/{m[1]}/{m[2]} && cd {m[2]} && git checkout {m[3]}"
    t = re.fullmatch(r"(https://\S+/([\w.-]+\.tar\.gz))#sha256=([0-9a-f]{64})", url)  # a release archive, checked
    return f"curl -L -o {t[2]} {t[1]}\necho '{t[3]}  {t[2]}' | shasum -a 256 -c\ntar xzf {t[2]}" if t else f"# {url}"


def command(r):
    """All the steps as one shell script. One with a pinned runtime or a host Hugging Face cache stops at the first step
    that fails, however it is run."""
    l = r["launch"]
    strict = l.get("kind") == "host" and (l.get("runtime") or any(w.get("layout") == "hub" for w in _weights(l)))
    return ("set -e\n\n" if strict else "") + "\n\n".join(f"# {s['title']}\n{s['code']}" for s in steps(r))


if __name__ == "__main__":
    r = pick(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(command(r) if r else f"no tested recipe for {sys.argv[1]!r} yet: https://local.sybilsolutions.ai")
