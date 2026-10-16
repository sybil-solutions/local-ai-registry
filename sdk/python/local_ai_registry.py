"""Local AI registry, Python. Standard library only.

    from local_ai_registry import pick, command
    print(command(pick("NVIDIA GeForce RTX 4090", vram=24)))

    python3 local_ai_registry.py "RTX 4090" 24      # prints the command for your GPU
"""
import json
import re
import sys
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


def _host_steps(l):
    """A program on the host. One with pinned packages or weights (MLX, oMLX or llama.cpp on Apple silicon) also downloads
    its weights, writes its launcher and raises the GPU wired-memory limit before it starts."""
    env = [f"{k}={_value(v)}" for k, v in (l.get("env") or {}).items()]
    if not l.get("pip") and not l.get("weights"):
        return [{"title": "Install", "code": _install(l["install"])}, {"title": "Start the server", "code": " ".join(env + [_q(c) for c in l["command"]])}]
    out = [{"title": "Install the engine", "code": "python3 -m pip install " + " ".join(_q(x) for x in l["pip"]) if l.get("pip") else _install(l["install"])}]
    if l.get("pip") and re.fullmatch(r"https://\S+\.tar\.gz#sha256=[0-9a-f]{64}", l.get("install") or ""):
        out.append({"title": "Install the native engine", "code": _install(l["install"])})
    if l.get("weights"):
        out.append({"title": "Download the weights", "code": "\n\n".join(
            f"hf download {w['repo']}{_files(w)} \\\n  --revision {w['revision']} \\\n  --local-dir {w['at']}"
            for w in l["weights"])})
    if l.get("config"):
        at = l["config"]["at"]
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
    out = [{"title": "Set up first", "code": f"# {l['setup']}\n# Full instructions: {l.get('source')}"}] if l.get("setup") else []
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
    """All the steps as one shell script."""
    return "\n\n".join(f"# {s['title']}\n{s['code']}" for s in steps(r))


if __name__ == "__main__":
    r = pick(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(command(r) if r else f"no tested recipe for {sys.argv[1]!r} yet: https://local.sybilsolutions.ai")
