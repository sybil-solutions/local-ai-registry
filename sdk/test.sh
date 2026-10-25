#!/bin/sh
# Both SDKs agree on every card: the right card for its driver name, and the same steps.
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, os, shlex, subprocess, sys
sys.path.insert(0, "sdk/python")
import local_ai_registry as L
cat = json.load(open("dist/catalog.json"))
def downloads(st):
    """(repo, files) of each weights download in the steps: `hf download` lines, or the pinned runtime's snapshot_download."""
    out = []
    for step in st:
        if step["title"] != "Download the weights":
            continue
        strict = step["code"].startswith("(\nset -e\n")
        for cmd in step["code"].split("\n") if strict else step["code"].split("\n\n"):
            cmd = cmd.replace("\\\n", "")
            if cmd.startswith('(hubenv "$c"; exec ') and cmd.endswith(")"):  # a download isolated in its staging cache
                cmd = cmd[len('(hubenv "$c"; exec '):-1]
            argv = shlex.split(cmd)
            if argv[:2] == ["hf", "download"]:
                out.append((argv[2], argv[3:argv.index("--revision")]))
            elif "-c" in argv and "snapshot_download" in " ".join(argv):
                i = argv.index("-c")
                out.append((argv[i + 3], argv[i + 6:]))
    return out
def check_downloads(k, launch, st):
    weights = launch.get("weights") or []
    weights = weights if isinstance(weights, list) else [weights]
    got = downloads(st)
    assert len(got) == len([w for w in weights if w and w.get("repo")]), (k, got)
    for w, (repo, fs) in zip(weights, got):
        files = w.get("files") or []
        files = files if isinstance(files, list) else [files]
        assert repo == w["repo"], (k, repo)
        assert fs == files, f"download ignores selected files for {k}"
# Identical driver names need memory to distinguish 10 and 12 GB variants.
variants = {"cards": {
    "rtx-3080-10gb": {"name": "GeForce RTX 3080", "vram_gb": 10},
    "rtx-3080-12gb": {"name": "GeForce RTX 3080", "vram_gb": 12},
    "rtx-3080-ti-12gb": {"name": "GeForce RTX 3080 Ti", "vram_gb": 12}}}
cases = [["NVIDIA GeForce RTX 3080", 10, "rtx-3080-10gb"],
         ["NVIDIA GeForce RTX 3080", 12, "rtx-3080-12gb"],
         ["NVIDIA GeForce RTX 3080", 11.8, "rtx-3080-12gb"],
         ["NVIDIA GeForce RTX 3080 Ti", 12, "rtx-3080-ti-12gb"],
         ["NVIDIA GeForce RTX 3080", None, "rtx-3080-10gb"]]
for gpu, memory, expected in cases:
    assert L.detect(variants, gpu, memory) == expected, (gpu, memory, expected)
variant_js = subprocess.run(["node", "--input-type=module", "-e", '''
import {readFileSync} from "node:fs"; import {detect} from "./sdk/js/index.js";
const {variants, cases} = JSON.parse(readFileSync(0, "utf8"));
for (const [gpu, memory, expected] of cases) {
  if (detect(variants, gpu, memory) !== expected) throw new Error(`${gpu}: ${memory}`);
}
'''], input=json.dumps({"variants": variants, "cases": cases}), text=True, check=True)
names = {i: c["name"] for i, c in cat["cards"].items()}
for i, c in cat["cards"].items():
    got = L.detect(cat, "NVIDIA GeForce " + c["name"] if c["vendor"] == "nvidia" else c["name"], c["vram_gb"])
    assert got == i, (i, got)
js = subprocess.run(["node", "--input-type=module", "-e", """
import { readFileSync } from "node:fs"; import { detect, steps } from "./sdk/js/index.js";
const cat = JSON.parse(readFileSync("dist/catalog.json"));
const out = {}; for (const [i, c] of Object.entries(cat.cards)) out[i] = [detect(cat, c.name, c.vram_gb), c.setups.flatMap((s) => [...s.picks, ...s.more]).map((k) => steps({ key: k, ...cat.recipes[k] }))];
console.log(JSON.stringify(out));"""], capture_output=True, text=True, check=True).stdout
n = 0
for i, (got, sts) in json.loads(js).items():
    assert got == i, (i, got)
    for k, st in zip([k for s in cat["cards"][i]["setups"] for k in s["picks"] + s["more"]], sts):
        assert st == L.steps({"key": k, **cat["recipes"][k]}), f"js and python steps differ for {k}"
        if cat["recipes"][k]["launch"].get("backend") == "cpu":
            command = st[-1]["code"]
            assert "--gpus" not in command and "--device" not in command, f"CPU launch requires a GPU for {k}"
        launch = cat["recipes"][k]["launch"]
        check_downloads(k, launch, st)
        if launch.get("prepare"):
            prep = next(s["code"] for s in st if s["title"] == "Prepare the model (first start)")
            argv = shlex.split(prep.split("\n", 1)[1].replace("\\\n", ""))
            image_index = argv.index(launch["image"])
            assert argv[image_index + 1:] == launch["prepare"]["args"], k
            mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-v"]
            output = next(m for m in mounts if m.endswith(":" + launch["prepare"]["at"]))
            assert output + ":ro" in st[-1]["code"], f"prepared model missing from serving mounts for {k}"
            assert all(m.endswith(":ro") for m in mounts if m != output), f"preparation can modify input weights for {k}"
        n += 1
# Execute only a synthetic command against a shell function, never the Docker daemon.
fixture = {"model": "node-template", "launch": {"image": "example.invalid/server:fixture", "backend": "cpu",
    "weights": [], "port": 8000, "args": ["--node-rank", "${NODE_RANK}", "--literal", "$(printf executed)"],
    "env": {"VLLM_HOST_IP": "${NODE_IP}", "NOTE": "${NOTE}"}}}
py_steps = L.steps(fixture)
js_steps = json.loads(subprocess.run(["node", "--input-type=module", "-e",
    'import {readFileSync} from "node:fs"; import {steps} from "./sdk/js/index.js"; console.log(JSON.stringify(steps(JSON.parse(readFileSync(0,"utf8")))));'],
    input=json.dumps(fixture), capture_output=True, text=True, check=True).stdout)
assert py_steps == js_steps
script = 'docker() { printf "%s\\n" "$@"; };\n' + py_steps[-1]["code"]
argv = subprocess.run(["sh", "-c", script], env={**os.environ, "NODE_RANK": "1", "NODE_IP": "192.0.2.1", "NOTE": "$(printf injected)"},
    capture_output=True, text=True, check=True).stdout.splitlines()
assert argv[argv.index("--node-rank") + 1] == "1", argv
assert "VLLM_HOST_IP=192.0.2.1" in argv, argv
assert "NOTE=$(printf injected)" in argv and argv[-1] == "$(printf executed)", argv
# Typed offload commands preserve execution constraints in both SDKs.
fixture = {"model": "typed-pack", "launch": {"image": "example.invalid/server@sha256:" + "a"*64,
    "backend": "nvidia", "weights": [{"repo": "fixture/raw", "revision": "b"*40, "at": "/models/raw"}],
    "port": 8000, "args": ["serve"], "entrypoint": ["python3", "/engine.py"], "shm": "8g",
    "env": {"MODE": "exact", "NVIDIA_VISIBLE_DEVICES": "all"},
    "resources": {"memoryBytes": 59055800320, "memorySwapBytes": 59055800320, "memlockUnlimited": True, "ipcLock": True},
    "prepare": {"at": "/models", "args": ["prepare"], "verifyArgs": ["verify-pack"], "gpu": False, "sizeGb": 117}}}
py_steps = L.steps(fixture)
js_steps = json.loads(subprocess.run(["node", "--input-type=module", "-e",
    'import {readFileSync} from "node:fs"; import {steps} from "./sdk/js/index.js"; console.log(JSON.stringify(steps(JSON.parse(readFileSync(0,"utf8")))));'],
    input=json.dumps(fixture), capture_output=True, text=True, check=True).stdout)
assert py_steps == js_steps
for step in py_steps[1:]:
    argv = shlex.split(step['code'].split("\n", 1)[-1].replace("\\\n", "")) if step['title'].startswith('Prepare') else shlex.split(step['code'].replace("\\\n", ""))
    assert "59055800320" in argv and "memlock=-1:-1" in argv and "IPC_LOCK" in argv
    assert "MODE=exact" in argv and "8g" in argv and "python3" in argv
    assert argv[argv.index(fixture['launch']['image'])+1:] == ["/engine.py", {"Prepare the model (first start)": "prepare", "Verify the prepared model": "verify-pack", "Start the server": "serve"}[step['title']]]
    mounts = [argv[i+1] for i, value in enumerate(argv) if value == '-v']
    assert any(m.endswith(':/models/raw:ro') for m in mounts)
    if step['title'] != 'Start the server':
        assert '--gpus' not in argv and '--device' not in argv and 'NVIDIA_VISIBLE_DEVICES=all' not in argv
    if step['title'] == 'Verify the prepared model':
        assert all(m.endswith(':ro') for m in mounts)
print(f"sdk ok: {len(cat['cards'])} cards, {n} recipes, js and python agree")
# A native binary and its Python adapter both need installation.
fixture = {"model": "native-fixture", "launch": {"kind": "host", "pip": ["tokenizers==0.22.2"],
    "install": "https://example.invalid/engine.tar.gz#sha256=" + "a" * 64,
    "command": ["python3", "adapter.py"], "weights": [], "config": {"at": "adapter.py", "text": "print('ready')"}}}
py_steps = L.steps(fixture)
js_steps = json.loads(subprocess.run(["node", "--input-type=module", "-e",
    'import {readFileSync} from "node:fs"; import {steps} from "./sdk/js/index.js"; console.log(JSON.stringify(steps(JSON.parse(readFileSync(0,"utf8")))));'],
    input=json.dumps(fixture), capture_output=True, text=True, check=True).stdout)
assert py_steps == js_steps
assert "pip install tokenizers==0.22.2" in py_steps[0]["code"]
assert "shasum -a 256 -c" in py_steps[1]["code"] and "tar xzf" in py_steps[1]["code"]
assert py_steps[2]["title"] == "Write the launcher"
# Two weights in one Hugging Face cache mount it once (docker refuses a duplicate mount point).
fixture = {"model": "hub-pair", "launch": {"image": "example.invalid/server:fixture", "backend": "cpu", "port": 8000, "args": [],
    "weights": [{"repo": "fixture/model", "revision": "b" * 40, "at": "/root/.cache/huggingface", "layout": "hub"},
                {"repo": "fixture/draft", "revision": "c" * 40, "at": "/root/.cache/huggingface", "layout": "hub"}]}}
py_steps = L.steps(fixture)
js_steps = json.loads(subprocess.run(["node", "--input-type=module", "-e",
    'import {readFileSync} from "node:fs"; import {steps} from "./sdk/js/index.js"; console.log(JSON.stringify(steps(JSON.parse(readFileSync(0,"utf8")))));'],
    input=json.dumps(fixture), capture_output=True, text=True, check=True).stdout)
assert py_steps == js_steps
assert py_steps[-1]["code"].count("/root/.cache/huggingface") == 1, py_steps[-1]["code"]
# Every catalog recipe, listed on a card or not: the same steps and the same one-script command in both SDKs.
every = json.loads(subprocess.run(["node", "--input-type=module", "-e", """
import { readFileSync } from "node:fs"; import { steps, command } from "./sdk/js/index.js";
const cat = JSON.parse(readFileSync("dist/catalog.json"));
console.log(JSON.stringify(Object.fromEntries(Object.entries(cat.recipes).map(([k, r]) => [k, [steps({ key: k, ...r }), command({ key: k, ...r })]]))));"""],
    capture_output=True, text=True, check=True).stdout)
assert sorted(every) == sorted(cat["recipes"])
for k, r in cat["recipes"].items():
    assert every[k] == [L.steps({"key": k, **r}), L.command({"key": k, **r})], f"js and python differ for {k}"
# A pinned host runtime (lab/test_host_runtime.py): the same steps for each shape, and both SDKs refuse every bad one.
sys.path.insert(0, "lab")
import test_host_runtime as T
import copy
full = T.launch()
shapes = {"full": full,
    "archive and python, nothing to download": {**full, "runtime": {k: full["runtime"][k] for k in ("archive", "python")}, "weights": [], "config": None},
    "no launcher": {**full, "config": None},
    "hub without ref": {**full, "weights": [{k: v for k, v in full["weights"][0].items() if k != "ref"}]},
    "hub with files": {**full, "weights": [{**full["weights"][0], "files": ["config.json", "model.safetensors"]}]},
    "two hub weights in one cache": {**full, "weights": [full["weights"][0], {**full["weights"][0], "repo": "org/draft", "revision": "c" * 40}]},
    "nested under one directory": T.launch(root="opt/rt"),
    "same repo in two separate caches": {**full, "weights": [full["weights"][0], {**full["weights"][0], "at": "rt/hf-other", "revision": "d" * 40}]},
    "hub weights without a runtime": {k: v for k, v in full.items() if k != "runtime"},
    "with a macOS floor": {**full, "runtime": {**full["runtime"], "macos": 15}}}
cases = {**{f"ok: {k}": {"model": "rt", "launch": v} for k, v in shapes.items()},
         **{f"bad: {k}": {"model": "rt", "launch": make()} for k, make in T.BAD.items()}}
js = json.loads(subprocess.run(["node", "--input-type=module", "-e", """
import { readFileSync } from "node:fs"; import { steps, command } from "./sdk/js/index.js";
const out = {}; for (const [k, r] of Object.entries(JSON.parse(readFileSync(0, "utf8")))) {
  try { out[k] = [steps(r), command(r)]; } catch (e) { out[k] = { threw: String(e.message) }; } }
console.log(JSON.stringify(out));"""], input=json.dumps(cases), capture_output=True, text=True, check=True).stdout)
for k, r in cases.items():
    if k.startswith("ok: "):
        assert js[k] == [L.steps(copy.deepcopy(r)), L.command(copy.deepcopy(r))], f"js and python differ for runtime {k}"
        assert js[k][1].startswith("set -e\n\n"), f"command for {k} does not stop at the first failure"
        check_downloads(k, r["launch"], js[k][0])
    else:
        assert "threw" in js[k] and js[k]["threw"].startswith("invalid host runtime"), f"js renders {k}: {js[k]}"
        try:
            L.steps(r)
            raise AssertionError(f"python renders {k}")
        except ValueError:
            pass
print(f"sdk ok: {len(every)} catalog recipes and {len(cases)} host runtime cases, js and python agree")
PY
