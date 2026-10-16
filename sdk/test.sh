#!/bin/sh
# Both SDKs agree on every card: the right card for its driver name, and the same steps.
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, os, shlex, subprocess, sys
sys.path.insert(0, "sdk/python")
import local_ai_registry as L
cat = json.load(open("dist/catalog.json"))
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
        weights = launch.get("weights") or []
        weights = weights if isinstance(weights, list) else [weights]
        downloads = [shlex.split(cmd.replace("\\\n", "")) for step in st if step["title"] == "Download the weights" for cmd in step["code"].split("\n\n")]
        for w, argv in zip(weights, downloads):
            files = w.get("files") or []
            files = files if isinstance(files, list) else [files]
            assert argv[:3] == ["hf", "download", w["repo"]], (k, argv)
            assert argv[3:argv.index("--revision")] == files, f"download ignores selected files for {k}"
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
PY
