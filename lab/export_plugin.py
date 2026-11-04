#!/usr/bin/env python3
"""plugin/v2 and plugin/v3 recipes for Omarchy Local AI, generated from
dist/catalog.json: each card's picks, recommended first, then each validated model the picks lack (its first recipe),
and any other recipe that states host `needs` (RAM, disk) for the plugin to offer only where the machine has them. Recipes validated before the lab keep their original ids,
so running deployments stay recognised. `--check` fails if it is stale. Standard library only."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

ROOT = lab.ROOT
GATEWAY = "ghcr.io/sybil-solutions/gateway@sha256:110e8b11dc147e245e7c4cc2b24c85bd77e723f225227e85e4f20cc8fe3b7dce"  # org-v6: one attested index over linux/amd64 + linux/arm64
MIN_DRIVER = {"tabbyapi": "575.0", "sglang": "580.0", "vllm": "580.0", "llama.cpp": "535.0"}
# The only container flags a catalog launch may hand the plugin: device nodes a GPU backend
# already implies. Anything else (host IPC, seccomp, privileges) keeps the recipe out.
DEVICE_FLAGS = {"--device /dev/kfd", "--device /dev/dri"}
# Images with no build attestation binding the digest to a source commit (marketplace review, omacom/omarchy-plugin-
# marketplace#11207). Their recipes stay in the registry with their lab evidence, but the plugin is not handed them
# until they are rebuilt through an attested workflow and lab-proven again on the new digest.
NO_PROVENANCE = {
    "sha256:9d95829ab41a9307e9b25c86e4af1ec52c6a02f7888f0e13a1ce93c0a7ff3814",  # 0xsero/sglang-exl3:v0.7.0-multiarch, built by hand
    "sha256:cc2bfc369c709c53d239ef2af41d4ae2530904975816bb2ca484188008586ac9",  # 0xsero/qwen38-3090-sglang
    "sha256:ee978fb3e3d11cf8577a014539a7ad4e2a8dff95163fc5d96c0a06fcf9c64640",  # morrowmake/vllm-cmp170hx
}


def exportable(launch):
    """A launch the plugin may run: one plain container whose flags are at most the GPU device nodes."""
    return (not any(launch.get(x) for x in ("kind", "machines", "build"))
            and set(launch.get("flags") or []) <= DEVICE_FLAGS)


def entry(key, r, meta):
    p = lab.profile(r["engine"])
    L = lab.render(r)
    caps = {"chat": True, "reasoning": "reasoning" in r["proof"][0]["gates"], "tools": "tools" in r["proof"][0]["gates"], "vision": L["vision"], **({"diffusion": True} if (meta.get("models", {}).get(r.get("model")) or {}).get("diffusion") else {})}
    if "plugin" in p:  # a launch frozen from a validated recipe: its own export fields
        x = p["plugin"]
        assert len(L["weights"]) == len(x["weights"]), "plugin input mounts do not match the accepted launch"
        rid = p["ids"].get(r["card"]) or f"{Path(key).name}.{r['card']}"
        if L.get("prepare") or L.get("resources"):
            for w, xw in zip(L["weights"], x["weights"]):
                assert set(xw) <= {"sizeGb", "layout", "mountPath", "dir", "files"}, "plugin cannot override pinned inputs"
                assert xw["layout"] == w.get("layout", "dir"), "plugin input layout changed"
                assert xw["mountPath"].rstrip("/") + ("/" + xw["dir"] if xw.get("dir") else "") == w["at"], "plugin input location changed"
                files = w.get("files") or []
                files = files if isinstance(files, list) else [files]
                assert sorted(filter(None, (xw.get("files") or "").split(","))) == sorted(files), "plugin input files changed"
        weights = [{**xw, "repository": w["repo"], "revision": w["revision"]} for w, xw in zip(L["weights"], x["weights"])]
        return {"id": rid, "name": x["name"], "family": x["family"], "format": x["format"], "engine": x["engine"], "servedName": x["servedName"],
                "sizeGb": x["sizeGb"], "cards": L.get("cards", 1), "image": L["image"], "minDriver": x["minDriver"], "weights": weights,
                "asset": {"name": x["asset"], "mountPath": L["config"]["at"], "text": L["config"]["text"]} if L["config"] else None,
                "scratch": x["scratch"], "launch": {"entrypoint": L["entrypoint"], "arguments": L["args"], "environment": L["env"], "port": L["port"], "shm": L["shm"],
                                                    **({"flags": L["flags"]} if L.get("flags") else {})},
                "serving": x["serving"], "capabilities": {**x["capabilities"], **{k: v for k, v in caps.items() if v}},
                **({"needs": L["needs"]} if L.get("needs") else {}),
                **({"prepare": L["prepare"]} if L.get("prepare") else {}),
                **({"resources": L["resources"]} if L.get("resources") else {})}
    if "defaults" not in p:
        return derived(key, r, meta, p, L, caps)
    m = meta["models"][r["model"]]
    b = meta["builds"][r["weights"]]
    name = L["weights"]["at"].rsplit("/", 1)[1]
    return {"id": f"{Path(key).name}.{r['card']}", "name": m["name"], "family": m["family"], "format": b["format"], "engine": p["engine"],
            "servedName": name, "sizeGb": b["size_gb"], "cards": 1, "image": L["image"], "minDriver": MIN_DRIVER.get(p["engine"], ""),
            "weights": [{"repository": L["weights"]["repo"], "revision": L["weights"]["revision"], "sizeGb": b["size_gb"], "layout": "dir",
                         "mountPath": L["weights"]["at"].rsplit("/", 1)[0], "dir": name, "files": ""}],
            "asset": {"name": f"{Path(key).name}.config.yml", "mountPath": L["config"]["at"], "text": L["config"]["text"]} if L["config"] else None,
            "scratch": None, "launch": {"entrypoint": L["entrypoint"], "arguments": L["args"], "environment": L["env"], "port": L["port"], "shm": L["shm"]},
            "serving": {"ctxTokens": L["ctx"], "kvTokens": lab.kv_tokens(r, L)}, "capabilities": caps,
            # a template profile's host needs (RAM for offloaded experts, NVMe) reach the plugin like a frozen launch's
            **({"needs": L["needs"]} if L.get("needs") else {})}


def derived(key, r, meta, p, L, caps):
    """A frozen launch without a plugin block, described from what the lab accepted: the launch as it ran, and the
    model and build it names in registry/models.json. Weights keep their layout (a folder, or a Hugging Face cache)."""
    m, b = meta["models"][r["model"]], meta["builds"][r["weights"]]
    ws = L["weights"] if isinstance(L["weights"], list) else [L["weights"]]
    weights = []
    for w in ws:
        if w.get("layout") == "hub":
            weights.append({"repository": w["repo"], "revision": w["revision"], "sizeGb": b["size_gb"], "layout": "hub", "mountPath": w["at"], "files": ""})
        else:
            files = w.get("files") or []
            # weights at /models itself mount there, with no folder of their own
            parent, _, leaf = w["at"].rstrip("/").rpartition("/")
            mount, folder = (parent, leaf) if parent else (w["at"].rstrip("/"), "")
            weights.append({"repository": w["repo"], "revision": w["revision"], "sizeGb": b["size_gb"], "layout": "dir",
                            "mountPath": mount, "dir": folder, "files": ",".join(files if isinstance(files, list) else [files])})
    args = L["args"] or []
    flag = lambda *names: next((args[i + 1] for i, a in enumerate(args[:-1]) if a in names), None)
    served = flag("--served-model-name", "--alias", "--model-name") or (ws[0]["at"].rsplit("/", 1)[1] if ws else flag("--model", "-m", "--model-path") or r["model"])
    return {"id": f"{Path(key).name}.{r['card']}", "name": m["name"], "family": m["family"], "format": b["format"], "engine": p["engine"],
            "servedName": served, "sizeGb": b["size_gb"], "cards": L.get("cards", 1), "image": L["image"], "minDriver": MIN_DRIVER.get(p["engine"], ""),
            "weights": weights,
            "asset": {"name": f"{Path(key).name}.config.yml", "mountPath": L["config"]["at"], "text": L["config"]["text"]} if L["config"] else None,
            "scratch": None, "launch": {"entrypoint": L["entrypoint"], "arguments": args, "environment": L["env"], "port": L["port"], "shm": L["shm"]},
            "serving": {"ctxTokens": L["ctx"], "kvTokens": lab.kv_tokens(r, L)}, "capabilities": caps,
            **({"needs": L["needs"]} if L.get("needs") else {})}


def aa(r, meta):
    """The model's Artificial Analysis Intelligence Index (registry/models.json, lab/aa.py), a build of another model
    taking its base's: {"aa": 41.8}, or nothing when AA does not list it."""
    models = meta.get("models") or {}
    m, seen = models.get(r.get("model")) or {}, 0
    while "of" in (m.get("aa") or {}) and seen < 4:
        m, seen = models.get(m["aa"]["of"]) or {}, seen + 1
    i = (m.get("aa") or {}).get("index")
    return {"aa": i} if isinstance(i, (int, float)) else {}


def mounts_ok(e):
    """The plugin's own rules: an id it accepts (a-z, 0-9, dots and dashes, at most 64), and no input mounted at, or
    above, where a prepared pack is mounted."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", e["id"]):
        return False
    out = (e.get("prepare") or {}).get("at")
    return not out or not any(out == w["mountPath"] or out.startswith(w["mountPath"].rstrip("/") + "/") for w in e["weights"])


def describable(r):
    """The plugin can describe a recipe whose launch has a plugin block or template defaults, or whose model and build
    registry/models.json knows (and whose image is pinned by digest)."""
    p = lab.profile(r["engine"])
    if "defaults" in p or "plugin" in p:
        return True
    meta = json.loads((ROOT / "registry" / "models.json").read_text(encoding="utf-8"))
    return (r.get("model") in meta["models"] and r.get("weights") in meta["builds"] and "@sha256:" in (p.get("image") or "")
            and bool(lab.render(r)["weights"]))


def eligible(r, version, vram_gb=None):  # vram_gb: one card's memory, None reads registry/cards
    L = r["launch"]
    # a recipe its publisher reported (not yet run by the lab) is exported too, flagged `reported`, so the plugin can
    # say so; one withdrawn is not
    # a launch with a companion (a second container on a card of another kind) is a candidate until a plugin runs it
    if any(L.get(x) for x in ("kind", "flags", "machines", "build", "companion")) or r["proof"][0].get("withdrawn"):
        return False
    if not describable(r):
        return False
    # the KV floor (lab.KV_FLOOR_TOKENS): on a setup over lab.KV_FLOOR_ABOVE_VRAM_GB, a recipe holding less KV than
    # that stays in the registry with its evidence but never reaches the plugin
    if lab.below_kv_floor(r, vram_gb=vram_gb):
        return False
    if (lab.render(r).get("image") or "").rpartition("@")[2] in NO_PROVENANCE:
        return False
    if L.get("prepare") or L.get("resources"):
        if version == 2:
            return False  # old plugins silently ignore these fields
        lab.check_execution(L)
        if not r["proof"][0].get("reported") and r["proof"][0].get("launch_sha256") != lab.launch_hash(L):
            return False  # acceptance must bind the whole typed launch
    return True


def build(version=2):
    cat = json.loads((ROOT / "dist" / "catalog.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "registry" / "models.json").read_text(encoding="utf-8"))
    hw = {}
    for card, c in sorted(cat["cards"].items()):
        vram = c.get("vram_gb") or 0  # one card's memory, for the KV floor (times the launch's cards)
        # the plugin runs one plain container per card: no host programs, host IPC or networking, or several machines
        # the picks, then any other recipe that states host `needs`: the plugin offers those only on a machine with
        # that much free RAM and disk (e.g. experts offloaded to system RAM, tables read from NVMe)
        keys = c["picks"] + [k for k in c["more"] if cat["recipes"][k]["launch"].get("needs")]
        ok = [k for k in keys if exportable(cat["recipes"][k]["launch"])
              and not cat["recipes"][k]["proof"][0].get("reported")  # the plugin runs one plain container we have checked
              and ("defaults" in lab.profile(cat["recipes"][k]["engine"]) or "plugin" in lab.profile(cat["recipes"][k]["engine"]))]  # and can describe
        # and, from the rest, the first recipe of each model the picks do not have, so every validated model reaches the
        # card once and the list does not repeat a model in a weaker setting
        # then each multi-card setup on one machine, its own picks and models the same way
        multi = [k for st in c.get("setups", []) if st["cards"] > 1 and st["machines"] == 1 for k in st["picks"] + st["more"]]
        ok = [k for k in c["picks"] if eligible(cat["recipes"][k], version, vram)]
        names = {entry(k, cat["recipes"][k], meta)["name"] for k in ok}
        for k in c["more"]:
            r = cat["recipes"][k]
            if not eligible(r, version, vram):
                continue
            name = entry(k, r, meta)["name"]
            if r["launch"].get("needs") or name not in names:
                ok.append(k)
                names.add(name)
        for n in sorted({cat["recipes"][k]["launch"].get("cards", 1) for k in multi}):
            seen = set()
            for k in multi:
                r = cat["recipes"][k]
                if r["launch"].get("cards", 1) != n or k in ok or not eligible(r, version, vram):
                    continue
                name = entry(k, r, meta)["name"]
                if name not in seen:
                    ok.append(k)
                    seen.add(name)
        # the plugin refuses a whole catalog for one recipe it cannot take, so none such leaves here
        ok = [k for k in ok if mounts_ok(entry(k, cat["recipes"][k], meta))]
        if ok:
            hw[card] = {"match": c["match"], "recipes": [dict(entry(k, cat["recipes"][k], meta), **({"reported": True} if cat["recipes"][k]["proof"][0].get("reported") else {}),
                                                              **aa(cat["recipes"][k], meta)) for k in ok]}
            for e in hw[card]["recipes"]:  # a needs block reaches the plugin only in the checked shape
                if "resources" in e:
                    e["launch"]["resources"] = e.pop("resources")
                if "needs" in e:
                    lab.check_needs(e["needs"])
    head = {"schemaVersion": f"omarchy-local-ai/recipes/{version}", "registryCommit": None, "generatedAt": None, "gateway": {"image": GATEWAY}}
    lines = ["{"] + [f'  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},' for k, v in head.items()] + ['  "hardware": {']
    lines += [f'    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(",", ":"))}' + ("," if i < len(hw) - 1 else "") for i, (k, v) in enumerate(hw.items())]
    return "\n".join(lines + ["  }", "}"]) + "\n"


if __name__ == "__main__":
    ok = True
    for version in (2, 3):
        out = ROOT / "plugin" / f"v{version}" / "recipes.json"
        text = build(version)
        if "--check" in sys.argv:
            current = out.exists() and out.read_text(encoding="utf-8") == text
            print(f"{out.relative_to(ROOT)} is " + ("current" if current else "stale: run lab/export_plugin.py"))
            ok = ok and current
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            json.loads(text)
            print(f"wrote {out.relative_to(ROOT)}: {len(text)} bytes")
    sys.exit(0 if ok else 1)
