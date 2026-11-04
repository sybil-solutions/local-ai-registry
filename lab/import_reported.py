#!/usr/bin/env python3
"""Bring in a recipe someone else published, exactly as they launch it, marked reported.

Each input is data/reported/<source>/<repo>.json: the launches read from that source's repository, pinned
to a commit. Every launch whose weights are not GGUF becomes a frozen launch in registry/launches/ and a
recipe whose proof names the source and the numbers it reported. Our six gates have not run on it yet; a
lab run on the card (`lab.py try ... --on endpoint`, by anyone who owns one) replaces the reported proof.

    lab/import_reported.py data/reported/miaai-lab/*.json
"""
import json, re, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab


def head(repo):
    """The weights' current commit on the Hub, for a source that did not pin one."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/main", timeout=60) as r:
        return json.loads(r.read())["sha"]


def slug(*parts):
    return re.sub(r"[^a-z0-9.]+", "-", "-".join(str(p) for p in parts if p).lower()).strip("-")


def freeze(v, source):
    if v["format"] == "gguf" or not v.get("model_id"):
        return None
    rev = v.get("hf_revision") or head(v["hf_repo"])
    weights = []
    if v.get("weights_mount"):
        weights.append({"repo": v["hf_repo"], "revision": rev, "at": v["weights_mount"], "layout": v.get("weights_layout", "dir")})
    ctx = int(v.get("ctx") or 0)
    name = slug(v["engine"], v["model_id"], v["format"], f"{v['bpw']}bpw" if v.get("bpw") else None, f"{ctx // 1024}k",
                f"{v['machines']}x" if v.get("machines", 1) > 1 else None,
                f"{v['gpus_per_machine']}cards" if (v.get("gpus_per_machine") or 1) > 1 else None, v["card"] if v["card"] != "dgx-spark-gb10-128gb" else None)
    src = f"https://github.com/{v['repo']}/tree/{v['commit']}"
    b = v.get("build") or {}
    ep = v.get("entrypoint")
    ep = ep if isinstance(ep, list) else [ep] if ep else []
    if not v.get("image") and not b.get("dockerfile"):  # a Python environment on the host, set up by the source's own script
        prof = {"kind": "host", "engine": v["engine"], "about": f"{v['model']} as {source} runs it ({v['repo']})",
                "command": ep + (v.get("args") or []), "env": v.get("env") or {}, "install": src, "port": v["port"], "ctx": ctx,
                "seqs": 1, "vision": bool(v.get("vision")), "backend": "nvidia"}
        if (v.get("gpus_per_machine") or 1) > 1:  # one card is the default and stays out of the profile
            prof["cards"] = v["gpus_per_machine"]
    else:
        prof = {"engine": v["engine"], "about": f"{v['model']} as {source} runs it ({v['repo']})",
                "image": v.get("image"), "backend": "nvidia", "port": v["port"], "entrypoint": v.get("entrypoint"), "args": v.get("args") or [],
                "env": v.get("env") or {}, "shm": v.get("shm"), "weights": weights, "config": None, "ctx": ctx, "seqs": 1,
                "vision": bool(v.get("vision")), "cards": v.get("gpus_per_machine") or 1}
        if not prof["image"]:  # built from the source's Dockerfile at the pinned commit
            prof["build"] = {"repo": v["repo"], "commit": v["commit"], "dockerfile": b.get("dockerfile")}
    if v.get("flags"):
        prof["flags"] = v["flags"]
    if v.get("machines", 1) > 1:
        prof["machines"] = v["machines"]
    if v.get("extra_setup"):
        prof["setup"] = v["extra_setup"]
    prof = {"id": name, **prof, "source": src}
    digest = "host" if prof.get("kind") else prof["image"].split("@sha256:")[1][:12] if prof["image"] and "@sha256:" in prof["image"] else "unpinned" if prof["image"] else "build"
    claims = " ".join(k for k in ("reasoning", "tools", "vision") if v.get(k))
    m = v.get("measured") or {}
    recipe = {"model": v["model_id"], "weights": f"{v['hf_repo']}@{rev}", "engine": f"{name}@{digest}", "set": {}, "card": v["card"],
              "proof": [{"at": v.get("date"), "on": source, "src": f"{v['repo']}@{v['commit'][:12]}", "gates": "", "claims": claims,
                         "tps": m.get("decode_tps"), "prefill": m.get("prefill_tps"), "reported": True}]}
    n = (v.get("machines") or 1) * (prof.get("cards") or 1)  # every card in the setup, as lab.recipe_path counts them
    kind = v["engine"]
    many = f".{n}x" if n > 1 else ""
    out = lab.RECIPES / lab.card(v["card"])["vendor"] / v["card"] / f"{v['model_id']}.{kind}.{ctx // 1024}k{many}.json"
    return out, prof, recipe


def main(files):
    best = {}  # one recipe per file name: the variant with the best reported speed, the source's default on a tie
    for f in files:
        for v in json.loads(Path(f).read_text(encoding="utf-8")):
            made = freeze(v, Path(f).parent.name)
            if made and (made[0] not in best or (made[2]["proof"][0]["tps"] or 0) > (best[made[0]][2]["proof"][0]["tps"] or 0)):
                best[made[0]] = made
    ids = {}  # two different launches under one profile id would overwrite each other's file
    for _, prof, _ in best.values():
        if ids.setdefault(prof["id"], prof) != prof:
            raise SystemExit(f"profile {prof['id']} names two different launches; nothing written")
    for out, prof, recipe in best.values():
        if out.exists() and not json.loads(out.read_text(encoding="utf-8"))["proof"][0].get("reported"):
            print(f"skip {out.relative_to(lab.ROOT)}: a checked recipe is there")
            continue
        (lab.LAUNCHES / f"{prof['id']}.json").write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
        assert lab.recipe_path(recipe, lab.render(recipe)) == out, out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recipe, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{out.relative_to(lab.ROOT)} ({out.stat().st_size} bytes, profile {prof['id']})")


if __name__ == "__main__":
    main(sys.argv[1:])
