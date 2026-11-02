#!/usr/bin/env python3
"""Bring a recipe validated before the lab from data/ into production, exactly as it was validated:
its launch becomes a frozen launch in registry/launches/, and a recipe points at it with a legacy proof.
A lab run of the card later replaces it.

    lab/import_data.py <data recipe id> <model id in registry/models.json> [--profile <name>]

Launches that need more than a plain container say so in the profile: `flags` (host IPC or networking),
`machines` (several computers, one per rank) or `kind: host` (a program, not a container). A launch on several
cards of one machine sets `cards`. A mount with its own `provision` (a draft model, say) is pinned to that
repository, not the model instance's. The proof's speed is decode at concurrency 1 when a sweep has that row."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

D = lab.ROOT / "data" / "registry"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def c1_decode(sweep):
    """Decode tok/s at concurrency 1 from a sweep's rows, the same measure the lab's speed gate reports."""
    for row in sweep.get("rows") or []:
        if row.get("concurrency") == 1 and (row.get("decode_tok_s_total") or row.get("decode_tok_s")):
            return row.get("decode_tok_s_total") or row.get("decode_tok_s")
    return None


def freeze(rid, model, name=None):
    r = load(D / "recipe" / f"{rid}.json")
    L = r["launch"]
    mi = load(D / "model-instance" / f"{r['model_instance_id']}.json")
    acc = (r.get("metadata") or {}).get("acceptance") or {}
    tps = None
    for sid in r.get("speed_sweep_ids") or []:
        f = D / "speed-sweep" / f"{sid}.json"
        if f.exists():
            s = load(f)
            tps = c1_decode(s) or (s.get("metrics") or {}).get("peak_generation_tps") or tps
    ctx = (r.get("serving") or {}).get("max_context_tokens") or 0
    engine = r["engine"]["name"].lower()
    if L["kind"] == "host":
        tag = r["metadata"].get("flm_tag")
        prof = {"kind": "host", "engine": engine, "about": f"{mi['served_name']} on {r['hardware_id']}, as validated in {rid}",
                "command": [engine, "serve", tag, "--port", str(L["container_port"])], "install": "https://github.com/FastFlowLM/FastFlowLM",
                "port": L["container_port"], "ctx": ctx, "seqs": 1, "vision": False, "backend": L.get("accelerator_backend")}
    else:
        weights, config, flags = [], None, []
        for m in L.get("mounts") or []:
            src = m["source"]
            if src.startswith("${MODEL_ROOT}/"):
                pv = m.get("provision") or {}
                weights.append({"repo": pv.get("repository") or mi["repository"], "revision": pv.get("revision") or mi["revision"], "at": m["target"], "layout": "dir"})
            elif "huggingface" in src:
                weights.append({"repo": mi["repository"], "revision": mi["revision"], "at": m["target"], "layout": "hub"})
            elif src.startswith("asset/"):
                config = {"at": m["target"], "text": (D / src).read_text(encoding="utf-8")}
        if L.get("ipc") == "host":
            flags.append("--ipc host")
        flags += [f"--security-opt {x}" for x in L.get("security_opt") or []] + [f"--ulimit {x}" for x in L.get("ulimits") or []]
        if L.get("network_mode") == "host" or r["hardware_count"] > 1 and "nnodes" in " ".join(L.get("arguments") or []):
            flags.append("--network host")
        prof = {"engine": engine, "about": f"{mi['served_name']} as validated in {rid}", "image": L["image"], "backend": L.get("accelerator_backend"),
                "port": L["container_port"], "entrypoint": L.get("entrypoint"), "args": L.get("arguments") or [], "env": L.get("environment") or {},
                "shm": L.get("shm_size"), "weights": weights, "config": config, "ctx": ctx, "seqs": 1,
                "vision": bool((r.get("capabilities") or {}).get("vision")), "cards": 1 if "--nnodes" in (L.get("arguments") or []) else r["hardware_count"]}
        if flags:
            prof["flags"] = flags
        if "--nnodes" in prof["args"]:
            prof["machines"] = r["hardware_count"]
    base = re.sub(r"[^a-z0-9.]+", "-", f"{engine}-{model}-{mi['weights'].get('precision') or mi['weights']['format']}-{ctx // 1024}k".lower()).strip("-")
    base = name or base
    name, n = base, 2
    while (lab.LAUNCHES / f"{name}.json").exists() and load(lab.LAUNCHES / f"{name}.json").get("frozen_from") != [rid]:
        name, n = f"{base}-v{n}", n + 1
    prof = {"id": name, **prof, "frozen_from": [rid]}
    (lab.LAUNCHES / f"{name}.json").write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
    digest = prof["image"].split("@sha256:")[1][:12] if prof.get("image") and "@sha256:" in prof["image"] else "host"
    recipe = {"model": model, "weights": f"{mi['repository']}@{mi['revision']}" if mi.get("revision") else "baked-into-image",
              "engine": f"{name}@{digest}", "set": {}, "card": r["hardware_id"],
              "proof": [{"at": (acc.get("accepted_at") or "")[:10] or None, "on": "legacy", "gates": "load chat" + (" reasoning tools" if (r.get("capabilities") or {}).get("tools") else ""),
                         "tps": round(tps, 1) if tps else None, "legacy": True}]}
    out = lab.recipe_path(recipe, lab.render(recipe))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recipe, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{rid} -> {out.relative_to(lab.ROOT)} (profile {name})")


if __name__ == "__main__":
    a = sys.argv[1:]
    freeze(a[0], a[1], a[a.index("--profile") + 1] if "--profile" in a else None)
