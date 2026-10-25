#!/usr/bin/env python3
"""Per-entry report of the generated files in this branch (dist/catalog.json, plugin/v2 and v3 recipes.json).

  python3 data/sources/rtx4080-laptop-tabbyapi-exl3-container/generated_delta.py BASE_REF OUT.json [--today YYYY-MM-DD]

Three versions of each file are compared:
  base        BASE_REF as committed
  base_today  BASE_REF rebuilt with `make` (catalog, hardware, plugin) on the same date as this branch
  head        this checkout
base -> base_today is the date-only churn (registry/models.json max_age_days); base_today -> head is what
this branch itself changes. Every added, removed or changed recipe, card pick and plugin id is listed, and
the plugin v2 and v3 changes are checked to be the same. The base rebuild runs in a temporary git worktree,
which is removed afterwards. Standard library only; needs git and Python 3.11+.
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FILES = {"catalog": "dist/catalog.json", "plugin_v2": "plugin/v2/recipes.json", "plugin_v3": "plugin/v3/recipes.json"}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout


def rebuilt(ref, today):
    d = tempfile.mkdtemp(prefix="generated-delta-")
    try:
        git("worktree", "add", "-q", "--detach", d, ref)
        # `make all` there, with catalog.py's date pinned to the branch's build date (catalog.build takes it)
        subprocess.run([sys.executable, "-I", "-c", "import datetime as dt, sys; sys.path.insert(0, 'lab'); import catalog; "
                        "catalog.OUT.write_text(catalog.build(dt.date.fromisoformat(sys.argv[1])), encoding='utf-8')", today],
                       cwd=d, check=True, capture_output=True, text=True)
        for script in ("lab/hardware.py", "lab/export_plugin.py"):
            subprocess.run([sys.executable, script], cwd=d, check=True, capture_output=True, text=True)
        return {k: json.load(open(os.path.join(d, f))) for k, f in FILES.items()}
    finally:
        git("worktree", "remove", "--force", d)
        shutil.rmtree(d, ignore_errors=True)


def at(ref):
    return {k: json.loads(git("show", f"{ref}:{f}")) for k, f in FILES.items()}


def here():
    return {k: json.load(open(os.path.join(ROOT, f))) for k, f in FILES.items()}


def catalog_delta(a, b):
    ra, rb = a["recipes"], b["recipes"]
    ca, cb = a["cards"], b["cards"]
    cards = {}
    for c in sorted(set(ca) | set(cb)):
        x, y = ca.get(c), cb.get(c)
        if x == y:
            continue
        if x is None or y is None:
            cards[c] = "added" if x is None else "removed"
            continue
        e = {}
        for k in ("picks", "more"):
            add, rem = sorted(set(y.get(k, [])) - set(x.get(k, []))), sorted(set(x.get(k, [])) - set(y.get(k, [])))
            if add or rem:
                e[k] = {"added": add, "removed": rem}
        other = sorted(k for k in set(x) | set(y) if k not in ("picks", "more") and x.get(k) != y.get(k))
        if other:
            e["other_fields_changed"] = other
        cards[c] = e
    return {"recipes_added": sorted(set(rb) - set(ra)), "recipes_removed": sorted(set(ra) - set(rb)),
            "recipes_changed": sorted(k for k in set(ra) & set(rb) if ra[k] != rb[k]), "cards": cards,
            "other_top_level_changed": sorted(k for k in set(a) | set(b) if k not in ("recipes", "cards") and a.get(k) != b.get(k))}


def plugin_ids(p):
    return {(c, r["id"]) for c, x in p["hardware"].items() for r in x["recipes"]}


def plugin_delta(a, b):
    ia, ib = plugin_ids(a), plugin_ids(b)
    ea = {(c, r["id"]): r for c, x in a["hardware"].items() for r in x["recipes"]}
    eb = {(c, r["id"]): r for c, x in b["hardware"].items() for r in x["recipes"]}
    name = lambda e: e.get("name")
    return {"added": [{"card": c, "id": i, "name": name(eb[(c, i)])} for c, i in sorted(ib - ia)],
            "removed": [{"card": c, "id": i, "name": name(ea[(c, i)])} for c, i in sorted(ia - ib)],
            "changed": [{"card": c, "id": i} for c, i in sorted(ia & ib) if ea[(c, i)] != eb[(c, i)]],
            "cards_added": sorted(set(b["hardware"]) - set(a["hardware"])),
            "cards_removed": sorted(set(a["hardware"]) - set(b["hardware"]))}


def strip(d):  # v2 and v3 entries differ in shape; compare the (card, id) changes
    return {k: [(e["card"], e["id"]) for e in d[k]] for k in ("added", "removed", "changed")} | {k: d[k] for k in ("cards_added", "cards_removed")}


def main():
    base_ref, out = sys.argv[1], sys.argv[2]
    today = sys.argv[sys.argv.index("--today") + 1] if "--today" in sys.argv else __import__("datetime").date.today().isoformat()
    base, base_today, head = at(base_ref), rebuilt(base_ref, today), here()
    rep = {"base_ref": base_ref, "base_sha": git("rev-parse", base_ref).strip(), "head_sha": git("rev-parse", "HEAD").strip(),
           "head_worktree_clean_for_generated_files": git("status", "--porcelain", "--", *FILES.values()) == "",
           "build_date": today, "files": FILES, "views": {}}
    for label, (a, b) in {"age_churn (base -> base_today)": (base, base_today), "this_branch (base_today -> head)": (base_today, head),
                          "total (base -> head)": (base, head)}.items():
        v = {"catalog": catalog_delta(a["catalog"], b["catalog"]),
             "plugin_v2": plugin_delta(a["plugin_v2"], b["plugin_v2"]), "plugin_v3": plugin_delta(a["plugin_v3"], b["plugin_v3"])}
        v["plugin_v2_v3_parity"] = strip(v["plugin_v2"]) == strip(v["plugin_v3"])
        v["counts"] = {"catalog_recipes_added": len(v["catalog"]["recipes_added"]), "catalog_recipes_removed": len(v["catalog"]["recipes_removed"]),
                       "catalog_recipes_changed": len(v["catalog"]["recipes_changed"]), "catalog_cards_touched": len(v["catalog"]["cards"]),
                       "plugin_v3_added": len(v["plugin_v3"]["added"]), "plugin_v3_removed": len(v["plugin_v3"]["removed"]),
                       "plugin_v3_changed": len(v["plugin_v3"]["changed"])}
        rep["views"][label] = v
    br = rep["views"]["this_branch (base_today -> head)"]
    tot = rep["views"]["total (base -> head)"]
    names = {e["name"] for k in ("added", "removed") for e in rep["views"]["age_churn (base -> base_today)"]["plugin_v3"][k]}
    rep["conclusions"] = {
        "this_branch_plugin_delta_empty": not any(br["plugin_v2"][k] or br["plugin_v3"][k] for k in ("added", "removed", "changed", "cards_added", "cards_removed")),
        "this_branch_catalog_recipes": {"added": br["catalog"]["recipes_added"], "removed": br["catalog"]["recipes_removed"], "changed": br["catalog"]["recipes_changed"]},
        "this_branch_catalog_cards": sorted(br["catalog"]["cards"]),
        "this_card_in_plugin_v2": "rtx-4080-laptop-12gb" in head["plugin_v2"]["hardware"],
        "this_card_in_plugin_v3": "rtx-4080-laptop-12gb" in head["plugin_v3"]["hardware"],
        "age_churn_plugin_models": sorted(names),
        "plugin_v2_v3_parity_all_views": all(v["plugin_v2_v3_parity"] for v in rep["views"].values()),
        "total_equals_age_churn_for_plugin": strip(tot["plugin_v3"]) == strip(rep["views"]["age_churn (base -> base_today)"]["plugin_v3"]),
    }
    open(out, "w").write(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": {k: v["counts"] for k, v in rep["views"].items()}, "conclusions": rep["conclusions"]}, indent=1))


if __name__ == "__main__":
    main()
