#!/usr/bin/env python3
"""Artificial Analysis Intelligence Index for every model in registry/models.json.

A card's "smart" pick is the model with the highest index among the recipes that run on it, so the registry
keeps one number per model: `aa = {index, name, slug, estimated, at}`, or `aa = {of: <model>}` for a
quantized build of a model that has its own entry (the build inherits its base's score). A model AA does
not list has no `aa` and is never a card's smart pick.

The index is read from AA's public leaderboard page (the API needs a key). Every model is served with
reasoning on, so the variant taken is the reasoning one at its default effort, in this order: "(Reasoning)",
"(Medium)", no suffix, "(High)", "(Xhigh)", "(Max)", "(Low)". The variant is recorded, so a number can be
checked against the leaderboard. Run it by hand; it rewrites registry/models.json. Standard library only."""
import json, re, sys, urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "registry" / "models.json"
URL = "https://artificialanalysis.ai/leaderboards/models"
ORDER = ["(reasoning)", "(medium)", "", "(high)", "(xhigh)", "(max)", "(low)"]
# Registry names AA writes differently. The value is AA's name without a variant suffix.
ALIAS = {"hy3": "Hy3", "nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning", "muse-glimmer-30b": "Muse Glimmer",
         "deepseek-v4-flash": "DeepSeek V4 Flash 0420"}


def leaderboard(html):
    """The models array the page embeds in its React flight payload."""
    chunks = [json.loads('"' + m + '"') for m in re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html)]
    text = "".join(chunks)
    at = text.find('{"models":[{"slug"')
    if at < 0:
        sys.exit("aa: the leaderboard page no longer embeds its models; nothing changed")
    return json.JSONDecoder().raw_decode(text[at:])[0]["models"]


def norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def pick(name, rows):
    """The reasoning variant of `name` at its default effort, or None."""
    found = {}
    for m in rows:
        base = re.sub(r"\s*\([^)]*\)\s*$", "", m["name"])
        if norm(base) == norm(name) and m.get("intelligenceIndex") is not None:
            var = re.search(r"\([^)]*\)\s*$", m["name"])
            found.setdefault(var.group(0).lower() if var else "", m)
    return next((found[o] for o in ORDER if o in found), None)


def base_of(key, models):
    k = re.sub(r"-(it-)?awq.*$", "", key)
    return k if k != key and k in models else None


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "local-ai-registry (lab/aa.py)"})
    rows = leaderboard(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    meta = json.loads(MODELS.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    scored, missing = 0, []
    for key, m in meta["models"].items():
        b = base_of(key, meta["models"])
        if b:
            m["aa"] = {"of": b}
            continue
        hit = pick(ALIAS.get(key, m["name"]), rows) or pick(m["name"].replace("-", " "), rows)
        if hit:
            m["aa"] = {"index": round(hit["intelligenceIndex"], 1), "name": hit["name"], "slug": hit["slug"],
                       "estimated": bool(hit.get("intelligenceIndexIsEstimated")), "at": today}
            scored += 1
        else:
            m.pop("aa", None)
            missing.append(key)
    MODELS.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"aa: {scored} models scored from {len(rows)} on the leaderboard; not listed: {', '.join(missing) or 'none'}")


def index(models, key):
    """A model's index, following a build to its base; None when AA does not list it."""
    a = (models.get(key) or {}).get("aa") or {}
    return index(models, a["of"]) if "of" in a else a.get("index")


if __name__ == "__main__":
    main()
