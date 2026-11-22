#!/usr/bin/env python3
"""Score a running server on the hill-climb's quality subset: eval_general.py's own items, request and grading
(MATH-500 150, MMLU-Pro 210, IFEval 200; seed 4242, reasoning effort medium, 16,000-token limit), 64 at a time.
Comparable with eval/.../inkling-2-mini/hillclimb/quality.jsonl.

    quality_eval.py <endpoint> <model> <label> <out dir>
"""
import collections, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VENV = ".venv-eval/bin/python"  # IFEval's checker needs absl and langdetect
try:
    import absl  # noqa: F401
except ImportError:
    os.execv(VENV, [VENV, *sys.argv])

sys.path.insert(0, ".")  # the folder with eval_general.py
import eval_general as eg  # noqa: E402

url, model, label, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
its = [it for it in eg.items() if it["set"] in ("math500", "mmlupro", "ifeval")]
t0 = time.monotonic()
with ThreadPoolExecutor(64) as ex:
    outs = list(ex.map(lambda it: (it, eg.call(url, model, it)), its))
wall = time.monotonic() - t0
rows, by, capped, toks = [], collections.defaultdict(list), 0, 0
for it, o in outs:
    g = eg.grade(it, o["content"])
    ok = g.get("correct", g.get("strict"))
    by[it["set"]].append(ok)
    rows.append({"iid": it["iid"], "set": it["set"], "ok": ok, "finish": o["finish"],
                 "completion_tokens": (o.get("usage") or {}).get("completion_tokens")})
    capped += o["finish"] == "length"; toks += (o.get("usage") or {}).get("completion_tokens", 0)
(out / f"{label}-items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
res = {"label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "scores": {s: round(100 * statistics.mean(v), 1) for s, v in sorted(by.items())},
       "n": {s: len(v) for s, v in by.items()}, "capped": capped, "completion_tokens": toks, "wall_s": round(wall, 1)}
with open(out / "quality.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
