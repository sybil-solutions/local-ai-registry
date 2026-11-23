#!/usr/bin/env python3
"""eqtest.py CFG... — greedy outputs per config (same prompts), saved to eq/<name>.json; prints pairwise agreement."""
import json, sys, os
sys.path.insert(0, "."); import hc
os.makedirs("eq", exist_ok=True)
P = hc.DECODE_PROMPTS + ["Write a short story about a lighthouse keeper who finds a message in a bottle."]
for path in sys.argv[1:]:
    cfg = json.load(open(path))
    w, *_ = hc.launch(cfg)
    outs = [hc.chat(p, 200, temperature=0.0)["text"] for p in P]
    w.stop.set()
    json.dump(outs, open(f"eq/{cfg['name']}.json", "w"))
    print(cfg["name"], "done", flush=True)
names = [json.load(open(p))["name"] for p in sys.argv[1:]]
ref = json.load(open(f"eq/{names[0]}.json"))
for n in names[1:]:
    o = json.load(open(f"eq/{n}.json"))
    for i, (a, b) in enumerate(zip(ref, o)):
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"{names[0]} vs {n} prompt{i}: identical={a == b} common_prefix_chars={k}/{len(a)}")
