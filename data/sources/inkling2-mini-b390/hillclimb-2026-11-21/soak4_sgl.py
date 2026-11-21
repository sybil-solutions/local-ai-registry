#!/usr/bin/env python3
"""Fill the 1M-token KV pool: 4 concurrent ~250K-token prompts with retrieval markers; report per-stream TTFT/decode/retrieval + RAM."""
import json, os, sys, time
if os.path.exists("SKIP_SOAK"):
    print("soak skipped (SKIP_SOAK flag)"); sys.exit(0)
sys.path.insert(0, "."); import hc_sgl as hc
from concurrent.futures import ThreadPoolExecutor
cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/final-best.json")); cfg["name"] = "soak4x256k"
watch, base, *_ = hc.launch(cfg)
docs = [hc.doc(250000, 100 + i) for i in range(4)]
t0 = time.monotonic()
def go(i):
    text, mk = docs[i]
    r = hc.chat(text, 128, temperature=0.0, timeout=7200)
    return {"stream": i, "prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 1), "decode": r["decode"] and round(r["decode"], 1),
            "retrieval": sum(m in r["text"] for m in mk)}
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(go, range(4)))
wall = time.monotonic() - t0
out = {"name": "soak4x256k", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "streams": res, "wall_s": round(wall, 1),
       "total_prompt_tokens": sum(r["prompt_tokens"] for r in res), "agg_prefill_tok_s": round(sum(r["prompt_tokens"] for r in res) / wall, 1),
       "ram_peak_gib": round(watch.peak / 2**30, 2), "ram_cap_ok": not watch.tripped, "on_ac": hc.on_ac()}
watch.stop.set()
open(sys.argv[2] if len(sys.argv) > 2 else "soak4.json", "w").write(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
