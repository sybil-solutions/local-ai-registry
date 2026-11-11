#!/usr/bin/env python3
"""The sweep's tables in the standard form: one per weight set, medians over the measured waves."""
import json, statistics, sys
KV = "262k (~3.0 GiB est. when full)"
def rows(q):
    out = {}
    for part in ("c1", "cn"):
        for r in json.load(open(f"runs/sweep-{q}-{part}/result.json"))["rows"]:
            out[(r["context"], r["concurrency"])] = r
    return out
def cell(r):
    waves = [w for w in r["waves"] if not w["warmup"] and "REQ_FAIL" not in w["flags"]]
    med = lambda xs: statistics.median(xs) if xs else None
    pre = med([w["prefill_tps"] for w in waves if w.get("prefill_tps")])
    tot = med([w["decode_total_tps"] for w in waves if w.get("decode_total_tps")])
    n = r["concurrency"]
    prefill = f"{pre:,.0f} tok/s" if pre else "–"
    if n == 1:
        decode = f"{tot:.1f} tok/s" if tot else "not run"
    elif tot:
        alone = sum(1 for w in waves if not w.get("decode_total_tps"))
        decode = f"{tot:.1f} tok/s ({tot / n:.1f}/stream)" + (f" ({alone} of {len(waves)} waves: streams ran one at a time)" if alone else "")
    else:
        decode = "not run (streams ran one at a time: each answer ended before the other prompt was read)"
    return prefill, decode
def table(q, title):
    rs = rows(q)
    print(f"**{title}**\n")
    print("| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |\n|---|---|---|---|---|---|")
    for key in (("8k", 1), ("8k", 2), ("8k", 4), ("32k", 1), ("32k", 2)):
        p, d = cell(rs[key])
        print(f"| {key[0]} | {p} | {d} | {key[1]} | {KV} | 1 |")
    z = cell(rs[("0", 1)])[1]
    peak = max(r["peak_memory_gb"] or 0 for r in rs.values())
    return z, peak
if __name__ == "__main__":
    k = json.load(open("runs/kld/result.json"))["candidates"]
    for q, name in (("8bit", "8-bit"), ("bf16", "bf16 (unquantized)"), ("4bit", "4-bit")):
        z, peak = table(q, f"LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX {name} · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once")
        c = k.get(q)
        quality = "the unquantized weights (the reference)" if not c else f"KL to bf16 {c['value']:.4f} nats (95% CI {c['ci95_low']:.4f}–{c['ci95_high']:.4f}), top-token agreement {c['top1_next_token_agreement']:.1%}"
        print(f"\nNo context, 1 stream: {z}. Quality: {quality}. Peak GPU memory in these rows {peak * 1e9 / 2**30:.2f} GiB.\n")
