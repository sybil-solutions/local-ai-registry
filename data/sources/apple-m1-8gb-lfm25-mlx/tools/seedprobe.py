#!/usr/bin/env python3
"""Does mlx_lm's sampler honour mx.random.seed in a worker thread? (task tool; runs on the task runtime, loads no model)

  PYTHONPATH=rt312/site rt312/python/bin/python3.12 -s seedprobe.py OUT.json

mlx_lm.server samples in its generation thread, not the main one. This draws 24 tokens from a uniform 4-way distribution
with mlx_lm's own make_sampler (temperature 1.0; its categorical sampler is mx.compile'd with mx.random.state) and with
plain mx.random.categorical, for seeds 7, 8, 7 and twice unseeded, first on the main thread and then on a worker thread.
"""
import json, sys, threading
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

logits = mx.log(mx.array([[0.25, 0.25, 0.25, 0.25]]))
SEEDS = [7, 8, 7, None, None]


def compiled(seed, n=24):
    if seed is not None:
        mx.random.seed(seed)
    s = make_sampler(1.0, top_p=1.0, top_k=0, min_p=0.0)  # the server's defaults
    return "".join(str(s(logits).item()) for _ in range(n))


def plain(seed, n=24):
    if seed is not None:
        mx.random.seed(seed)
    return "".join(str(mx.random.categorical(logits).item()) for _ in range(n))


res = {"mlx": mx.__version__, "seeds": SEEDS, "main": {"compiled": [compiled(s) for s in SEEDS], "plain": [plain(s) for s in SEEDS]}}
t = threading.Thread(target=lambda: res.__setitem__("thread", {"compiled": [compiled(s) for s in SEEDS], "plain": [plain(s) for s in SEEDS]}))
t.start(); t.join()
for where in ("main", "thread"):
    for kind in ("compiled", "plain"):
        d = res[where][kind]
        res.setdefault("findings", {})[f"{where} {kind}"] = {
            "seed 7 repeats": d[0] == d[2], "seed 7 != seed 8": d[0] != d[1], "unseeded draws differ": d[3] != d[4],
            "draws vary within a sequence": all(len(set(x)) > 1 for x in d)}
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k, v in res["findings"].items():
    print(f"{k:16s} {v}")
