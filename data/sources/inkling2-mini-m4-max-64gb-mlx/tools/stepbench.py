"""In-process decode-step benchmark + greedy equivalence for Inkling 2 fast paths.

  python tools/stepbench.py MODEL OUT.json [--fast gates,mask] [--ctx 1024] [--steps 128]

Prefills a fixed deterministic prompt of --ctx tokens, then times --steps greedy decode steps (forward + argmax,
synchronised per token). Saves the greedy tokens and the first step's logits so variants can be compared exactly.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import numpy as np

a = sys.argv[1:]
model_path, out = a[0], a[1]
opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
fast = {s for s in (opt("--fast", "") or "").split(",") if s}
ctx, steps = int(opt("--ctx", 1024)), int(opt("--steps", 128))
if fast:
    from inkling2_mlx_serve import apply_fast_paths
    apply_fast_paths(fast)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(model_path)
text = open(os.path.join(os.path.dirname(__file__), "..", "THEORY.md")).read() * 50
ids = tok.encode(text)[:ctx]
cache = make_prompt_cache(model)
x = mx.array(ids)[None]
CH = int(opt("--chunk", 512))
for i in range(0, len(ids) - 1, CH):  # prefill in chunks, leave the last token for the first step
    chunk = x[:, i:min(i + CH, len(ids) - 1)]
    model(chunk, cache=cache); mx.eval([c.state for c in cache])
y = x[:, -1:]
toks, times, first_logits = [], [], None
for s in range(steps + 16):
    t0 = time.perf_counter()
    logits = model(y, cache=cache)[:, -1]
    y = mx.argmax(logits, axis=-1, keepdims=True)
    mx.eval(y)
    dt = time.perf_counter() - t0
    if s == 0:
        first_logits = np.array(logits.astype(mx.float32))
    toks.append(int(y.item()))
    if s >= 16:
        times.append(dt)
ms = statistics.median(times) * 1e3
res = {"fast": sorted(fast), "ctx": ctx, "ms_per_tok_median": round(ms, 3), "tok_s": round(1e3 / ms, 2),
       "ms_p10": round(np.percentile(times, 10) * 1e3, 3), "ms_p90": round(np.percentile(times, 90) * 1e3, 3),
       "tokens": toks, "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
np.save(out.replace(".json", "-logits.npy"), first_logits)
json.dump(res, open(out, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "tokens"}))
