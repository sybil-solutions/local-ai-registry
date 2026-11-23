"""Split a decode step into Python graph-build time and GPU eval time; count graph nodes."""
import sys, os, time, statistics
sys.path.insert(0, os.getcwd())
import mlx.core as mx
fast = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
if fast:
    from inkling2_mlx_serve import apply_fast_paths; apply_fast_paths(fast)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1]); cache = make_prompt_cache(model)
ids = tok.encode(open("THEORY.md").read() * 20)[:1024]
model(mx.array(ids)[None], cache=cache); mx.eval([c.state for c in cache])
y = mx.array([[ids[-1]]]); b, e = [], []
for s in range(80):
    t0 = time.perf_counter(); out = mx.argmax(model(y, cache=cache)[:, -1], -1, keepdims=True); t1 = time.perf_counter()
    mx.eval(out); t2 = time.perf_counter(); y = out
    if s >= 16: b.append(t1 - t0); e.append(t2 - t1)
# node count of one step's graph
def count(arr):
    seen, stack = set(), [arr]
    while stack:
        a = stack.pop()
        if id(a) in seen: continue
        seen.add(id(a))
    return len(seen)
print(f"fast={sorted(fast)} build_ms={statistics.median(b)*1e3:.2f} eval_ms={statistics.median(e)*1e3:.2f} total={statistics.median([x+z for x,z in zip(b,e)])*1e3:.2f}")
