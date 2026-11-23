"""Time a cold 16K-token prefill in 2048-token steps (as mlx_lm.server), with optional fast paths; also save logits."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx, numpy as np
fast = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
if fast:
    from inkling2_mlx_serve import apply_fast_paths; apply_fast_paths(fast)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1]); ids = tok.encode(open("THEORY.md").read() * 200)[:16384]
res = []
for rep in range(3):
    cache = make_prompt_cache(model); t0 = time.perf_counter()
    for i in range(0, len(ids), 2048):
        out = model(mx.array(ids[i:i + 2048])[None], cache=cache); mx.eval(out, [c.state for c in cache])
    res.append(len(ids) / (time.perf_counter() - t0))
np.save(sys.argv[3], np.array(out[0, -1].astype(mx.float32)))
print(f"fast={sorted(fast)} prefill tok/s runs={[round(r) for r in res]}")
