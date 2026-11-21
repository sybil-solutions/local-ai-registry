"""Cost of one forward with L new tokens (verify-style) vs L=1, at 1K context, best fast paths."""
import sys, os, time, statistics
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from inkling2_mlx_serve import apply_fast_paths
apply_fast_paths({"gates", "mask", "moe", "conv", "attn", "gates3"})
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1]); ids = tok.encode(open("THEORY.md").read() * 20)[:1024]
for Lk in (1, 2, 3, 4, 6, 8):
    cache = make_prompt_cache(model); model(mx.array(ids)[None], cache=cache); mx.eval([c.state for c in cache])
    ts = []
    for s in range(40):
        x = mx.array([ids[(s * Lk + j) % 1000] for j in range(Lk)])[None]
        t = time.perf_counter(); mx.eval(model(x, cache=cache)); ts.append(time.perf_counter() - t)
    ms = statistics.median(ts[8:]) * 1e3
    print(f"L={Lk}: {ms:6.2f} ms/forward  -> {ms/Lk:6.2f} ms per token if all accepted")
