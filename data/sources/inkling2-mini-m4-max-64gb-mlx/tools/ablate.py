"""PROFILING ONLY (changes the model output): decode-step time with one component replaced by a cheap stand-in."""
import sys, os, time, statistics
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from inkling2_mlx_serve import apply_fast_paths
apply_fast_paths({"gates", "mask", "moe", "conv", "attn", "gates3"})
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import layers as L
mode = sys.argv[2]
if mode == "no_moe":
    L.SparseMoeBlock.__call__ = lambda self, h: h * 0.5
elif mode == "no_experts":  # router kept, experts skipped
    orig = L.SparseMoeBlock.__call__
    L.SparseMoeBlock.__call__ = lambda self, h: (self.gate(h)[1].sum(-1, keepdims=True).astype(h.dtype) * h)
elif mode == "no_attn":
    L.SlidingAttention.__call__ = lambda self, h, k, v, **kw: h
    L.FullAttention.__call__ = lambda self, h, k, v, **kw: h
elif mode == "no_gates":
    L.DecoderLayer._mix_messages = lambda self, states, m: states
    L.MessageGate.__call__ = lambda self, h, m: h
model, tok = load(sys.argv[1]); cache = make_prompt_cache(model)
ids = tok.encode(open("THEORY.md").read() * 20)[:1024]
model(mx.array(ids)[None], cache=cache); mx.eval([c.state for c in cache])
y = mx.array([[ids[-1]]]); ts = []
for s in range(100):
    t0 = time.perf_counter(); y = mx.argmax(model(y, cache=cache)[:, -1], -1, keepdims=True); mx.eval(y)
    if s >= 20: ts.append(time.perf_counter() - t0)
print(f"{mode:12s} {statistics.median(ts)*1e3:6.2f} ms/step")
