"""Min-of-N GPU time of each decode component for one token, x visits per token."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
fast = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
if fast:
    from inkling2_mlx_serve import apply_fast_paths; apply_fast_paths(fast)
from mlx_lm import load
model, tok = load(sys.argv[1]); lm = model.model.language_model
def t(f, n=200):
    for _ in range(10): mx.eval(f())
    best = []
    for _ in range(n):
        s = time.perf_counter(); mx.eval(f()); best.append(time.perf_counter() - s)
    best.sort(); return best[len(best)//10] * 1e3  # p10
x = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16); xf = x.astype(mx.float32)
mx.eval(x, xf)
rows = []
e_nv = t(lambda: lm.layers[6].mlp(x)); e_mx = t(lambda: lm.layers[20].mlp(x))
rows += [("moe nvfp4 (x44)", e_nv, 44), ("moe mxfp8 (x4)", e_mx, 4)]
rows += [("router+experts only nvfp4 (x44)", t(lambda: lm.layers[6].mlp.experts(x, mx.array([[1,5,9,20,33,64,90,127]]))), 44)]
g = lm.layers[6].q_message_gate
rows += [("message gate (x144)", t(lambda: g(xf, xf)), 144)]
rows += [("lm_head (x1)", t(lambda: model.lm_head(x)), 1)]
at = lm.layers[6].self_attn
rows += [("q_proj 2304->3072 (4bit)", t(lambda: at.q_proj(x)), 0)]
rows += [("rms_norm f32 (single op)", t(lambda: mx.fast.rms_norm(xf, None, 1e-6)), 0)]
rows += [("empty eval (x+1)", t(lambda: xf + 1), 0)]
tot = 0
for name, ms, k in rows:
    tot += ms * k; print(f"{name:34s} {ms*1e3:8.1f} us  x{k:<4d} = {ms*k:6.2f} ms")
print(f"sum of counted components: {tot:.2f} ms")
