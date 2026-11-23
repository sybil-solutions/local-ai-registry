"""Latency of dependent chains: 100 calls where each output feeds the next input (like the decode step)."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from mlx_lm.models.inkling2 import layers as L
D, R = 2304, 64
g3 = [L.MessageGate(D, R) for _ in range(3)]
for g in g3:
    g.down_proj.weight = (mx.random.normal(g.down_proj.weight.shape) * 0.02).astype(mx.bfloat16)
    g.forget_gate_proj.weight = mx.random.normal(g.forget_gate_proj.weight.shape) * 0.1
    g.message_gate_proj.weight = mx.random.normal(g.message_gate_proj.weight.shape) * 0.1
mx.eval([g.parameters() for g in g3])
def chain(fn, n=100):
    x = mx.random.normal((1, 1, D)); m = mx.random.normal((1, 1, D)); mx.eval(x, m)
    def go():
        y = x
        for _ in range(n): y = fn(y, m)
        return y
    mx.eval(go()); best = 1e9
    for _ in range(5):
        t = time.perf_counter(); mx.eval(go()); best = min(best, time.perf_counter() - t)
    return best / n * 1e6
add = lambda y, m: y + m
print(f"elementwise add      {chain(add):6.1f} us/link")
print(f"reference gate x1    {chain(lambda y, m: g3[0](y, m)):6.1f} us/link")
from inkling2_mlx_serve import apply_fast_paths, apply_kernel_gates
import inkling2_mlx_serve as S
apply_fast_paths({"gates"})
print(f"compiled gate x1     {chain(lambda y, m: g3[0](y, m)):6.1f} us/link")
apply_kernel_gates()
print(f"metal-kernel gate x1 {chain(lambda y, m: g3[0](y, m)):6.1f} us/link")
