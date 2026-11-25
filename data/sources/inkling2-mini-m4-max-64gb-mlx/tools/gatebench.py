"""Micro: per-call GPU time of message-gate formulations, 100 independent calls per eval (no sync overhead)."""
import time, mlx.core as mx
D, R = 2304, 64
h = [mx.random.normal((1, 1, 3, D)) for _ in range(100)]; m = [mx.random.normal((1, 1, 3, D)) for _ in range(100)]
wd = mx.random.normal((3, R, 2 * D)).astype(mx.bfloat16) * 0.02; wfm = mx.random.normal((3, 2 * D, R)) * 0.1
mx.eval(h, m, wd, wfm)
def norm_cat(h, m):
    return mx.concatenate([mx.fast.rms_norm(h, None, 1e-6), mx.fast.rms_norm(m, None, 1e-6)], axis=-1)
@mx.compile
def matmul_v(h, m, wd, wfm):
    s = (norm_cat(h, m).astype(wd.dtype)[..., None, :] @ wd.swapaxes(-1, -2)).astype(mx.float32)
    fg = mx.tanh(s @ wfm.swapaxes(-1, -2))[..., 0, :]
    return (1 - fg[..., :D]) * h + fg[..., D:] * m
@mx.compile
def reduce_v(h, m, wd, wfm):
    x = norm_cat(h, m).astype(wd.dtype)                               # (1,1,3,2D)
    s = (x[..., None, :] * wd).astype(mx.float32).sum(-1)             # (1,1,3,R)  multiply-reduce
    fg = mx.tanh((s[..., None, :] * wfm).sum(-1))                     # (1,1,3,2D)
    return (1 - fg[..., :D]) * h + fg[..., D:] * m
wdT = mx.contiguous(wd.swapaxes(-1, -2)); wfmT = mx.contiguous(wfm.swapaxes(-1, -2)); mx.eval(wdT, wfmT)
@mx.compile
def matmulT_v(h, m, wdT, wfmT):  # pre-transposed weights: plain (…,1,K)@(K,N)
    s = (norm_cat(h, m).astype(wdT.dtype)[..., None, :] @ wdT).astype(mx.float32)
    fg = mx.tanh(s @ wfmT)[..., 0, :]
    return (1 - fg[..., :D]) * h + fg[..., D:] * m
def bench(f, *w):
    mx.eval([f(a, b, *w) for a, b in zip(h, m)])
    best = 1e9
    for _ in range(5):
        t = time.perf_counter(); mx.eval([f(a, b, *w) for a, b in zip(h, m)]); best = min(best, time.perf_counter() - t)
    return best / 100 * 1e6
ref = matmul_v(h[0], m[0], wd, wfm)
for name, f, w in (("matmul", matmul_v, (wd, wfm)), ("reduce", reduce_v, (wd, wfm)), ("matmulT", matmulT_v, (wdT, wfmT))):
    out = f(h[0], m[0], *w); print(f"{name:8s} {bench(f, *w):7.1f} us/call (3 gates)  max|d| vs matmul {float(mx.abs(out - ref).max()):.2e}")
