import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.attn_triton import attention_prefill_esimd, reference_attention
dev = "xpu"; H, Hkv, D, E = 24, 8, 128, 1024; torch.manual_seed(0)
def mk(n, pre, window):
    L = pre + n
    k = torch.randn(L, Hkv, D, device=dev); k = (k / k.norm(dim=-1, keepdim=True) * 20).bfloat16(); v = torch.randn(L, Hkv, D, device=dev).bfloat16()
    q = torch.randn(n, H, D, device=dev); q = (q / q.norm(dim=-1, keepdim=True) * 20).bfloat16()
    kpos = torch.arange(L, device=dev, dtype=torch.int32); qpos = torch.arange(pre, L, device=dev, dtype=torch.int32)
    if window:  # SWA scratch: only the last window-1 prefix keys + chunk
        keep = max(0, pre - (window - 1)); k, v, kpos = k[keep:], v[keep:], kpos[keep:]
    table = (torch.randn(n, H, E, device=dev) * 2).bfloat16(); sink = torch.randn(n, H, device=dev)
    return q, k, v, qpos, kpos, table, sink
for (n, pre, w) in ((300, 0, 0), (700, 2500, 0), (700, 2500, 1024), (1000, 0, 1024)):
    q, k, v, qp, kp, tb, sk = mk(n, pre, w)
    o = attention_prefill_esimd(q, k, v, int(qp[0]), int(kp[0]), tb, sk, window=w)
    r = reference_attention(q, k, v, kp, qp, tb, sk, window=w)
    print(f"n={n} pre={pre} w={w}: rel err {((o-r).norm()/r.norm()).item():.2e}", flush=True)
for (n, pre, w) in ((4096, 4096, 0), (4096, 4096, 1024), (4096, 28672, 0), (4096, 126976, 0)):
    q, k, v, qp, kp, tb, sk = mk(n, pre, w)
    for qtw in (1, 4, 8, 16):
        f = lambda: attention_prefill_esimd(q, k, v, int(qp[0]), int(kp[0]), tb, sk, window=w, qtw=qtw)
        f(); torch.xpu.synchronize(); t0 = time.perf_counter()
        for _ in range(3): f()
        torch.xpu.synchronize(); dt = (time.perf_counter() - t0) / 3
        pairs = n * (min(pre, w - 1) if w else pre) + n * n / 2
        print(f"ESIMD n={n} pre={pre} w={w} qtw={qtw}: {dt*1e3:8.1f} ms  {pairs*H*4*D/dt/1e12:5.2f} TFLOPS", flush=True)
