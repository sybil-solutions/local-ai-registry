"""SYCL decode attention vs dense reference (ring/paged, bf16/fp8, nq 1/4), and bandwidth."""
import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.attn_triton import attention_decode_sycl, reference_attention
dev = "xpu"; H, Hkv, D, E = 24, 8, 128, 1024; torch.manual_seed(0)
def tm(f, n=30):
    for _ in range(3): f()
    torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t0) / n * 1e3
# correctness: ring (SWA, window 1024) with 2 requests in slots 3 and 1, positions wrapped
R = 1088; S = 4
for nq in (1, 4):
    kb = torch.randn(S * R, Hkv, D, device=dev) ; kb = (kb / kb.norm(dim=-1, keepdim=True) * 20).bfloat16(); vb = torch.randn(S * R, Hkv, D, device=dev).bfloat16()
    rpos = torch.full((S, R), -1, dtype=torch.int32, device=dev)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=dev); lasts = [2500, 700]
    for s, last in zip(slots.tolist(), lasts):
        p = torch.arange(max(0, last - R + 1), last + 1, device=dev); rpos[s, p % R] = p.int()
    qpos = torch.tensor(sum([[last - nq + 1 + i for i in range(nq)] for last in lasts], []), dtype=torch.int32, device=dev)
    q = torch.randn(2 * nq, H, D, device=dev); q = (q / q.norm(dim=-1, keepdim=True) * 20).bfloat16()
    rr = torch.randn(2 * nq, H, 16, device=dev); proj = torch.randn(16, E, device=dev) * 0.5; table = torch.einsum('thr,re->the', rr, proj); sink = torch.randn(2 * nq, H, device=dev)
    out = attention_decode_sycl(q, kb, vb, slots, qpos, rr, proj, sink, n_q=nq, paged=False, ring_pos=rpos, ring=R, window=1024, nsplit=8)
    refs = []
    for bi, s in enumerate(slots.tolist()):
        rows = slice(bi * nq, bi * nq + nq)
        refs.append(reference_attention(q[rows], kb[s * R:(s + 1) * R], vb[s * R:(s + 1) * R], rpos[s], qpos[rows], table[rows], sink[rows], window=1024))
    ref = torch.cat(refs); err = ((out - ref).norm() / ref.norm()).item()
    print(f"ring nq={nq}: rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
# paged fp8 global
for L in (300, 5000):
    pool = torch.randn(L + 64, Hkv, D, device=dev); pool = pool / pool.norm(dim=-1, keepdim=True) * 20
    poolv = torch.randn(L + 64, Hkv, D, device=dev)
    pk, pv = pool.to(torch.float8_e4m3fn), poolv.to(torch.float8_e4m3fn)
    r2t = torch.zeros(4, 8192, dtype=torch.int32, device=dev); perm = torch.randperm(L + 64, device=dev)[:L].int(); r2t[2, :L] = perm
    slots = torch.tensor([2], dtype=torch.int32, device=dev); kvlen = torch.tensor([L], dtype=torch.int32, device=dev)
    qpos = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    q = torch.randn(1, H, D, device=dev); q = (q / q.norm(dim=-1, keepdim=True) * 20).bfloat16()
    rr = torch.randn(1, H, 16, device=dev); proj = torch.randn(16, E, device=dev) * 0.5; table = torch.einsum('thr,re->the', rr, proj); sink = torch.randn(1, H, device=dev)
    out = attention_decode_sycl(q, pk, pv, slots, qpos, rr, proj, sink, n_q=1, paged=True, r2t=r2t, kv_len=kvlen, nsplit=16)
    ref = reference_attention(q, pk[perm.long()].float(), pv[perm.long()].float(), torch.arange(L, device=dev), qpos, table, sink)
    err = ((out - ref).norm() / ref.norm()).item(); print(f"paged fp8 L={L}: rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
# bandwidth
q1 = torch.randn(1, H, D, device=dev).bfloat16(); qp = lambda L: torch.tensor([L - 1], dtype=torch.int32, device=dev)
tab = torch.randn(1, H, 16, device=dev); prj = torch.randn(16, E, device=dev); snk = torch.randn(1, H, device=dev); sl = torch.tensor([0], dtype=torch.int32, device=dev)
for L in (4096, 65536, 262144):
    pool = torch.randn(L + 16, Hkv, D, device=dev).to(torch.float8_e4m3fn)
    r2t = torch.arange(L, device=dev, dtype=torch.int32)[None].contiguous(); kvl = torch.tensor([L], dtype=torch.int32, device=dev)
    for ns in (16, 64, 128):
        t = tm(lambda: attention_decode_sycl(q1, pool, pool, sl, qp(L), tab, prj, snk, n_q=1, paged=True, r2t=r2t, kv_len=kvl, nsplit=ns))
        print(f"paged fp8 L={L:6d} nsplit {ns:3d}: {t:7.3f} ms  {2*L*Hkv*D/t/1e6:6.1f} GB/s", flush=True)
kb = torch.randn(R, Hkv, D, device=dev).bfloat16(); rp = torch.arange(R, device=dev, dtype=torch.int32)[None].contiguous()
for ns in (4, 8, 16):
    t = tm(lambda: attention_decode_sycl(q1, kb, kb, sl, torch.tensor([R - 1], dtype=torch.int32, device=dev), tab, prj, snk, n_q=1, paged=False, ring_pos=rp, ring=R, window=1024, nsplit=ns))
    print(f"ring bf16 R={R} nsplit {ns}: {t:.3f} ms", flush=True)
