"""Triton Inkling2 attention vs dense reference: SWA/global, prefill chunk with prefix, decode, ragged batch, fp8 KV, splits."""
import sys, torch, math
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.attn_triton import attention_ragged, reference_attention
torch.manual_seed(0)
dev = "xpu"; H, Hkv, D, E = 24, 8, 128, 1024
def case(name, reqs, window, fp8=False, split_len=4096):
    # reqs: list of (prefix_len, n_new)
    pool_k, pool_v, locs, poss, starts, qs, qpos, counts, tabs, sinks, refs = [], [], [], [], [0], [], [], [], [], [], []
    base = 0
    for (pre, new) in reqs:
        L = pre + new
        kk = torch.randn(L, Hkv, D, device=dev); kk = kk / kk.norm(dim=-1, keepdim=True) * 20
        vv = torch.randn(L, Hkv, D, device=dev)
        q = torch.randn(new, H, D, device=dev); q = q / q.norm(dim=-1, keepdim=True) * 20
        perm = torch.randperm(L, device=dev) + base          # scattered pool rows
        pool_k.append((perm, kk)); pool_v.append((perm, vv))
        kp = torch.arange(L, device=dev, dtype=torch.int32)
        qp = torch.arange(pre, L, device=dev, dtype=torch.int32)
        tab = torch.randn(new, H, E, device=dev) * 2
        sk = torch.randn(new, H, device=dev)
        locs.append(perm); poss.append(kp); starts.append(starts[-1] + L)
        qs.append(q); qpos.append(qp); counts.append(new); tabs.append(tab); sinks.append(sk)
        kr, vr = kk, vv
        if fp8:
            kr = kk.to(torch.float8_e4m3fn).float(); vr = vv.to(torch.float8_e4m3fn).float()
        refs.append(reference_attention(q, kr, vr, kp, qp, tab, sk, window=window))
        base += L
    N = base
    K = torch.zeros(N, Hkv, D, device=dev); V = torch.zeros(N, Hkv, D, device=dev)
    for (perm, kk), (_, vv) in zip(pool_k, pool_v): K[perm] = kk; V[perm] = vv
    if fp8: K, V = K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
    else: K, V = K.bfloat16(), V.bfloat16()
    out = attention_ragged(torch.cat(qs).bfloat16(), K, V, torch.cat(locs).int(), torch.cat(poss), torch.tensor(starts, device=dev, dtype=torch.int32),
                           torch.cat(qpos), counts, torch.cat(tabs).bfloat16(), torch.cat(sinks), window=window, split_len=split_len)
    ref = torch.cat(refs)
    err = ((out - ref).norm() / ref.norm()).item()
    print(f"{name:34s} rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
case("global prefill 300", [(0, 300)], 0)
case("global chunk prefix 2500 + 700", [(2500, 700)], 0)
case("swa chunk prefix 2500 + 700", [(2500, 700)], 1024)
case("global decode bs3", [(5000, 1), (37, 1), (1200, 1)], 0)
case("swa decode bs3", [(5000, 1), (37, 1), (1200, 1)], 1024)
case("verify bs2 x4", [(3000, 4), (900, 4)], 0)
case("global splits (split 512)", [(4000, 3), (100, 2)], 0, split_len=512)
case("fp8 kv global decode", [(3000, 1), (64, 1)], 0, fp8=True)
