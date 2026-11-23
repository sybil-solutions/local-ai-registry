import sys, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.fused import rms_norm, mix, route
from exl3xpu.inkling2.ref import rms
dev = "xpu"; torch.manual_seed(0); D = 2304
x = torch.randn(3, D, device=dev) * 3; w = torch.rand(D, device=dev)
print("rms", ((rms_norm(x, w) - rms(x, w)).abs().max() / rms(x, w).abs().max()).item())
for S, NQ, T in ((0, 1, 1), (3, 1, 2), (4, 3, 1), (2, 2, 5)):
    buf = torch.randn(10, T, D, device=dev); idx = torch.tensor([7, 2, 5, 0][:S], dtype=torch.int32, device=dev)
    cur = torch.randn(T, D, device=dev); q = torch.randn(NQ, D, device=dev) * D ** -0.5 * 3
    nw = torch.rand(NQ, D, device=dev)
    states = rms(torch.stack([*[buf[i] for i in idx.tolist()], cur], 1))
    p = torch.einsum("tsd,qd->tqs", states, q).softmax(-1); ref = torch.einsum("tqs,tsd->qtd", p, states)
    o = mix(buf, idx, cur, q); on = mix(buf, idx, cur, q, nw)
    refn = torch.stack([rms(ref[j], nw[j]) for j in range(NQ)])
    print(f"mix S={S} NQ={NQ} T={T}: err {((o-ref).abs().max()/ref.abs().max()).item():.1e} normed {((on-refn).abs().max()/refn.abs().max()).item():.1e}")
lg = torch.randn(3, 128, device=dev); b = torch.randn(128, device=dev) * 0.1
ids, ww = route(lg, b, 8, 8.0)
pr = torch.sigmoid(lg); ti = (pr + b).topk(8, -1).indices; tw = pr.gather(1, ti); tw = tw / tw.sum(-1, keepdim=True) * 8
print("route ids equal", bool((ids.long() == ti).all()), "w err", (ww - tw).abs().max().item())
from exl3xpu.inkling2.fused import route_gemv
xr = torch.randn(3, 2304, device=dev); wr = torch.randn(128, 2304, device=dev) * 0.02
ids2, w2 = route_gemv(xr, wr, b, 8, 8.0)
pr = torch.sigmoid(xr @ wr.T); ti = (pr + b).topk(8, -1).indices; tw = pr.gather(1, ti); tw = tw / tw.sum(-1, keepdim=True) * 8
print("route_gemv ids equal", bool((ids2.long() == ti).all()), "w err", (w2 - tw).abs().max().item())
