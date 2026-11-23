"""Triton NVFP4/MXFP8 MoE vs dequant reference (ref.py), decode/verify/prefill shapes; plus timing."""
import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.moe_triton import Inkling2ExpertsSYCL as Inkling2ExpertsXPU
from exl3xpu.inkling2.ref import nvfp4_dequant, mxfp8_dequant
torch.manual_seed(0); dev = "xpu"
E, H, I = 128, 2304, 1152
def mk(nv):
    if nv:
        w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
        s13 = torch.randint(0x30, 0x40, (E, 2 * I, H // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
        s2 = torch.randint(0x30, 0x40, (E, H, I // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        a13 = torch.rand(E, device=dev) * 0.02 + 0.005; a2 = torch.rand(E, device=dev) * 0.02 + 0.005
        return w13, s13, w2, s2, a13, a2
    w13 = (torch.randn(E, 2 * I, H, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s13 = torch.randint(118, 122, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    w2 = (torch.randn(E, H, I, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s2 = torch.randint(118, 122, (E, H, I // 32), dtype=torch.uint8, device=dev)
    return w13, s13, w2, s2, None, None
def ref(x, ids, tw, t):
    w13, s13, w2, s2, a13, a2 = t
    out = torch.zeros(x.shape[0], H, device=dev)
    for ti in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = ids[ti, j].item()
            W13 = nvfp4_dequant(w13[e], s13[e], a13[e]) if a13 is not None else mxfp8_dequant(w13[e], s13[e])
            W2 = nvfp4_dequant(w2[e], s2[e], a2[e]) if a2 is not None else mxfp8_dequant(w2[e], s2[e])
            h = (x[ti:ti+1].bfloat16().float() @ W13.float().T)
            g, u = h[:, 0::2], h[:, 1::2]
            act = (4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)).bfloat16().float()
            out[ti] += (act @ W2.float().T)[0] * tw[ti, j]
    return out
for nv in (True, False):
    t = mk(nv)
    m = Inkling2ExpertsXPU(*t[:4], *t[4:]) if nv else Inkling2ExpertsXPU(*t[:4])
    for T in (1, 4, 8, 40):
        x = torch.randn(T, H, device=dev)
        ids = torch.stack([torch.randperm(E, device=dev)[:8] for _ in range(T)])
        if T == 4: ids[1] = ids[0]   # shared experts across verify rows
        tw = torch.rand(T, 8, device=dev)
        y = m(x, ids, tw); r = ref(x, ids, tw, t)
        err = ((y - r).norm() / r.norm()).item()
        print(f"{'nvfp4' if nv else 'mxfp8'} T={T:3d} rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
    # timing: decode T=1 and prefill T=4096
    for T in (1, 2, 4, 4096):
        x = torch.randn(T, H, device=dev); ids = torch.randint(0, E, (T, 8), device=dev); tw = torch.rand(T, 8, device=dev)
        for _ in range(3): m(x, ids, tw)
        torch.xpu.synchronize(); n = 20 if T < 100 else 3; t0 = time.perf_counter()
        for _ in range(n): m(x, ids, tw)
        torch.xpu.synchronize(); dt = (time.perf_counter() - t0) / n
        uniq = len(ids.unique())
        byts = uniq * (m.w13[0].numel() + m.s13[0].numel() + m.w2[0].numel() + m.s2[0].numel())
        print(f"  {'nvfp4' if nv else 'mxfp8'} T={T:4d}: {dt*1e3:7.3f} ms  {byts/dt/1e9:6.1f} GB/s  {2*T*8*3*H*I/dt/1e12:5.2f} TFLOPS", flush=True)
