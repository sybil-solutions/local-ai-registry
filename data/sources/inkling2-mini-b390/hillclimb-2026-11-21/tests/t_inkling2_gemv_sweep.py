"""Sweep moe_gemv2 variants (RP rows in flight, GPL groups/lane, NSG sub-groups, RPS rows/sub-group) on cache-cold
weights: Q8 dense (attention-like shapes, 48 distinct), NVFP4 experts (8 of 128, 44 distinct layers). Checks vs v1."""
import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.moe_triton import ops, moe_align
from exl3xpu.inkling2.dense import quant_q8
o = ops(); dev = "xpu"; torch.manual_seed(0)
def tm(f, n=5):
    f(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t0) / n
VARS = {0: "RP1 GPL1 NSG16", 1: "RP2 GPL1 NSG16", 2: "RP2 GPL2 NSG16", 3: "RP4 GPL1 NSG16", 4: "RP2 GPL1 NSG8", 5: "RP1 GPL2 NSG16", 6: "RP2 GPL1 NSG32"}
# dense Q8: rotate over 48 layer-like matrices [8600, 2304]
mats = [quant_q8(torch.randn(8600, 2304, device=dev, dtype=torch.bfloat16)) for _ in range(24)]
x = torch.randn(1, 2304, device=dev, dtype=torch.bfloat16)
sid = torch.tensor([0], dtype=torch.int32, device=dev); eob = torch.zeros(1, dtype=torch.int32, device=dev)
out = torch.empty(1, 8600, device=dev)
ref = torch.empty_like(out)
o.moe_gemv(x, mats[0][0], mats[0][1], None, ref, sid, eob, None, 1, 1, 1)
byts = sum(m[0].numel() + m[1].numel() * 2 for m in mats)
t1 = tm(lambda: [o.moe_gemv(x, q, s, None, out, sid, eob, None, 1, 1, 1) for q, s in mats])
print(f"Q8 dense v1: {byts/t1/1e9:6.1f} GB/s", flush=True)
for v, name in VARS.items():
    for rps in (4, 8):
        o.moe_gemv2(x, mats[0][0], mats[0][1], None, out, sid, eob, None, 1, 1, 1, v, rps)
        err = ((out - ref).norm() / ref.norm()).item()
        t = tm(lambda: [o.moe_gemv2(x, q, s, None, out, sid, eob, None, 1, 1, 1, v, rps) for q, s in mats])
        print(f"Q8 dense v2 {name} RPS{rps}: {byts/t/1e9:6.1f} GB/s  err {err:.1e}", flush=True)
del mats
# NVFP4 experts: 12 distinct layers [128, 2304, 1152], 8 experts each, M=1
E = 128
layers = [(torch.randint(0, 256, (E, 2304, 1152), dtype=torch.uint8, device=dev), torch.randint(0x30, 0x40, (E, 2304, 144), dtype=torch.uint8, device=dev)) for _ in range(6)]
alpha = torch.rand(E, device=dev)
ids = torch.randperm(E, device=dev)[:8][None].int()
sid8, eob8 = moe_align(ids, E, 1)
h = torch.empty(8, 2304, device=dev); href = torch.empty_like(h)
o.moe_gemv(x, layers[0][0], layers[0][1], alpha, href, sid8, eob8, None, 8, 8, 1)
byts = 6 * 8 * (2304 * 1152 + 2304 * 144)
t1 = tm(lambda: [o.moe_gemv(x, w, s, alpha, h, sid8, eob8, None, 8, 8, 1) for w, s in layers])
print(f"NVFP4 v1: {byts/t1/1e9:6.1f} GB/s", flush=True)
for v, name in VARS.items():
    for rps in (4, 8):
        o.moe_gemv2(x, layers[0][0], layers[0][1], alpha, h, sid8, eob8, None, 8, 8, 1, v, rps)
        err = ((h - href).norm() / href.norm()).item()
        t = tm(lambda: [o.moe_gemv2(x, w, s, alpha, h, sid8, eob8, None, 8, 8, 1, v, rps) for w, s in layers])
        print(f"NVFP4 v2 {name} RPS{rps}: {byts/t/1e9:6.1f} GB/s  err {err:.1e}", flush=True)
