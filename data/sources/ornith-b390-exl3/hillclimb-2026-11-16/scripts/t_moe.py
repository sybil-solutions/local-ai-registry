# MoE prefill DPAS variant check: same random experts/routing, save outputs + timing
import torch, sys, time
lib = sys.argv[1]; tag = sys.argv[2]
torch.ops.load_library(lib); X = torch.ops.exl3xpu_moe
dev = "xpu"; g = torch.Generator().manual_seed(1)
H, I, K, E, M, TOPK = 2048, 512, 4, 256, 4096, 8
blob = int(X.blob_bytes(H, I, K)); nsc = 3 * H + 3 * I
blobs = torch.randint(0, 256, (E, blob), dtype=torch.uint8, generator=g)
signs = (torch.randint(0, 2, (E, nsc), generator=g) * 2 - 1).to(torch.float16)
blobs[:, blob - 2 * nsc:] = signs.view(torch.uint8).view(E, 2 * nsc)
blobs = blobs.to(dev)
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p
ptrs = torch.tensor([s64(blobs[e].data_ptr()) for e in range(E)], dtype=torch.int64, device=dev)
x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
scores = torch.randn(M, E, generator=g)
w, ids = torch.topk(torch.softmax(scores, -1), TOPK, dim=-1)
ids = ids.to(torch.int32).to(dev); w = (w / w.sum(-1, keepdim=True)).float().to(dev)
out = X.moe_forward(x, ids, w, ptrs, I, K, E); torch.xpu.synchronize()
best = 1e9
for r in range(5):
    for _ in range(2): X.moe_forward(x, ids, w, ptrs, I, K, E)
    torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(10): X.moe_forward(x, ids, w, ptrs, I, K, E)
    torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t0) / 10)
torch.save(out.float().cpu(), f"/out/res_{tag}.pt")
fl = 2 * M * TOPK * 3 * H * I
print(f"{tag}: {best*1e3:.2f} ms/layer-call  {fl/best/1e12:.1f} TFLOPS  finite={torch.isfinite(out).all().item()}", flush=True)
