"""Q8 dense GEMV vs bf16 reference; bandwidth on distinct weights (cache-cold rotation)."""
import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from exl3xpu.inkling2.dense import Q8Linear
dev = "xpu"; torch.manual_seed(0)
for (N, K, nrot) in ((3072, 2304, 48), (1024, 2304, 48), (2304, 3072, 48), (201024, 2304, 1)):
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nrot)]
    ls = [Q8Linear(w, keep_bf16=False) for w in ws]
    for M in (1, 4):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x.float(), ws[0].float())
        y = ls[0](x)
        err = ((y - ref).norm() / ref.norm()).item()
        for _ in range(3):
            for l in ls: l(x)
        torch.xpu.synchronize(); t0 = time.perf_counter(); reps = 5 if nrot > 1 else 20
        for _ in range(reps):
            for l in ls: l(x)
        torch.xpu.synchronize(); dt = (time.perf_counter() - t0) / (reps * nrot)
        print(f"N{N} K{K} M{M}: rel err {err:.1e}  {dt*1e6:7.1f} us  {N*K*1.0625/dt/1e9:6.1f} GB/s (q8 bytes, {nrot} distinct)", flush=True)
