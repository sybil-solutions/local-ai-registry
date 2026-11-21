"""SYCL experts on the real checkpoint tensors (layer 0 NVFP4, layer 20 MXFP8) vs ref.experts math."""
import sys, faulthandler, torch
faulthandler.enable()
sys.path.insert(0, "/opt/trellis-serve/xpu")
from safetensors import safe_open
from exl3xpu.inkling2.moe_triton import Inkling2ExpertsSYCL
from exl3xpu.inkling2.ref import nvfp4_dequant, mxfp8_dequant
f = safe_open("/models/Inkling-2-Mini/model-00001-of-00001.safetensors", "pt", device="cpu")
for il in (0, 20):
    e = f"model.llm.layers.{il}.mlp.experts."
    g = lambda n: f.get_tensor(e + n).to("xpu")
    nv = (e + "w13_weight_amax") in f.keys()
    print(il, "nvfp4" if nv else "mxfp8", {n: (f.get_tensor(e+n).dtype, tuple(f.get_tensor(e+n).shape), f.get_tensor(e+n).is_contiguous()) for n in ("w13_weight", "w13_weight_scale")}, flush=True)
    if nv:
        a13 = g("w13_weight_amax") / g("w13_weight_outer_scale_denominator"); a2 = g("w2_weight_amax") / g("w2_weight_outer_scale_denominator")
        m = Inkling2ExpertsSYCL(g("w13_weight"), g("w13_weight_scale"), g("w2_weight"), g("w2_weight_scale"), a13, a2)
    else:
        m = Inkling2ExpertsSYCL(g("w13_weight"), g("w13_weight_scale"), g("w2_weight"), g("w2_weight_scale"))
    x = torch.randn(40, 2304, device="xpu") * 0.5
    ids = torch.stack([torch.randperm(128, device="xpu")[:8] for _ in range(40)]).int(); tw = torch.rand(40, 8, device="xpu")
    print(" dequant call...", flush=True)
    y = m.prefill(x, ids, tw); torch.xpu.synchronize(); print(" prefill ok", y.abs().mean().item(), flush=True)
    yd = torch.cat([m.decode(x[i:i+1], ids[i:i+1], tw[i:i+1]) for i in range(4)]); torch.xpu.synchronize()
    print(" decode vs prefill rel", ((yd - y[:4]).norm() / y[:4].norm()).item(), flush=True)
