"""Drive the SGLang Inkling2 model class without a server: fake ForwardBatch / attention backend / KV pool.
Checks: prefill (chunked) + decode logits vs the reference forward; two interleaved requests.

  python3 t_inkling2_sgl_offline.py /models/Inkling-2-Mini
"""
import sys, faulthandler, json, types, torch
faulthandler.enable()
sys.path.insert(0, "/opt/trellis-serve/xpu")
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
import exl3xpu.inkling2.sgl_model as SM
from exl3xpu.inkling2.config import Inkling2LoopSglConfig

md = sys.argv[1]
dev = torch.device("xpu")
cfg = Inkling2LoopSglConfig(**json.load(open(f"{md}/config.json")))


class Pool:
    def __init__(self, layers, size, dtype):
        self.k = [torch.zeros(size, 8, 128, device=dev, dtype=dtype) for _ in range(layers)]
        self.v = [torch.zeros(size, 8, 128, device=dev, dtype=dtype) for _ in range(layers)]
    def set_kv_buffer(self, layer, loc, k, v, layer_id_override=None, **kw):
        self.k[layer_id_override][loc] = k.to(self.k[0].dtype); self.v[layer_id_override][loc] = v.to(self.v[0].dtype)
    def get_key_buffer(self, i): return self.k[i]
    def get_value_buffer(self, i): return self.v[i]


SIZE = 16400
pool = Pool(cfg.num_hidden_layers, SIZE, torch.float8_e4m3fn if "--fp8" in sys.argv else torch.bfloat16)
r2t = torch.zeros(8, 16384, dtype=torch.int32, device=dev)
r2t[0] = torch.arange(1, 16385, dtype=torch.int32) % SIZE; r2t[1] = torch.arange(4097, 4097 + 16384, dtype=torch.int32) % SIZE
backend = types.SimpleNamespace(token_to_kv_pool=pool, req_to_token=r2t)
import sglang.srt.model_executor.forward_context as FC
FC.get_attn_backend = lambda: backend


class Mode:
    def __init__(self, kind): self.kind = kind
    def is_idle(self): return False
    def is_extend(self): return self.kind == "extend"
    def is_target_verify(self): return False
    def is_decode(self): return self.kind == "decode"


def fb_extend(slots, pres, ns):
    loc = torch.cat([r2t[s, p:p + n] for s, p, n in zip(slots, pres, ns)]).long()
    pos = torch.cat([torch.arange(p, p + n) for p, n in zip(pres, ns)]).to(dev)
    return types.SimpleNamespace(forward_mode=Mode("extend"), req_pool_indices=torch.tensor(slots, device=dev),
                                 extend_seq_lens_cpu=ns, extend_prefix_lens_cpu=pres, seq_lens_cpu=None, out_cache_loc=loc), pos


def fb_decode(slots, pres):
    loc = torch.stack([r2t[s, p] for s, p in zip(slots, pres)]).long()
    return types.SimpleNamespace(forward_mode=Mode("decode"), req_pool_indices=torch.tensor(slots, device=dev),
                                 seq_lens_cpu=torch.tensor([p + 1 for p in pres]), seq_lens=torch.tensor([p + 1 for p in pres], device=dev), out_cache_loc=loc), torch.tensor(pres, device=dev)


import sglang.srt.layers.logits_processor as LP
LP.LogitsProcessor = lambda c: None
m = SM.Inkling2LoopForConditionalGeneration(cfg)
m.logits_processor = lambda ids, h, head, fb: head.quant_method.apply(head, (h if h.shape[0] <= 512 else h[-1:]).contiguous()).float()
with safe_open(f"{md}/model-00001-of-00001.safetensors", "pt", device="cpu") as f:
    m.load_weights((k, f.get_tensor(k)) for k in f.keys())
print("loaded", flush=True)

tok = PreTrainedTokenizerFast(tokenizer_file=f"{md}/tokenizer.json")
text = open(f"{md}/README.md").read()[2000:6000]
ids = torch.tensor(tok.encode(text)[:300], device=dev)
cap = 30.0
sc = lambda x: cap * torch.tanh(x / cap)

if "--prof-prefill" in sys.argv:
    from torch.profiler import profile, ProfilerActivity
    import time
    n = int(sys.argv[sys.argv.index("--prof-prefill") + 1])
    big = ids.repeat(40)[:2 * n]
    fb, pos = fb_extend([0], [0], [n]); m.forward(big[:n], pos, fb); torch.xpu.synchronize()
    t0 = time.perf_counter(); fb, pos = fb_extend([0], [0], [n]); m.forward(big[:n], pos, fb); torch.xpu.synchronize()
    print(f"prefill chunk {n} (prefix 0): {time.perf_counter() - t0:.2f} s", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        fb, pos = fb_extend([0], [n], [n]); m.forward(big[n:2 * n], pos, fb); torch.xpu.synchronize()
    ka = prof.key_averages()
    kern = [k for k in ka if k.self_device_time_total > 0 and not k.key.startswith(("aten::", "ze", "ur"))]
    print(f"prefill chunk {n} (prefix {n}): kernel time {sum(k.self_device_time_total for k in kern)/1e6:.2f} s, kernels {sum(k.count for k in kern)}")
    for k in sorted(kern, key=lambda k: -k.self_device_time_total)[:25]:
        print(f"{k.self_device_time_total/1e3:9.1f} ms {k.count:6d}x  {k.key[:100]}")
    cpu = sorted([k for k in ka if k.key.startswith("aten::")], key=lambda k: -k.cpu_time_total)[:8]
    for k in cpu: print(f"cpu {k.cpu_time_total/1e3:9.1f} ms {k.count:6d}x {k.key}")
    sys.exit(0)
# request A: prefill 0..199 in two chunks (120 + 80), then decode 200..219 token by token
outs = []
fb, pos = fb_extend([0], [0], [120]); outs.append(m.forward(ids[:120], pos, fb))
fb, pos = fb_extend([0], [120], [80]); outs.append(m.forward(ids[120:200], pos, fb))
if "--graph" in sys.argv:
    # capture one decode step with static buffers, replay for 200..219
    s_ids = torch.zeros(1, dtype=torch.long, device=dev); s_pos = torch.zeros(1, dtype=torch.long, device=dev)
    fb, _ = fb_decode([0], [199])
    s_seq = fb.seq_lens; s_loc = fb.out_cache_loc
    fb.seq_lens, fb.out_cache_loc = s_seq, s_loc
    s_ids.copy_(ids[199:200]); s_pos.fill_(199)
    st = torch.xpu.Stream()
    with torch.xpu.stream(st):
        for _ in range(2): m.forward(s_ids, s_pos, fb)   # warm (writes pos 199 again: same token, harmless)
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        s_out = m.forward(s_ids, s_pos, fb)
    print("captured", flush=True)
    if "--ab" in sys.argv:
        import statistics, time
        import exl3xpu.inkling2.moe_triton as MT
        variants = sys.argv[sys.argv.index("--ab") + 1].split("|")
        graphs = []
        mods = {"SM": SM, "MT": MT}
        for var in variants:
            for kv in filter(None, var.split(";")):   # e.g. SM._FUSED_PREP=1;SM.NSPLIT_SWA=16;MT.GEMV_EXPERT=3,8
                k, v = kv.split("=")
                mod, attr = k.split(".")
                cur = getattr(mods[mod], attr)
                val = tuple(int(x) for x in v.split(",")) if isinstance(cur, tuple) else (v == "1") if isinstance(cur, bool) else type(cur)(v)
                setattr(mods[mod], attr, val)
            # warm (first-use allocations / Triton compiles outside capture)
            for _ in range(2): m.forward(s_ids, s_pos, fb)
            torch.xpu.synchronize()
            gg = torch.xpu.XPUGraph()
            with torch.xpu.graph(gg):
                m.forward(s_ids, s_pos, fb)
            graphs.append(gg)
        res = {i: [] for i in range(len(graphs))}
        for rnd in range(6):
            for i, gg in enumerate(graphs):
                torch.xpu.synchronize(); t0 = time.perf_counter()
                for _ in range(20): gg.replay()
                torch.xpu.synchronize(); res[i].append((time.perf_counter() - t0) / 20 * 1e3)
        for i, var in enumerate(variants):
            print(f"AB [{var}]: median {statistics.median(res[i]):.1f} ms  all {[round(x,1) for x in res[i]]}", flush=True)
        sys.exit(0)
    for i in range(200, 220):
        s_ids.copy_(ids[i:i + 1]); s_pos.fill_(i); s_seq.fill_(i + 1); s_loc.copy_(r2t[0, i:i + 1].long())
        g.replay(); outs.append(s_out.clone())
    torch.xpu.synchronize()
    import time; t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.xpu.synchronize(); print(f"graph decode step {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms", flush=True)
elif "--prof" in sys.argv:
  for i in range(200, 220):
    fb, pos = fb_decode([0], [i]); outs.append(m.forward(ids[i:i + 1], pos, fb))
  from torch.profiler import profile, ProfilerActivity
  torch.xpu.synchronize()
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    for _ in range(3):
      fb, pos = fb_decode([0], [219]); m.forward(ids[219:220], pos, fb)
    torch.xpu.synchronize()
  ka = prof.key_averages()
  rows = sorted([k for k in ka if k.device_time_total > 0], key=lambda k: -k.device_time_total)
  tot = sum(k.self_device_time_total for k in ka) / 3
  print(f"device time per step {tot/1e3:.1f} ms, kernels/step {sum(k.count for k in rows if k.self_device_time_total>0)/3:.0f}")
  kern = [k for k in ka if k.self_device_time_total > 0 and not k.key.startswith("aten::")]
  print(f"device kernels/step {sum(k.count for k in kern)/3:.0f}, kernel time/step {sum(k.self_device_time_total for k in kern)/3/1e3:.1f} ms")
  for k in sorted(kern, key=lambda k: -k.count)[:40]:
    print(f"{k.count/3:6.0f}x {k.self_device_time_total/3/1e3:7.2f} ms  {k.key[:110]}")
else:
  for i in range(200, 220):
    fb, pos = fb_decode([0], [i]); outs.append(m.forward(ids[i:i + 1], pos, fb))
  import time; torch.xpu.synchronize(); t0 = time.perf_counter()
  for _ in range(5):
    fb, pos = fb_decode([0], [219]); m.forward(ids[219:220], pos, fb)
  torch.xpu.synchronize(); print(f"eager decode step {(time.perf_counter() - t0) / 5 * 1e3:.1f} ms", flush=True)
logits = sc(torch.cat(outs))[:, :200058]
print("forward ok", logits.shape, flush=True)

from exl3xpu.inkling2.ref import Inkling2Ref
del m; torch.xpu.empty_cache()
ref = Inkling2Ref(md)
rl = ref.forward(ids[:220])
def kl(a, b):
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)
k = kl(rl.float(), logits.float())
print(f"KL vs ref: prefill max {k[:200].max().item():.2e} mean {k[:200].mean().item():.2e} | decode max {k[200:].max().item():.2e} mean {k[200:].mean().item():.2e}")
print(f"argmax agree {(rl.argmax(-1) == logits.argmax(-1)).float().mean().item():.3f}")
nll = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:220]); print(f"ppl sgl {nll.exp().item():.3f}")
