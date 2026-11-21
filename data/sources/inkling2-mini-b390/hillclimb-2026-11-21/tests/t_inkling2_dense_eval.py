"""Quality of the decode dense format: score N tokens with all dense linears on the GEMV path (Q8 or FP4) vs bf16.
  python3 t_inkling2_dense_eval.py MODEL_DIR OUT.pt [--gemv]   (env INKLING2_DENSE_FP4=0/1)
Saves log-probs (fp16) to OUT.pt; if REF.pt is given as 3rd arg, prints KLD / top-1 agreement / ppl vs it."""
import sys, json, types, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
import sglang.srt.layers.logits_processor as LP
LP.LogitsProcessor = lambda c: None
import exl3xpu.inkling2.sgl_model as SM
from exl3xpu.inkling2.config import Inkling2LoopSglConfig
from exl3xpu.inkling2.dense import Q8Linear
md, out = sys.argv[1], sys.argv[2]
ref_path = next((a for a in sys.argv[3:] if a.endswith(".pt")), None)
dev = torch.device("xpu")
cfg = Inkling2LoopSglConfig(**json.load(open(f"{md}/config.json")))
N = 1024
class Pool:
    def __init__(s, L, size):
        s.k = [torch.zeros(size, 8, 128, device=dev, dtype=torch.float8_e4m3fn) for _ in range(L)]; s.v = [torch.zeros_like(x) for x in s.k]
    def set_kv_buffer(s, layer, loc, k, v, layer_id_override=None, **kw):
        s.k[layer_id_override][loc] = k.to(s.k[0].dtype); s.v[layer_id_override][loc] = v.to(s.v[0].dtype)
    def get_key_buffer(s, i): return s.k[i]
    def get_value_buffer(s, i): return s.v[i]
r2t = torch.arange(1, N + 2, dtype=torch.int32, device=dev)[None].repeat(8, 1).contiguous()
import sglang.srt.model_executor.forward_context as FC
FC.get_attn_backend = lambda: types.SimpleNamespace(token_to_kv_pool=Pool(cfg.num_hidden_layers, N + 8), req_to_token=r2t)
class Mode:
    def __init__(s, k): s.k = k
    def is_idle(s): return False
    def is_extend(s): return s.k == "extend"
    def is_target_verify(s): return False
    def is_decode(s): return False
m = SM.Inkling2LoopForConditionalGeneration(cfg)
m.logits_processor = lambda ids, h, head, fb: h
with safe_open(f"{md}/model-00001-of-00001.safetensors", "pt", device="cpu") as f:
    m.load_weights((k, f.get_tensor(k)) for k in f.keys() if not k.startswith("model.visual"))
if "--gemv" in sys.argv:   # route every dense linear (any M) through the decode GEMV path (Q8 or FP4 weights)
    for lin in list(m.lin.values()) + [x[0] for x in m.fused.values()]:
        lin.w = None
tok = PreTrainedTokenizerFast(tokenizer_file=f"{md}/tokenizer.json")
text = open(f"{md}/README.md").read()
ids = torch.tensor(tok.encode(text)[:N], device=dev)
fb = types.SimpleNamespace(forward_mode=Mode("extend"), req_pool_indices=torch.tensor([0], device=dev), extend_seq_lens_cpu=[N],
                           extend_prefix_lens_cpu=[0], seq_lens_cpu=None, out_cache_loc=r2t[0, :N].long())
with torch.no_grad():
    h = m.forward(ids, torch.arange(N, device=dev), fb)
    lp = torch.cat([torch.log_softmax(30 * torch.tanh(m.lm_q8.__class__.__call__(m.lm_q8, h[i:i + 64]) / 30)[:, :200058], -1).half()
                    for i in range(0, N, 64)])
torch.save(lp.cpu(), out)
nll = -lp[:-1].float().gather(1, ids[1:, None]).mean()
print(f"ppl {nll.exp().item():.3f}", flush=True)
if ref_path:
    ref = torch.load(ref_path).to(dev).float(); cur = lp.float()
    kld = (ref.exp() * (ref - cur)).sum(-1)
    print(f"KLD vs ref: mean {kld.mean().item():.4f}  p99 {kld.quantile(0.99).item():.4f}  top1 agree {(ref.argmax(-1) == cur.argmax(-1)).float().mean().item():.4f}")
