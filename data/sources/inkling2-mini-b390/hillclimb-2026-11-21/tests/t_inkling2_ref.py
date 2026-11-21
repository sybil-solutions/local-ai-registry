"""Smoke-test the Inkling-2-Mini reference: chat-templated greedy generation + incremental-vs-full logits.

  python3 t_inkling2_ref.py /models/Inkling-2-Mini [--ppl FILE]
"""
import sys, time, math
import torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from transformers import PreTrainedTokenizerFast
from exl3xpu.inkling2.ref import Inkling2Ref

md = sys.argv[1]
tok = PreTrainedTokenizerFast(tokenizer_file=f"{md}/tokenizer.json")
t0 = time.time()
m = Inkling2Ref(md)
torch.xpu.synchronize(); print(f"load {time.time()-t0:.1f}s", flush=True)

# 1) incremental decode == full forward on the same tokens
ids = torch.tensor(tok.encode("The quick brown fox jumps over the lazy dog. In 1969, the first humans landed on the"), device="xpu")
full = m.forward(ids)
m.reset()
part = torch.cat([m.forward(ids[:-3])] + [m.forward(ids[i:i + 1]) for i in range(len(ids) - 3, len(ids))])
d = (full - part).abs().max().item()
print(f"incremental vs full: max |dlogit| {d:.4f}, argmax equal {bool((full.argmax(-1) == part.argmax(-1)).all())}")
print("next token after prompt:", repr(tok.decode([full[-1].argmax().item()])))

# 2) greedy continuation of a raw text prompt
m.reset()
x = m.forward(ids)
out = []
for _ in range(24):
    nt = x[-1].argmax().item(); out.append(nt)
    x = m.forward(torch.tensor([nt], device="xpu"))
print("greedy:", repr(tok.decode(out)))

# 3) perplexity on a text file chunk
if "--ppl" in sys.argv:
    text = open(sys.argv[sys.argv.index("--ppl") + 1]).read()
    ids = torch.tensor(tok.encode(text)[:512], device="xpu")
    m.reset()
    lg = m.forward(ids)
    nll = torch.nn.functional.cross_entropy(lg[:-1].float(), ids[1:])
    print(f"ppl@512 {math.exp(nll.item()):.3f}")
