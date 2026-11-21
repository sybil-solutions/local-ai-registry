"""Incremental (chunked / token-by-token) vs full forward of the reference: per-position KL; isolates cache bugs."""
import sys, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from transformers import PreTrainedTokenizerFast
from exl3xpu.inkling2.ref import Inkling2Ref
md = sys.argv[1]
tok = PreTrainedTokenizerFast(tokenizer_file=f"{md}/tokenizer.json")
m = Inkling2Ref(md)
text = open(f"{md}/README.md").read()[2000:6000]
ids = torch.tensor(tok.encode(text)[:300], device="xpu")
full = m.forward(ids).float()
def kl(a, b):
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)
nll = torch.nn.functional.cross_entropy(full[:-1], ids[1:])
print(f"ppl(full, 300 tok README) {nll.exp().item():.3f}")
# same full forward twice -> numeric noise floor
m.reset(); full2 = m.forward(ids).float()
print(f"full vs full: max KL {kl(full, full2).max().item():.2e}")
for split in (299, 200, 150):
    m.reset()
    a = m.forward(ids[:split]); b = m.forward(ids[split:])
    inc = torch.cat([a, b]).float()
    k = kl(full, inc)
    print(f"split {split}: KL first-part max {k[:split].max().item():.2e} | second-part max {k[split:].max().item():.2e} mean {k[split:].mean().item():.2e}  argmax agree {(full.argmax(-1)==inc.argmax(-1)).float().mean().item():.3f}")
m.reset(); a = m.forward(ids[:280]); outs=[a]
for i in range(280, 300): outs.append(m.forward(ids[i:i+1]))
inc = torch.cat(outs).float(); k = kl(full, inc)
print("token-by-token 280..299 KL:", [f"{x:.1e}" for x in k[280:].tolist()])
