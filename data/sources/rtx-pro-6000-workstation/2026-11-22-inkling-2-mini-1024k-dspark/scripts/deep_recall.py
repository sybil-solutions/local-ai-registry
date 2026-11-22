#!/usr/bin/env python3
"""Long-context recall at depth against a running server: three facts planted at chosen depths of one ~900K-token
document (War and Peace, then Anna Karenina, Project Gutenberg plain text), one question asking for all three.

The registry's context gate plants its code in the last ~20 tokens before the question, inside Inkling's 1,024-token
sliding window, so it passes without the global layers having to reach back. This puts facts far from the question.
Each prompt starts with a unique line, so nothing comes from the prefix cache.

    deep_recall.py <endpoint> <model> <filler dir> <out.jsonl> [--tokens 900000] [--depths 0.05,0.5,0.95]
"""
import argparse, json, re, time, urllib.request, uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("endpoint"); ap.add_argument("model"); ap.add_argument("filler"); ap.add_argument("out")
ap.add_argument("--tokens", type=int, default=900000)
ap.add_argument("--depths", default="0.05,0.5,0.95")
ap.add_argument("--set", default="a", choices=["a", "b"], help="which three facts")
a = ap.parse_args()
URL = a.endpoint.rstrip("/")
FACTS = {  # harmless office trivia (Inkling has declined to repeat "access codes" in earlier probes)
    "a": [("fern", "the potted fern on the third floor is named Gerald", "gerald"),
          ("van", "the delivery van parked behind the warehouse is painted teal", "teal"),
          ("room", "the meeting room on the fifth floor is called Juniper", "juniper")],
    "b": [("cat", "the office cat that sleeps in the mailroom is named Biscuit", "biscuit"),
          ("bike", "the bicycle chained by the loading dock is bright orange", "orange"),
          ("kettle", "the kettle in the second-floor kitchen is called Old Faithful", "old faithful")],
}[a.set]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    return post("/tokenize", {"model": a.model, "prompt": text})["count"]


def book(p):
    t = Path(p).read_text(encoding="utf-8").replace("\r\n", "\n")
    return t[t.index("\n", t.index("*** START OF")) + 1: t.index("*** END OF")].strip()


text = book(Path(a.filler) / "pg2600.txt") + "\n\n" + book(Path(a.filler) / "pg1399.txt")
cpt = 200000 / ntok(text[:200000])  # characters per token on this text
body = text[: int(a.tokens * cpt)]
for _ in range(3):  # trim or extend to within 0.5% of the target, counted by the served tokenizer
    n = ntok(body)
    if abs(n - a.tokens) < a.tokens * 0.005:
        break
    body = text[: int(len(body) * a.tokens / n)]
depths = [float(x) for x in a.depths.split(",")]
cuts = sorted((int(len(body) * d), f) for d, f in zip(depths, FACTS))
parts, last = [], 0
for cut, (key, fact, _) in cuts:
    cut = body.rfind("\n", 0, cut) + 1 or cut  # at a line break, so no word is split
    parts += [body[last:cut], f"\nOffice note: {fact}.\n"]
    last = cut
parts.append(body[last:])
q = ("Three office notes are hidden somewhere in the very long text above, each starting with 'Office note:'. "
     "Quote each of the three notes exactly, one per line.")
prompt = f"Document {uuid.uuid4().hex}\n\n" + "".join(parts) + "\n\n" + q
t0 = time.monotonic()
out = post("/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                                    "max_tokens": 16000, "seed": 4242,
                                    "chat_template_kwargs": {"reasoning_effort": "medium"}, "reasoning_effort": "medium"})
secs = time.monotonic() - t0
msg = out["choices"][0]["message"]
ans = (msg.get("content") or "").lower()
res = {"set": a.set, "prompt_tokens": out["usage"]["prompt_tokens"], "completion_tokens": out["usage"]["completion_tokens"],
       "seconds": round(secs, 1), "finish": out["choices"][0].get("finish_reason"),
       "facts": [{"depth": d, "fact": f[1], "found": f[2] in ans} for d, f in zip(depths, FACTS)],
       "answer": msg.get("content"), "thinking_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
res["all_found"] = all(x["found"] for x in res["facts"])
with open(a.out, "a") as fh:
    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "answer"}), flush=True)
