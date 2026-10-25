#!/usr/bin/env python3
"""The same sampler comparison against a stock (unpatched) mlx_lm.server (task tool).

  pristine_sampler.py ENDPOINT MODEL OUT.json

Stock mlx_lm 0.31.3 serves an unseeded request on its batch path and a seeded one on its sequential path
(_is_batchable: no seed). Stock caps every answer at 512 tokens unless the request says otherwise, so each request here
sends max_tokens 512 explicitly; the comparison is of equal and different outputs, not of the recipe.
"""
import hashlib, json, sys, urllib.request

ep, model, out = sys.argv[1], sys.argv[2], sys.argv[3]


def chat(**kw):
    body = {"model": model, "messages": [{"role": "user", "content": "Write one sentence about a red kite flying over a lake."}], "max_tokens": 512, **kw}
    req = urllib.request.Request(ep + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        o = json.loads(r.read())
    c = o["choices"][0]
    m = c["message"]
    text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\x00" + (m.get("content") or "")
    return {"request": kw, "finish": c.get("finish_reason"), "completion_tokens": (o.get("usage") or {}).get("completion_tokens"),
            "answer_sha256": hashlib.sha256(text.encode()).hexdigest()}


res = {"model": model, "server": "stock mlx_lm.server 0.31.3", "requests": []}
for label, kw in (("t0 #1", {"temperature": 0.0}), ("t1 #1 no seed (batch path)", {"temperature": 1.0}),
                  ("t1 #2 no seed (batch path)", {"temperature": 1.0}), ("t1 seed7 #1 (sequential path)", {"temperature": 1.0, "seed": 7}),
                  ("t1 seed7 #2 (sequential path)", {"temperature": 1.0, "seed": 7}), ("t1 seed8 (sequential path)", {"temperature": 1.0, "seed": 8})):
    res["requests"].append({"label": label, **chat(**kw)})
    print(label, res["requests"][-1]["answer_sha256"][:16], res["requests"][-1]["finish"], flush=True)
h = {x["label"]: x["answer_sha256"] for x in res["requests"]}
res["comparisons"] = {
    "batch path: unseeded repeats equal": h["t1 #1 no seed (batch path)"] == h["t1 #2 no seed (batch path)"],
    "sequential path: seed 7 == seed 8": h["t1 seed7 #1 (sequential path)"] == h["t1 seed8 (sequential path)"],
    "t0 == t1 batch": h["t0 #1"] == h["t1 #1 no seed (batch path)"],
    "batch == sequential at t1": h["t1 #1 no seed (batch path)"] == h["t1 seed7 #1 (sequential path)"],
}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["comparisons"]))
