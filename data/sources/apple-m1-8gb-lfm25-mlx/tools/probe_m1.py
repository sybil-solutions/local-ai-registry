#!/usr/bin/env python3
"""The recipe's limits and its sampler, against a running server (task tool, not part of the lab).

  probe_m1.py ENDPOINT MODEL OUT.json

No request sets max_tokens, a stop or any budget: every answer runs to its natural end. Standard library only. The prompts
use the lab's own context-gate format (filler lines, then a code near the end); the server's usage gives their tokens.
1. Sampler: one short prompt, distinct from the lab's, at temperature 0 twice, at 1.0 twice without a seed, at 1.0 twice
   with seed 7, and once with seed 8. Each answer's sha256 is recorded, so equal and different outputs show.
2. Over the prompt limit (24,576): prompts of about 24,900 and 30,000 tokens must get HTTP 400 at once, with no answer.
3. Back to back: ten distinct prompts of about 24,000 tokens (inside the limit), each opening differently so that no two
   share a prefix, each with its own code to recall. The server keeps at most one finished cache (--prompt-cache-size 1);
   memory over the ten is in the run's memwatch log, and the server log prints its prompt-cache count before each request.
4. A long answer: a prompt of about 24,000 tokens that asks for a long story and then the code on the last line, so that
   prompt plus answer can pass the 32,768-token window; it records the totals and whether the code came back.
Writes OUT.json with every request's tokens (the server's usage), seconds, finish reason and result.
"""
import hashlib, json, sys, time, urllib.error, urllib.request

ep, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
TOKENS_PER_LINE = 22242 / 1160  # the lab's filler on this tokenizer: 22,242 prompt tokens from its 1,160 lines at ctx 32,768
LIMIT, WINDOW = 24576, 32768


def post(body):
    req = urllib.request.Request(ep + "/v1/chat/completions", data=json.dumps({"model": model, **body}).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return r.status, json.loads(r.read()), round(time.monotonic() - t, 1)
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read() or b"{}"), round(time.monotonic() - t, 1)


def chat(messages, **kw):
    started = time.time()
    status, o, secs = post({"messages": messages, **kw})
    rec = {"request": dict(kw), "status": status, "seconds": secs, "started_unix": round(started, 1), "finished_unix": round(time.time(), 1)}
    if status != 200:
        return {**rec, "error": o.get("error")}
    c, u = o["choices"][0], o.get("usage") or {}
    content = c["message"].get("content") or ""
    reasoning = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    return {**rec, "finish": c.get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "content_chars": len(content), "reasoning_chars": len(reasoning),
            "answer_sha256": hashlib.sha256((reasoning + "\x00" + content).encode()).hexdigest(), "content_tail": content[-160:]}


def archive(tokens, volume=0, code="58213"):
    """The lab's filler, opened by a line naming its volume (so volumes share no prefix), with the code near the end."""
    lines = int(tokens / TOKENS_PER_LINE)
    head = f"Archive volume {volume}: {['north', 'south', 'east', 'west', 'river', 'hill', 'harbor', 'market', 'mill', 'bridge', 'quay'][volume % 11]} records. "
    filler = " ".join(f"Line {i}: the archive notes that shipment {(i + volume) * 7 % 997} left dock {(i + volume) % 13} on schedule."
                      for i in range(lines))
    return head + filler + f" The access code for the vault is {code}. " + "Anything else is routine.", lines


def log(rec, *keys):
    print(json.dumps({k: rec.get(k) for k in keys}), flush=True)


res = {"started_unix": int(time.time()), "model": model, "limit": LIMIT, "window": WINDOW, "sampler": [], "over_limit": [],
       "back_to_back": [], "long_answer": None}
PROMPT = [{"role": "user", "content": "Write one sentence about a red kite flying over a lake."}]
for label, kw in (("t0 #1", {"temperature": 0.0}), ("t0 #2", {"temperature": 0.0}),
                  ("t1 #1", {"temperature": 1.0}), ("t1 #2", {"temperature": 1.0}),
                  ("t1 seed7 #1", {"temperature": 1.0, "seed": 7}), ("t1 seed7 #2", {"temperature": 1.0, "seed": 7}),
                  ("t1 seed8", {"temperature": 1.0, "seed": 8})):
    res["sampler"].append({"label": label, **chat(PROMPT, **kw)})
    log(res["sampler"][-1], "label", "status", "answer_sha256", "completion_tokens", "finish")
s = {x["label"]: x.get("answer_sha256") for x in res["sampler"]}
res["sampler_comparisons"] = {
    "t0 #1 == t0 #2": s["t0 #1"] == s["t0 #2"],
    "t1 #1 == t1 #2 (no seed)": s["t1 #1"] == s["t1 #2"],
    "t1 seed7 #1 == t1 seed7 #2": s["t1 seed7 #1"] == s["t1 seed7 #2"],
    "t1 seed7 == t1 seed8": s["t1 seed7 #1"] == s["t1 seed8"],
    "t0 == t1 #1": s["t0 #1"] == s["t1 #1"],
}
for target in (24900, 30000):
    prompt, lines = archive(target)
    r = chat([{"role": "user", "content": prompt + "\n\nWhat is the access code for the vault? Reply with only the code."}], temperature=0.6)
    res["over_limit"].append({"target_prompt_tokens": target, "lines": lines, "rejected": r["status"] == 400 and "finish" not in r, **r})
    log(res["over_limit"][-1], "target_prompt_tokens", "status", "seconds", "error")
for k in range(10):
    code = str(10000 + (k * 7919) % 90000)
    prompt, lines = archive(24000, volume=k + 1, code=code)
    r = chat([{"role": "user", "content": prompt + "\n\nWhat is the access code for the vault? Reply with only the code."}], temperature=0.6)
    res["back_to_back"].append({"volume": k + 1, "code": code, "lines": lines, "recalled": r["status"] == 200 and code in r.get("content_tail", ""), **r})
    log(res["back_to_back"][-1], "volume", "status", "prompt_tokens", "completion_tokens", "recalled", "finish", "seconds")
prompt, lines = archive(24000, volume=11, code="47106")
r = chat([{"role": "user", "content": prompt + "\n\nNow write a detailed 3,000-word story about the dock workers in this archive. "
                                     "After the story, on its own last line, write CODE: followed by the access code for the vault."}],
         temperature=0.6)
total = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
res["long_answer"] = {"code": "47106", "lines": lines, "recalled": r["status"] == 200 and "47106" in r.get("content_tail", ""),
                      "prompt_plus_answer": total, "passed_window": total > WINDOW, **r}
log(res["long_answer"], "status", "prompt_tokens", "completion_tokens", "prompt_plus_answer", "passed_window", "recalled", "finish", "seconds")
res["finished_unix"] = int(time.time())
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"sampler": res["sampler_comparisons"], "over_limit_rejected": [x["rejected"] for x in res["over_limit"]],
                  "back_to_back_recalled": sum(x["recalled"] for x in res["back_to_back"]),
                  "long_answer": {k: res["long_answer"][k] for k in ("prompt_plus_answer", "passed_window", "recalled")}}))
