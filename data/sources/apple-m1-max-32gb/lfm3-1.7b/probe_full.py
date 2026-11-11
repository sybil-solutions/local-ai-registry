#!/usr/bin/env python3
"""One prompt that fills a set share of the window, with a code planted near its end; the model must give the code back.
The prompt is sized with the served tokenizer (the lab's context gate sizes its own from an estimate).
Usage: fullwindow.py <port> <prompt tokens> [out.json]"""
import json, random, sys, time, urllib.request
E, target = f"http://127.0.0.1:{sys.argv[1]}", int(sys.argv[2])
def call(path, body=None, timeout=7200):
    req = urllib.request.Request(E + path, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.monotonic() - t
count = lambda text: call("/v1/token/encode", {"text": text})[0]["length"]
served = call("/v1/models")[0]["data"][0]["id"]
rng = random.Random(time.time_ns())
code = str(rng.randrange(10000, 99999))
line = lambda i: f"Line {i}: the archive notes that shipment {rng.randrange(997)} left dock {rng.randrange(13)} on schedule, run {rng.getrandbits(24):06x}."
per = count(" ".join(line(i) for i in range(400))) / 400
tail = f" The access code for the vault is {code}. Anything else is routine.\n\nWhat is the access code for the vault? Reply with only the code."
lines = [line(i) for i in range(int(target / per) + 200)]
text = " ".join(lines) + tail
while count(text) > target - 40:  # leave room for the chat template
    over = count(text) - (target - 40)
    del lines[-max(1, int(over / per)):]
    text = " ".join(lines) + tail
out, secs = call("/v1/chat/completions", {"model": served, "messages": [{"role": "user", "content": text}], "temperature": 0.6})
m, u, t = out["choices"][0]["message"], out.get("usage") or {}, out.get("timings") or {}
res = {"served": served, "target_tokens": target, "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "seconds": round(secs, 1),
       "prefill_tps": round(u.get("prompt_tokens", 0) / secs) if secs else None, "server_prompt_tps": t.get("prompt_per_second"), "server_decode_tps": t.get("predicted_per_second"),
       "completion_tokens": u.get("completion_tokens"), "peak_memory_gb": t.get("peak_memory"), "peak_gib": round(t["peak_memory"] * 1e9 / 2**30, 2) if t.get("peak_memory") else None,
       "code": code, "content": m.get("content"), "pass": code in (m.get("content") or ""), "finish": out["choices"][0].get("finish_reason"), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps(res))
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(json.dumps(res, indent=1))
