#!/usr/bin/env python3
"""probe_full.py with two changes: the sampling temperature is optional (`default` leaves it to the model's
generation_config, as the lab's gates do), and the answer is capped, so a runaway answer ends itself.
The reasoning and the answer are saved, so a reader can see how an answer that did not end went.
Usage: probe_capped.py <port> <prompt tokens> <temperature|default> <max answer tokens> out.json"""
import json, random, sys, time, urllib.request
E, target, temp, cap, dest = f"http://127.0.0.1:{sys.argv[1]}", int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
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
body = {"model": served, "messages": [{"role": "user", "content": text}], "max_tokens": cap}
if temp != "default":
    body["temperature"] = float(temp)
out, secs = call("/v1/chat/completions", body)
m, u, t = out["choices"][0]["message"], out.get("usage") or {}, out.get("timings") or {}
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
res = {"served": served, "target_tokens": target, "temperature": temp, "max_tokens": cap, "prompt_tokens": u.get("prompt_tokens"),
       "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "seconds": round(secs, 1),
       "server_prompt_tps": t.get("prompt_per_second"), "server_decode_tps": t.get("predicted_per_second"),
       "completion_tokens": u.get("completion_tokens"), "peak_memory_gb": t.get("peak_memory"),
       "peak_gib": round(t["peak_memory"] * 1e9 / 2**30, 2) if t.get("peak_memory") else None,
       "code": code, "content": m.get("content"), "pass": code in (m.get("content") or ""), "finish": out["choices"][0].get("finish_reason"),
       "reasoning_chars": len(reasoning), "reasoning_head": reasoning[:3000], "reasoning_tail": reasoning[-3000:],
       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps({k: v for k, v in res.items() if not k.startswith("reasoning_")}))
open(dest, "w").write(json.dumps(res, indent=1))
