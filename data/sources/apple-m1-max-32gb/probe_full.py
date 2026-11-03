"""One needle request near the full 131,072-token window; prints server-reported usage and timing."""
import json, time, urllib.request
E = "http://127.0.0.1:8080"
def post(path, body, timeout=7200):
    r = urllib.request.urlopen(urllib.request.Request(E + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read())
line = lambda i: f"Line {i}: the archive notes that shipment {i * 7 % 997} left dock {i % 13} on schedule."
n = 4000
while post("/v1/token/encode", {"text": " ".join(line(i) for i in range(n))})["length"] < 128000:
    n = int(n * 1.15)
while post("/v1/token/encode", {"text": " ".join(line(i) for i in range(n))})["length"] > 128500:
    n -= 50
prompt = " ".join(line(i) for i in range(n)) + " The access code for the vault is 58213. Anything else is routine."
t = time.monotonic()
out = post("/v1/chat/completions", {"model": "models/LFM2.5-24B-A2B-MLX-6bit-1deec53e", "temperature": 0.6,
            "messages": [{"role": "user", "content": prompt + "\n\nWhat is the access code for the vault? Reply with only the code."}]})
secs = time.monotonic() - t
c = out["choices"][0]
print(json.dumps({"lines": n, "usage": out["usage"], "seconds": round(secs, 1), "finish": c["finish_reason"],
                  "content": c["message"].get("content"), "pass": "58213" in (c["message"].get("content") or "")}))
