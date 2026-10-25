#!/usr/bin/env python3
"""A near-limit prompt followed by an answer, streamed, so the answer's length and its end are visible (task tool).

  long_answer.py ENDPOINT MODEL OUT.json

Each request sends no max_tokens, stop or budget: the server never cuts the answer. The client watches for at most
`budget` seconds; an answer still running then is recorded as not ended, and the client closes the connection, which
stops mlx_lm's generation (its stream loop ends in ctx.stop()). The prompt is the lab's context-gate format (as
probe_m1.py), about 24,000 tokens, inside the 24,576-token limit; the server's own count comes from its keepalive lines
(": keepalive done/total"). Text-carrying stream events count the generated tokens approximately (one per step on the
sequential path; mlx_lm reports no usage on a stream).
1. within the window: a short answer, then the code on the last line;
2. past the window: a long story, then the code on the last line, which needs more than the 8,192 tokens the window
   leaves after a 24,000-token prompt.
"""
import json, re, sys, time, urllib.request

ep, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
TOKENS_PER_LINE = 22242 / 1160
WINDOW = 32768


def archive(tokens, volume, code):
    lines = int(tokens / TOKENS_PER_LINE)
    head = f"Archive volume {volume}: harbor ledgers. "
    filler = " ".join(f"Line {i}: the archive notes that shipment {(i + volume) * 7 % 997} left dock {(i + volume) % 13} on schedule."
                      for i in range(lines))
    return head + filler + f" The access code for the vault is {code}. " + "Anything else is routine."


def stream(prompt, budget):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.6, "stream": True}
    req = urllib.request.Request(ep + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    rec = {"prompt_tokens": None, "events": 0, "content": "", "reasoning_chars": 0, "finish": None, "ended": False, "first_event_s": None}
    with urllib.request.urlopen(req, timeout=3600) as r:
        rec["status"] = r.status
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            m = re.match(r": keepalive (\d+)/(\d+)", line)
            if m:
                rec["prompt_tokens"] = int(m.group(2))
            elif line == "data: [DONE]":
                rec["ended"] = True
                break
            elif line.startswith("data: "):
                ch = (json.loads(line[6:]).get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    rec["events"] += 1
                    if rec["first_event_s"] is None:
                        rec["first_event_s"] = round(time.monotonic() - t0, 1)
                rec["content"] += d.get("content") or ""
                rec["reasoning_chars"] += len(d.get("reasoning") or d.get("reasoning_content") or "")
                rec["finish"] = ch.get("finish_reason") or rec["finish"]
            if time.monotonic() - t0 > budget:
                break  # closing the connection stops the generation
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


def repeated_share(text, n=40):
    """Share of the last 4,000 characters' n-character windows seen earlier in that span: near 1 for a loop."""
    tail = text[-4000:]
    grams = [tail[i:i + n] for i in range(0, max(0, len(tail) - n), 10)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return round(rep / len(grams), 2) if grams else 0.0


CASES = [("within the window", "61842", 24000, 1200,
          "\n\nIn three sentences, say what this archive records. Then, on its own last line, write CODE: followed by the access code for the vault."),
         ("past the window", "47106", 24000, 1500,
          "\n\nNow write a detailed 3,000-word story about the dock workers in this archive. After the story, on its own last line, "
          "write CODE: followed by the access code for the vault.")]
res = {"window": WINDOW, "cases": []}
for label, code, tokens, budget, ask in CASES:
    r = stream(archive(tokens, 12 + len(res["cases"]), code) + ask, budget)
    total = (r["prompt_tokens"] or 0) + r["events"]
    rec = {"case": label, "code": code, "budget_s": budget, "prompt_plus_events": total, "passed_window": total > WINDOW,
           "recalled": code in r["content"][-200:], "repeated_share_of_tail": repeated_share(r["content"]),
           "content_chars": len(r["content"]), "content_tail": r["content"][-600:], **{k: v for k, v in r.items() if k != "content"}}
    res["cases"].append(rec)
    print(json.dumps({k: rec[k] for k in ("case", "status", "prompt_tokens", "events", "prompt_plus_events", "passed_window", "ended",
                                          "finish", "recalled", "repeated_share_of_tail", "seconds")}), flush=True)
json.dump(res, open(out, "w"), indent=1)
