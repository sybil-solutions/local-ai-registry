#!/usr/bin/env python3
"""Speed sweep of one served model: cold prefill and decode at the standard rows (context x concurrency).

Each wave sends `concurrency` requests at once. Every request has its own context (a nonce first, so nothing is
answered from a prefix cache) of the row's size in the served tokenizer's tokens, followed by a request for a long
piece of prose on a topic of its own. No request sets an output limit: every answer runs to its natural end.

  cold prefill   prompt tokens / time to the first token (at concurrency > 1: all prompts' tokens / the slowest first token)
  decode, one    (completion tokens - 1) / (last token time - first token time), per stream
  decode, total  tokens every stream wrote inside the window in which all of them were decoding / that window

Usage: sweep.py --port 8795 --label NAME [--rows 0:1,8k:1,8k:2,8k:4,32k:1,32k:2] [--warm 2] [--waves 3] [--thinking]
Writes runs/<label>/result.json (medians and every wave) and runs/<label>/texts/ (every answer).
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request
from pathlib import Path

WORDS = 300  # with the drafting this model does first, an answer of 20 to 40 s
TOPICS = ["the history of lighthouses", "how rivers shape valleys", "the craft of bookbinding", "life in a medieval market town",
          "how bread is made, from grain to loaf", "the building of a wooden sailing ship", "a year in the life of a beekeeper",
          "how a mechanical clock works", "the migration of arctic terns", "the making of stained glass windows",
          "how tides work", "the history of the bicycle", "an orchard through four seasons", "how a blacksmith forges a tool",
          "the life of a mountain village in winter", "how maps were made before satellites"]


def call(endpoint, path, body=None, timeout=600):
    req = urllib.request.Request(endpoint + path, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count(endpoint, text):
    """Tokens in a text by the served tokenizer: /v1/token/encode where there is one, vLLM's /tokenize otherwise."""
    if not text:
        return 0
    try:
        return call(endpoint, "/v1/token/encode", {"text": text})["length"]
    except Exception:
        out = call(endpoint, "/tokenize", {"prompt": text, "add_special_tokens": False})
        return int(out.get("count") or len(out.get("tokens") or []))


def context(endpoint, tokens, rng):
    """A text of `tokens` tokens no request has sent before: a nonce line, then numbered archive lines."""
    if not tokens:
        return ""
    nonce = f"Archive {rng.getrandbits(64):016x}, volume {rng.randrange(10**6)}.\n"
    line = lambda i: f"Entry {i}: on day {rng.randrange(1, 366)} the clerk recorded {rng.randrange(10, 9999)} crates of {rng.choice(['tea', 'wool', 'salt', 'iron', 'grain', 'timber', 'glass', 'paper'])} leaving pier {rng.randrange(1, 40)} under a {rng.choice(['clear', 'grey', 'windy', 'wet', 'cold', 'bright'])} sky.\n"
    sample = "".join(line(i) for i in range(200))
    per_line = count(endpoint, sample) / 200
    lines = [line(i) for i in range(int(tokens / per_line) + 50)]
    text = nonce + "".join(lines)
    while count(endpoint, text) > tokens and lines:  # trim to the size from above, a few lines at a time
        over = count(endpoint, text) - tokens
        del lines[-max(1, int(over / per_line)):]
        text = nonce + "".join(lines)
    return text


def stream(endpoint, served, prompt, thinking, out):
    """One streamed request. Fills `out` with times, the pieces as they came, and what the server reported."""
    body = {"model": served, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True}, **({} if thinking is None else {"enable_thinking": thinking})}
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out.update(sent=time.monotonic(), pieces=[], answer="", error=None, usage=None, timings=None, finish=None)
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                raw = raw.decode().strip()
                if not raw.startswith("data:") or raw == "data: [DONE]":
                    continue
                d = json.loads(raw[5:])
                if d.get("usage"):
                    out["usage"] = d["usage"]
                if d.get("timings"):
                    out["timings"] = d["timings"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    thought, said = delta.get("reasoning_content") or delta.get("reasoning") or "", delta.get("content") or ""
                    if thought or said:
                        out["pieces"].append((time.monotonic(), thought + said))
                        out["answer"] += said
                    out["finish"] = ch.get("finish_reason") or out["finish"]
    except Exception as e:  # a failed request is a flagged cell, not a crash
        out["error"] = repr(e)
    out["done"] = time.monotonic()


def loops(text):
    """A 64-word run that repeats: the model is going round in circles."""
    words, seen = text.split(), set()
    for i in range(0, max(0, len(words) - 64), 8):
        key = " ".join(words[i:i + 64])
        if key in seen:
            return True
        seen.add(key)
    return False


def wave(endpoint, served, ctx_tokens, conc, thinking, rng, texts, tag, prefill_only=False):
    prompts = []
    for k in range(conc):
        topic = rng.choice(TOPICS)
        ask = f"Write an essay of about {WORDS} words on {topic}. Use full paragraphs, no lists."
        if prefill_only:  # the first token is all that is timed, so the answer may be one sentence
            ask = f"In one sentence, say what {topic} is about."
        ctx = context(endpoint, ctx_tokens, rng)
        prompts.append((ctx + "\nThe archive above is background only; do not summarise it.\n\n" if ctx else "") + ask)
    outs = [{} for _ in range(conc)]
    load = os.getloadavg()[0]
    threads = [threading.Thread(target=stream, args=(endpoint, served, p, thinking, o)) for p, o in zip(prompts, outs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flags, rows = set(), []
    for k, o in enumerate(outs):
        text = "".join(p for _, p in o["pieces"])
        (texts / f"{tag}-s{k}.txt").write_text(text)
        if o["error"] or not o["pieces"]:
            flags.add("REQ_FAIL")
            rows.append({"error": o["error"]})
            continue
        u = o["usage"] or {}
        prompt_tokens = u.get("prompt_tokens") or count(endpoint, prompts[k])
        completion = u.get("completion_tokens") or count(endpoint, text)
        first, last = o["pieces"][0][0], o["pieces"][-1][0]
        if (u.get("prompt_tokens_details") or {}).get("cached_tokens"):
            flags.add("CACHE_HIT")
        if completion < 300 and not prefill_only:
            flags.add("SHORT_OUT")
        if loops(o["answer"]):  # the answer alone: a draft in the thinking that the answer then writes out is not a loop
            flags.add("LOOP")
        rows.append({"prompt_tokens": prompt_tokens, "completion_tokens": completion, "ttft_s": round(first - o["sent"], 3), "decode_s": round(last - first, 3),
                     "decode_tps": round((completion - 1) / (last - first), 2) if last > first else None, "finish": o["finish"], "thinking_chars": len(text) - len(o["answer"]),
                     "peak_memory_gb": (o["timings"] or {}).get("peak_memory"), "server_prompt_tps": (o["timings"] or {}).get("prompt_per_second"),
                     "server_decode_tps": (o["timings"] or {}).get("predicted_per_second"), "_first": first, "_last": last, "_sent": o["sent"]})
    good = [r for r in rows if "error" not in r]
    res = {"load_1m": round(load, 1), "flags": sorted(flags), "streams": rows}
    if good:
        res["prefill_tps"] = round(sum(r["prompt_tokens"] for r in good) / max(r["ttft_s"] for r in good), 1) if ctx_tokens else None
        res["decode_per_stream"] = [r["decode_tps"] for r in good]
        if conc == 1:
            res["decode_total_tps"] = good[0]["decode_tps"]
        else:  # the window in which every stream was decoding
            a, b = max(r["_first"] for r in good), min(r["_last"] for r in good)
            if b - a < 1.0:
                res["decode_total_tps"], res["note"] = None, "streams ran one at a time"
            else:
                inside = sum(count(endpoint, "".join(p for t, p in o["pieces"] if a <= t <= b)) for o in outs if not o["error"] and o["pieces"])
                res["decode_total_tps"], res["window_s"] = round(inside / (b - a), 2), round(b - a, 1)
        res["peak_memory_gb"] = max((r["peak_memory_gb"] or 0) for r in good) or None
    for r in rows:
        for k in ("_first", "_last", "_sent"):
            r.pop(k, None)
    return res


def main():
    global WORDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", default="0:1,8k:1,8k:2,8k:4,32k:1,32k:2")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--server-default-thinking", action="store_true", help="say nothing about thinking: the server's default applies")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--words", type=int, default=WORDS, help="length of the essay each request asks for")
    ap.add_argument("--prefill-only", action="store_true", help="short answers: only the cold prefill is measured")
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    WORDS = a.words
    endpoint = f"http://{a.host}:{a.port}"
    if a.server_default_thinking:
        a.thinking = None
    served = call(endpoint, "/v1/models")["data"][0]["id"]
    run = Path(__file__).resolve().parent / "runs" / a.label
    texts = run / "texts"
    texts.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    size = lambda s: int(float(s[:-1]) * 1024) if s.endswith("k") else int(s)
    result = {"label": a.label, "served": served, "thinking": a.thinking, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    for spec in a.rows.split(","):
        ctx, conc = spec.split(":")
        ctx_tokens, conc = size(ctx), int(conc)
        waves = []
        for w in range(a.warm + a.waves):
            res = wave(endpoint, served, ctx_tokens, conc, a.thinking, rng, texts, f"{ctx}-c{conc}-w{w}", a.prefill_only)
            if a.prefill_only:
                res["decode_total_tps"], res["decode_per_stream"] = None, []
            res["warmup"] = w < a.warm
            waves.append(res)
            print(f"{a.label} {ctx} C{conc} wave {w}{' (warm-up)' if w < a.warm else ''}: prefill {res.get('prefill_tps')} decode {res.get('decode_total_tps')} per-stream {res.get('decode_per_stream')} peak {res.get('peak_memory_gb')} GB load {res['load_1m']} {' '.join(res['flags'])}", flush=True)
        kept = [x for x in waves if not x["warmup"]]
        med = lambda key: (lambda v: round(statistics.median(v), 2) if v else None)([x[key] for x in kept if x.get(key) is not None])
        per = [s for x in kept for s in (x.get("decode_per_stream") or []) if s]
        result["rows"].append({"context": ctx, "context_tokens": ctx_tokens, "concurrency": conc, "prefill_tps": med("prefill_tps"), "decode_total_tps": med("decode_total_tps"),
                               "decode_per_stream_median": round(statistics.median(per), 2) if per else None, "peak_memory_gb": max((x.get("peak_memory_gb") or 0) for x in kept) or None,
                               "flags": sorted({f for x in kept for f in x["flags"]}), "notes": sorted({x["note"] for x in kept if x.get("note")}),
                               "load_1m": [x["load_1m"] for x in kept], "waves": waves})
        (run / "result.json").write_text(json.dumps(result, indent=1))
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
