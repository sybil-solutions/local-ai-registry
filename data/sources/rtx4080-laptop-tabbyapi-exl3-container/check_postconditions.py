#!/usr/bin/env python3
"""Postconditions for one qualification run directory (task tool). Exit 0 only if all hold:

- the stock lab exited 0 (lab.exit), and the speed repeat command exited 0 (repeat.exit)
- speed-repeat.json is one JSON object with pass true and tps >= 15
- recorder.jsonl holds EXACTLY 7 POST /v1/chat/completions records (6 lab gate requests: chat,
  reasoning, tool call, tool result, context, speed; plus the repeat). GET /v1/models and tokenizer calls
  are not counted.
- each of the 7: HTTP 200, no proxy/stream/client error, no error payload, request without
  max_tokens / max_completion_tokens (absent or None), its raw request and response bodies on disk
- non-streamed: at least one choice, every finish_reason "stop", or "tool_calls" with tool calls present
- streamed (request stream true, response SSE): no error chunks, at least one finish_reason, the last
  one "stop", none "length"

Writes postconditions.json next to the inputs and prints the failures.

  check_postconditions.py RUN_DIR
"""
import json, os, sys

MIN_TPS = 15.0
CHAT = "/v1/chat/completions"
EXPECTED = 7


def check(run):
    fails = []

    def exit_code(name):
        try:
            return int(open(os.path.join(run, name)).read().strip())
        except Exception:
            return None

    lab, rep = exit_code("lab.exit"), exit_code("repeat.exit")
    if lab != 0:
        fails.append(f"lab exit {lab}")
    if rep != 0:
        fails.append(f"repeat exit {rep}")
    try:
        lines = [l for l in open(os.path.join(run, "speed-repeat.json")).read().splitlines() if l.strip()]
        r = json.loads(lines[-1]) if len(lines) == 1 else None
        if r is None:
            fails.append(f"speed-repeat.json has {len(lines)} lines, expected one JSON object")
        else:
            if r.get("pass") is not True:
                fails.append(f"repeat pass {r.get('pass')!r}")
            if not isinstance(r.get("tps"), (int, float)) or r["tps"] < MIN_TPS:
                fails.append(f"repeat tps {r.get('tps')!r} < {MIN_TPS}")
    except Exception as e:
        fails.append(f"speed-repeat.json unreadable: {e!r}")

    bodies = os.path.join(run, "recorder.bodies")
    try:
        rows = [json.loads(l) for l in open(os.path.join(run, "recorder.jsonl")) if l.strip()]
    except Exception as e:
        return fails + [f"recorder.jsonl unreadable: {e!r}"], []
    ids = [x.get("n") for x in rows]
    if len(ids) != len(set(ids)):
        fails.append("recorder ids are not unique")
    chats = [x for x in rows if x.get("method") == "POST" and (x.get("path") or "").split("?")[0] == CHAT]
    other_posts = [x for x in rows if x.get("method") == "POST" and x not in chats]
    if len(chats) != EXPECTED:
        fails.append(f"{len(chats)} POST {CHAT} records, expected exactly {EXPECTED}")
    for x in rows:
        if x.get("proxy_error") or x.get("stream_error") or x.get("client_error"):
            if x not in chats:
                fails.append(f"#{x.get('n')} {x.get('path')}: recorder error {x.get('proxy_error') or x.get('stream_error') or x.get('client_error')}")
    summary = []
    for x in chats:
        n, req = x.get("n"), x.get("request") or {}
        tag = f"#{n}"
        bad = []
        if x.get("status") != 200:
            bad.append(f"HTTP {x.get('status')}")
        for k in ("proxy_error", "stream_error", "client_error", "error", "error_body"):
            if x.get(k):
                bad.append(f"{k} {x[k]!r}"[:300])
        for k in ("max_tokens", "max_completion_tokens"):
            if req.get(k) is not None:  # absent, or present as null: no cap
                bad.append(f"request sets {k}={req.get(k)!r}")
        if "has_max_tokens" not in req or "has_max_completion_tokens" not in req:
            bad.append("request summary lacks the cap fields")
        if not isinstance(n, int):
            fails.append(f"record without an id: {x!r}"[:300])
            continue
        if not os.path.exists(os.path.join(bodies, f"{n:03d}.request")):
            bad.append("raw request body missing")
        stream = x.get("stream") is True
        if bool(req.get("stream")) != stream:
            bad.append(f"request stream {req.get('stream')!r} but response stream {stream}")
        if stream:
            if not os.path.exists(os.path.join(bodies, f"{n:03d}.sse")):
                bad.append("raw SSE missing")
            fr = x.get("finish_reasons") or []
            if x.get("errors"):
                bad.append(f"stream error chunks {x['errors']!r}"[:300])
            if not fr:
                bad.append("stream has no finish_reason")
            elif fr[-1] != "stop":
                bad.append(f"stream final finish_reason {fr[-1]!r}")
            if "length" in fr:
                bad.append("stream finish_reason length")
            finishes = fr
        else:
            if not os.path.exists(os.path.join(bodies, f"{n:03d}.json")):
                bad.append("raw response body missing")
            ch = x.get("choices")
            if not ch:
                bad.append("no choices")
            finishes = [c.get("finish_reason") for c in ch or []]
            for c in ch or []:
                f = c.get("finish_reason")
                if f == "stop":
                    continue
                if f == "tool_calls" and c.get("tool_calls"):
                    continue
                bad.append(f"finish_reason {f!r}" + (" without tool calls" if f == "tool_calls" else ""))
        summary.append({"n": n, "stream": stream, "finish": finishes, "status": x.get("status"), "ok": not bad})
        fails += [f"{tag}: {b}" for b in bad]
    if other_posts:
        summary.append({"other_posts": len(other_posts)})
    return fails, summary


def main():
    run = sys.argv[1]
    fails, summary = check(run)
    out = {"pass": not fails, "failures": fails, "chat_records": summary}
    with open(os.path.join(run, "postconditions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
