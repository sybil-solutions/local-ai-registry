#!/usr/bin/env python3
"""Check the requests a reproduction sent, from the raw bytes observer.py kept (task tool, not part of the lab). It
reads the saved copies after the run and changes nothing that was relayed.

  check_run.py OBSERVER_DIR [--expect 7]

Reads conn-*.req/.resp as observer.py saves them, or the committed conn-*.req.gz/.resp.gz (the only change from the copy
in data/sources/gtx1650super-vllm-autoround/).

Each saved response is parsed as real HTTP (http.client.HTTPResponse, which removes chunked framing). Passes only if
there are exactly --expect POST /v1/chat/completions requests (the lab's six gate requests plus the speed repeat), and
each one:
- had a well-formed JSON request body of the declared Content-Length, with a messages list;
- sent no output or thinking limit: none of max_tokens, max_completion_tokens, min_tokens, stop, stop_token_ids,
  ignore_eos, or any key containing "budget", at the top level or inside chat_template_kwargs;
- got HTTP 200 with a body that parses (JSON, or for a streamed request text/event-stream), carries no "error" field
  anywhere, and has exactly one choice per message or event;
- ended on its own with exactly one non-null finish_reason: "tool_calls" for the request that offers tools after a
  user turn (the tool gate's first call), with an actual named function call and JSON arguments; "stop" for every
  other, with non-empty content and no tool call. A streamed answer must deliver content and end with data: [DONE].
Prints one JSON line per request and a verdict; exit 0 only on a pass. Standard library only."""
import glob, gzip, http.client, io, json, os, sys

FORBIDDEN = {"max_tokens", "max_completion_tokens", "min_tokens", "stop", "stop_token_ids", "ignore_eos"}


class _Saved:  # lets HTTPResponse read a saved byte string as if it were a socket
    def __init__(self, data):
        self.data = data

    def makefile(self, *_args, **_kw):
        return io.BytesIO(self.data)


def parse_request(raw):
    head, sep, body = raw.partition(b"\r\n\r\n")
    if not sep:
        raise ValueError("request has no header terminator")
    lines = head.decode("latin-1").split("\r\n")
    hdr = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:])}
    if "content-length" not in hdr or int(hdr["content-length"]) != len(body):
        raise ValueError(f"request body is {len(body)} bytes, Content-Length {hdr.get('content-length')}")
    req = json.loads(body)
    if not isinstance(req, dict) or not isinstance(req.get("messages"), list) or not req["messages"]:
        raise ValueError("request has no messages")
    return req


def parse_response(raw):
    r = http.client.HTTPResponse(_Saved(raw))  # type: ignore[arg-type]  # only makefile() is used
    r.begin()
    return r.status, (r.getheader("Content-Type") or ""), r.read()


def has_error(obj):
    if isinstance(obj, dict):
        return "error" in obj or any(has_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_error(v) for v in obj)
    return False


def one_choice(obj, problems, where):
    ch = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(ch, list) or len(ch) != 1 or not isinstance(ch[0], dict):
        problems.append(f"{where}: choices is not a list of one object")
        return None
    return ch[0]


def tool_call_ok(calls):
    if not isinstance(calls, list) or not calls:
        return False
    for c in calls:
        f = (c or {}).get("function") or {}
        if not f.get("name") or not isinstance(f.get("arguments"), str):
            return False
        try:
            json.loads(f["arguments"])
        except ValueError:
            return False
    return True


def check(req_raw, resp_raw):
    problems = []
    req = parse_request(req_raw)
    msgs = req["messages"]
    want = "tool_calls" if req.get("tools") and msgs[-1].get("role") == "user" else "stop"
    keys = set(req) | set(req.get("chat_template_kwargs") or {})
    limits = sorted(k for k in keys if k in FORBIDDEN or "budget" in k)
    if limits:
        problems.append(f"limits sent: {limits}")
    status, ctype, body = parse_response(resp_raw)
    if status != 200:
        problems.append(f"HTTP {status}")
    finishes, content, calls = [], "", []
    if req.get("stream"):
        if "text/event-stream" not in ctype:
            problems.append(f"stream Content-Type {ctype!r}")
        datas = [l[5:].strip() for l in body.decode("utf-8").splitlines() if l.startswith("data:")]
        if not datas or datas[-1] != "[DONE]" or datas.count("[DONE]") != 1:
            problems.append("stream does not end with exactly one data: [DONE]")
        for i, d in enumerate(x for x in datas if x != "[DONE]"):
            ev = json.loads(d)
            if has_error(ev):
                problems.append(f"event {i} carries an error")
            c = one_choice(ev, problems, f"event {i}")
            if c is None:
                continue
            delta = c.get("delta") or {}
            content += (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
            calls += delta.get("tool_calls") or []
            if c.get("finish_reason") is not None:
                finishes.append(c["finish_reason"])
    else:
        if "application/json" not in ctype:
            problems.append(f"Content-Type {ctype!r}")
        obj = json.loads(body)
        if has_error(obj):
            problems.append("response carries an error")
        c = one_choice(obj, problems, "response")
        if c is not None:
            m = c.get("message")
            if not isinstance(m, dict):
                problems.append("choice has no message")
            else:
                content, calls = m.get("content") or "", m.get("tool_calls") or []
            if c.get("finish_reason") is not None:
                finishes.append(c["finish_reason"])
    if finishes != [want]:
        problems.append(f"finish_reason {finishes}, want [{want!r}]")
    if want == "tool_calls" and not tool_call_ok(calls):
        problems.append("finish_reason tool_calls without a named function call with JSON arguments")
    if want == "stop" and (calls or not content.strip()):
        problems.append("a stop answer must have content and no tool call")
    return {"stream": bool(req.get("stream")), "temperature": req.get("temperature"), "finish": finishes, "want": want,
            "problems": problems}


def saved(d):
    """(conn, request bytes, response bytes) for every saved connection in d: conn-*.req/.resp as observer.py writes them,
    or the committed conn-*.req.gz/.resp.gz. A connection saved both ways, or a request without its response, is an error."""
    found = {}
    for p in glob.glob(os.path.join(d, "conn-*.req")) + glob.glob(os.path.join(d, "conn-*.req.gz")):
        conn = os.path.basename(p).split(".req")[0]
        if conn in found:
            raise SystemExit(f"{conn}: saved both plain and gzipped")
        found[conn] = p
    out = []
    for conn in sorted(found):
        q = found[conn]
        gz = q.endswith(".gz")
        read = (lambda f: gzip.open(f, "rb").read()) if gz else (lambda f: open(f, "rb").read())
        resp = q[: -len(".req.gz")] + ".resp.gz" if gz else q[: -len(".req")] + ".resp"
        if not os.path.exists(resp):
            raise SystemExit(f"{conn}: request saved without its response")
        out.append((conn, read(q), read(resp)))
    return out


def main():
    d = sys.argv[1]
    expect = int(sys.argv[sys.argv.index("--expect") + 1]) if "--expect" in sys.argv else 7
    rows = []
    for conn, req, resp in saved(d):
        if not req.startswith(b"POST /v1/chat/completions "):
            continue
        try:
            r = check(req, resp)
        except (ValueError, OSError, http.client.HTTPException) as e:
            r = {"problems": [f"unreadable: {type(e).__name__}: {e}"]}
        rows.append({"conn": conn, **r})
        print(json.dumps(rows[-1]))
    bad = [r for r in rows if r["problems"]]
    ok = len(rows) == expect and not bad
    print(json.dumps({"verdict": "pass" if ok else "fail", "chat_requests": len(rows), "expected": expect,
                      "failing": [r["conn"] for r in bad]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
