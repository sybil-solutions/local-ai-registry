#!/usr/bin/env python3
"""Supplementary audit of an observer.py capture (task tool; check_run.py stays the primary, unmodified check).

  audit_capture.py OBSERVER_DIR [--allow-sampler seed,...]

Reads conn-*.req/.resp as observer.py saves them or the committed .req.gz/.resp.gz. A capture with no connection fails.
--allow-sampler lets the named sampler-state keys through (only keys from the sampler list, for a deliberate sampler
probe); output and stop limits and budgets are never allowed.

1. Lists every saved connection: method, path, HTTP status, request sha256.
2. Fails if any POST whose path contains "completions" is not exactly POST /v1/chat/completions (which check_run audits).
3. For every POST body: walks all keys at every depth, except the content of `messages`, and fails on any output or stop limit,
   any key containing "budget", or a sampler-state key: max_tokens, max_completion_tokens, min_tokens, stop, stop_token_ids,
   ignore_eos, seed, n, best_of, logit_bias, repetition_penalty, presence_penalty, frequency_penalty, top_k, top_p, min_p.
   Prints each chat request's top-level keys and sampling fields.
4. For streamed chat answers: collects delta.content and reasoning separately and fails a "stop" answer whose content
   (excluding reasoning) is empty.
5. A connection closed before any answer (an empty response), or a stream cut off by the client, is listed as unanswered
   or partial, with its request still audited; whether answers ended on their own is check_run.py's verdict, not this one.
Standard library only; reads the saved bytes and changes nothing."""
import glob, gzip, hashlib, http.client, io, json, os, sys

LIMITS = {"max_tokens", "max_completion_tokens", "min_tokens", "stop", "stop_token_ids", "ignore_eos"}
SAMPLER = {"seed", "n", "best_of", "logit_bias", "repetition_penalty", "presence_penalty", "frequency_penalty", "top_k", "top_p", "min_p"}


class _Saved:
    def __init__(self, data):
        self.data = data

    def makefile(self, *_a, **_k):
        return io.BytesIO(self.data)


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            yield p, k
            if k != "messages":
                yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


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
    d, bad, rows, unanswered = sys.argv[1], [], [], []
    allowed = set(sys.argv[sys.argv.index("--allow-sampler") + 1].split(",")) if "--allow-sampler" in sys.argv else set()
    if not allowed <= SAMPLER:
        raise SystemExit(f"--allow-sampler takes sampler-state keys only, not {sorted(allowed - SAMPLER)}")
    for conn, raw, resp in saved(d):
        line = raw.split(b"\r\n", 1)[0].decode("latin-1")
        method, path = (line.split(" ") + ["", ""])[:2]
        status, body, partial = None, b"", False
        if resp:
            r = http.client.HTTPResponse(_Saved(resp)); r.begin(); status = r.status
            try:
                body = r.read()
            except http.client.IncompleteRead as e:  # a stream the client closed: audit what arrived
                body, partial = e.partial, True
        row = {"conn": conn, "method": method, "path": path, "status": status,
               "req_sha256": hashlib.sha256(raw).hexdigest()[:16]}
        if not resp:
            row["response"] = "none (the connection closed before an answer)"
            unanswered.append(conn)
        elif partial:
            row["response"] = "partial (the client closed the stream)"
            unanswered.append(conn)
        if method == "POST" and "completions" in path and path != "/v1/chat/completions":
            bad.append(f"{row['conn']}: unaudited completions route {path}")
        if method == "POST":
            req = json.loads(raw.partition(b"\r\n\r\n")[2] or b"{}")
            hits = [p for p, k in walk(req) if k in LIMITS or (k in SAMPLER and k not in allowed) or "budget" in k.lower()]
            if hits:
                bad.append(f"{row['conn']}: limit/budget/sampler keys {hits}")
            if path == "/v1/chat/completions":
                row["top_level_keys"] = sorted(req)
                row["temperature"] = req.get("temperature")
                if req.get("stream"):
                    content, reasoning, fin = "", "", []
                    for l in body.decode("utf-8", "replace").splitlines():
                        if l.startswith("data:") and l.strip() != "data: [DONE]":
                            try:
                                c = json.loads(l[5:])["choices"][0]
                            except ValueError:  # the last event of a stream cut off mid-line
                                continue
                            dl = c.get("delta") or {}
                            content += dl.get("content") or ""
                            reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
                            if c.get("finish_reason"):
                                fin.append(c["finish_reason"])
                    row.update(stream_content_chars=len(content), stream_reasoning_chars=len(reasoning), finish=fin)
                    if fin == ["stop"] and not content.strip():
                        bad.append(f"{row['conn']}: streamed stop answer has no content outside reasoning")
        rows.append(row)
        print(json.dumps(row))
    if not rows:
        bad.append(f"no saved connection in {d} (an empty capture proves nothing)")
    print(json.dumps({"verdict": "pass" if not bad else "fail", "connections": len(rows), "problems": bad,
                      **({"allowed_sampler_keys": sorted(allowed)} if allowed else {}),
                      **({"unanswered_or_partial": unanswered} if unanswered else {})}))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
