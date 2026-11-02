#!/usr/bin/env python3
"""Pass-through recorder (task tool, NOT part of the registry lab or the kit).

Listens on 127.0.0.1:LISTEN, forwards every request unchanged to 127.0.0.1:UPSTREAM, streams the
response back byte for byte as it arrives, and appends one JSON line per request to OUT.jsonl with
what the lab does not keep: every choice's finish_reason, usage, content / reasoning lengths, tool
calls, error payloads, and for streams every finish_reason seen plus chunk counts. Every raw request
body, the raw response body of every non-streamed reply and the full SSE text of every stream go to
OUT.bodies/. Request ids come from one locked counter, so concurrent requests never share an id.

On SIGTERM it stops taking new connections, waits for every in-flight request to finish and write its
row, and exits 0, so a row is never lost or torn when the recorder is stopped right after the last reply.

  finish_recorder.py LISTEN UPSTREAM OUT
"""
import http.client, http.server, json, os, signal, socketserver, sys, threading, time

LISTEN, UPSTREAM, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
BODIES = OUT + ".bodies"
os.makedirs(BODIES, exist_ok=True)
lock = threading.Lock()
seq_lock = threading.Lock()
seq = [0]


def next_id():
    with seq_lock:
        seq[0] += 1
        return seq[0]
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection", "content-length"}


def record(row):
    with lock:
        with open(OUT + ".jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def summarize_request(raw):
    try:
        b = json.loads(raw)
    except Exception:
        return {"bytes": len(raw)}
    if not isinstance(b, dict):
        return {"bytes": len(raw), "parse": "not an object"}
    return {"bytes": len(raw), "model": b.get("model"), "stream": b.get("stream"),
            "has_max_tokens": "max_tokens" in b, "has_max_completion_tokens": "max_completion_tokens" in b,
            "max_tokens": b.get("max_tokens"), "max_completion_tokens": b.get("max_completion_tokens"),
            "temperature": b.get("temperature"),
            "messages": len(b.get("messages") or []), "tools": len(b.get("tools") or []),
            "keys": sorted(b)}


def summarize_json(body):
    try:
        d = json.loads(body)
    except Exception:
        return {"parse": "not json", "head": body[:300].decode(errors="replace")}
    out = {"usage": d.get("usage"), "error": d.get("error") or d.get("detail")}
    ch = []
    for c in d.get("choices") or []:
        m = c.get("message") or {}
        ch.append({"finish_reason": c.get("finish_reason"), "stop_str": c.get("stop_str"),
                   "content_chars": len(m.get("content") or ""),
                   "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                   "tool_calls": [(t.get("function") or {}).get("name") for t in (m.get("tool_calls") or [])]})
    out["choices"] = ch
    return out


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def handle_one(self, method):
        n = next_id()
        t0 = time.time()
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else b""
        if raw:
            with open(os.path.join(BODIES, f"{n:03d}.request"), "wb") as f:
                f.write(raw)
        row = {"n": n, "t_start": round(t0, 3), "method": method, "path": self.path,
               "request": summarize_request(raw) if raw else None}
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        try:
            up = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=7200)
            up.request(method, self.path, body=raw or None, headers=hdrs)
            r = up.getresponse()
        except Exception as e:
            row["proxy_error"] = repr(e)
            record(row)
            try:
                self.send_error(502)
            except OSError:
                pass  # the client has gone too
            return
        row["status"] = r.status
        ctype = r.getheader("Content-Type") or ""
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        if "text/event-stream" in ctype:
            self.send_header("Connection", "close")
            self.end_headers()
            finishes, chunks, content_pieces, reasoning_pieces, errors = [], 0, 0, 0, []
            path = os.path.join(BODIES, f"{n:03d}.sse")
            with open(path, "wb") as f:
                while True:
                    try:
                        line = r.readline()
                    except Exception as e:
                        row["stream_error"] = repr(e)
                        break
                    if not line:
                        break
                    f.write(line)
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except Exception as e:
                        row["client_error"] = repr(e)
                        break
                    s = line.strip()
                    if s.startswith(b"data:") and s != b"data: [DONE]":
                        chunks += 1
                        try:
                            d = json.loads(s[5:])
                        except Exception:
                            continue
                        if d.get("error"):
                            errors.append(d["error"])
                        for c in d.get("choices") or []:
                            delta = c.get("delta") or {}
                            content_pieces += bool(delta.get("content"))
                            reasoning_pieces += bool(delta.get("reasoning_content") or delta.get("reasoning"))
                            if c.get("finish_reason"):
                                finishes.append(c["finish_reason"])
                        if d.get("usage"):
                            row["usage"] = d["usage"]
            row.update({"stream": True, "chunks": chunks, "content_chunks": content_pieces,
                        "reasoning_chunks": reasoning_pieces, "finish_reasons": finishes, "errors": errors})
            self.close_connection = True
        else:
            try:
                body = r.read()
            except Exception as e:
                row["proxy_error"] = repr(e)
                record(row)
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            with open(os.path.join(BODIES, f"{n:03d}.json"), "wb") as f:
                f.write(body)
            if "json" in ctype and self.path.startswith("/v1/chat"):
                row.update(summarize_json(body))
            elif r.status >= 400:
                row["error_body"] = body[:2000].decode(errors="replace")
        up.close()
        row["seconds"] = round(time.time() - t0, 3)
        record(row)

    def do_GET(self):
        self.handle_one("GET")

    def do_POST(self):
        self.handle_one("POST")


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = False  # server_close() joins in-flight handlers, so each writes its row first
    block_on_close = True


if __name__ == "__main__":
    srv = S(("127.0.0.1", LISTEN), H)
    # shutdown() must come from another thread than serve_forever's
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    srv.serve_forever()
    srv.server_close()
