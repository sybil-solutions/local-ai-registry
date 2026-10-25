#!/usr/bin/env python3
"""Count the tokens of streamed answers from the raw bytes observer.py kept, with the model's own pinned tokenizer.

  count_stream_tokens.py TOKENIZER_JSON SITE_DIR OBSERVER_DIR CONN [CONN ...]

Each response is read from CONN.resp or the committed CONN.resp.gz. For each connection, the saved response is parsed as real HTTP (http.client, which removes chunked framing); every SSE
`data:` event's delta is taken as reasoning_content or reasoning, then content, in order, exactly as lab.py's stream_rate()
joins them. The joined text is re-encoded with tokenizers.Tokenizer.from_file(TOKENIZER_JSON), without special tokens.
That is an estimate of the generated tokens, not a count of them: re-encoding decoded text can split a boundary
differently, and tokens the server strips from the deltas (think delimiters, EOS) are not in the text. The stream's own
counts are reported too: `events` (every data event) and `content_events` (events carrying reasoning or content text).
A stream the client closed before its end is read up to where it stopped and reported with "partial": true.
SITE_DIR (the task runtime's packages) provides `tokenizers`; nothing is downloaded. Prints one JSON object per connection.
"""
import gzip, http.client, io, json, os, sys


class _Sock:
    def __init__(self, raw):
        self.f = io.BytesIO(raw)

    def makefile(self, *a, **k):
        return self.f


def response(d, c):
    """The saved response bytes of connection c: c.resp, or the committed c.resp.gz."""
    plain, gz = os.path.join(d, f"{c}.resp"), os.path.join(d, f"{c}.resp.gz")
    if os.path.exists(plain) and os.path.exists(gz):
        raise SystemExit(f"{c}: saved both plain and gzipped")
    if os.path.exists(gz):
        return gzip.open(gz, "rb").read()
    if os.path.exists(plain):
        return open(plain, "rb").read()
    raise SystemExit(f"{c}: no saved response in {d}")


def stream(raw):
    """(status, joined text, data events, events carrying text, finish reasons, partial) of one saved streamed response."""
    r = http.client.HTTPResponse(_Sock(raw))
    r.begin()
    try:
        body, partial = r.read(), False
    except http.client.IncompleteRead as e:  # the client closed the stream: count what arrived
        body, partial = e.partial, True
    pieces, events, texted, finish = [], 0, 0, []
    for line in body.decode("utf-8", "replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            choices = json.loads(line[6:]).get("choices", [])
        except ValueError:
            if partial:
                continue  # the last event, cut off mid-line
            raise
        events += 1
        for ch in choices:
            delta = ch.get("delta") or {}
            piece = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
            pieces.append(piece)
            texted += bool(piece)
            if ch.get("finish_reason"):
                finish.append(ch["finish_reason"])
    return r.status, "".join(pieces), events, texted, finish, partial


def main():
    tok_path, site, d, conns = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    if not conns:
        raise SystemExit("name at least one connection")
    sys.path.insert(0, site)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    for c in conns:
        status, text, events, texted, finish, partial = stream(response(d, c))
        print(json.dumps({"conn": c, "status": status, "events": events, "content_events": texted, "chars": len(text),
                          "chars_div_4": len(text) // 4, "retokenized_tokens": len(tok.encode(text, add_special_tokens=False).ids),
                          "finish": finish, **({"partial": True} if partial else {})}))


if __name__ == "__main__":
    main()
