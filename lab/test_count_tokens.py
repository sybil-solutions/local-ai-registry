"""count() falls back to Anthropic's /v1/messages/count_tokens (mtplx, oMLX) and takes the template's tokens off."""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

FRAME = 9  # tokens the fake chat template wraps around every message


class CountTokensOnly(BaseHTTPRequestHandler):
    """Like mtplx and oMLX: no tokenize route, only count_tokens, which counts the templated prompt."""
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path != "/v1/messages/count_tokens":
            self.send_response(404); self.end_headers(); return
        CountTokensOnly.seen.append(body)
        if not body.get("model"):
            self.send_response(422); self.end_headers(); return
        words = sum(len(m["content"].split()) for m in body["messages"])
        out = json.dumps({"input_tokens": words + FRAME}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class TokenizeToo(CountTokensOnly):
    """A server with a tokenize route: that route still wins and count_tokens is never asked."""

    def do_POST(self):
        if self.path == "/tokenize":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"tokens": list(range(len((body.get("prompt") or "").split()) * 2))}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)
            return
        super().do_POST()


def serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


class CountTokensFallback(unittest.TestCase):
    def setUp(self):
        CountTokensOnly.seen = []

    def test_count_tokens_minus_the_template(self):
        srv, url = serve(CountTokensOnly)
        try:
            self.assertEqual(lab.count(url, "one two three four five", "served-id"), (5, "tokenizer"))
            self.assertEqual({b["model"] for b in CountTokensOnly.seen}, {"served-id"})
        finally:
            srv.shutdown(); srv.server_close()

    def test_empty_text_counts_zero(self):
        srv, url = serve(CountTokensOnly)
        try:
            self.assertEqual(lab.count(url, "", "served-id"), (0, "tokenizer"))
        finally:
            srv.shutdown(); srv.server_close()

    def test_without_a_model_it_stays_an_estimate(self):
        srv, url = serve(CountTokensOnly)
        try:
            self.assertEqual(lab.count(url, "x" * 40), (10, "estimate"))
            self.assertEqual(CountTokensOnly.seen, [])
        finally:
            srv.shutdown(); srv.server_close()

    def test_a_tokenize_route_still_comes_first(self):
        srv, url = serve(TokenizeToo)
        try:
            self.assertEqual(lab.count(url, "one two three", "served-id"), (6, "tokenizer"))
            self.assertEqual(CountTokensOnly.seen, [])
        finally:
            srv.shutdown(); srv.server_close()


if __name__ == "__main__":
    unittest.main()
