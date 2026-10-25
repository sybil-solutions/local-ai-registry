"""A proof says when its tps is an estimate: lab.gates() against a local stub server, with and without a tokenize route."""
import http.server
import json
import threading
import unittest
from unittest.mock import patch

import lab

STORY = "word " * 400  # 2,000 characters; streamed as 200 pieces, quickly


class Stub(http.server.BaseHTTPRequestHandler):
    """An OpenAI-style server that passes every gate. `tokenize` decides whether it has a /tokenize route."""
    tokenize = False

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.reply(200, {"data": [{"id": "stub/model"}]}) if self.path == "/v1/models" else self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path in ("/tokenize", "/v1/tokenize", "/v1/token/encode"):
            if not self.tokenize or self.path != "/tokenize" or "prompt" not in body:
                return self.reply(404, {"error": "not found"})
            return self.reply(200, {"count": len(body["prompt"].split())})  # one token per word
        if self.path != "/v1/chat/completions":
            return self.reply(404, {})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(0, len(STORY), 10):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': STORY[i:i + 10]}}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        last = body["messages"][-1]
        text = last.get("content") or ""
        msg, finish = {"role": "assistant", "content": "red, blue, yellow"}, "stop"
        if "17 * 23" in text:
            msg = {"role": "assistant", "content": "391", "reasoning_content": "17 times 23 is 391."}
        elif body.get("tools") and last["role"] == "user":
            msg, finish = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]}, "tool_calls"
        elif last["role"] == "tool":
            msg = {"role": "assistant", "content": "It is 17 C in Paris."}
        elif "access code" in text:
            msg = {"role": "assistant", "content": "58213"}
        self.reply(200, {"choices": [{"message": msg, "finish_reason": finish}],
                         "usage": {"prompt_tokens": int(len(text) / 4.2), "completion_tokens": 5}})


class TpsEstimate(unittest.TestCase):
    def serve(self, tokenize):
        handler = type("H", (Stub,), {"tokenize": tokenize})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)  # cleanups run last-in first-out: stop serving, then close the socket
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def run_gates(self, tokenize):
        launch = {"ctx": 4096, "weights": []}
        with patch.object(lab, "MIN_TPS", 0.0):  # the stub's stream rate is whatever the machine gives; not under test
            passed, proof, ev = lab.gates(self.serve(tokenize), launch)
        self.assertTrue(passed, ev["ok"])
        return proof, ev

    def test_without_a_tokenize_route_the_proof_notes_the_estimate(self):
        proof, ev = self.run_gates(tokenize=False)
        self.assertEqual(ev["speed"]["counted"], "estimate")
        self.assertIs(proof["tps_estimated"], True)  # the typed flag the catalog and the site read
        self.assertEqual(proof["note"], lab.ESTIMATED_TPS)
        self.assertIn("estimate", proof["note"])
        self.assertEqual(ev["speed"]["tokens_total"], len(STORY) // 4)  # characters/4, as the note says

    def test_with_a_tokenize_route_the_proof_has_no_note(self):
        proof, ev = self.run_gates(tokenize=True)
        self.assertEqual(ev["speed"]["counted"], "tokenizer")
        self.assertNotIn("note", proof)
        self.assertNotIn("tps_estimated", proof)
        self.assertEqual(ev["speed"]["tokens_total"], 400)  # the server's own count

    def test_stream_rate_keeps_its_three_values_and_reports_modes_only_when_asked(self):
        ep = self.serve(tokenize=False)
        out = lab.stream_rate(ep, "stub/model", "story")
        self.assertEqual(len(out), 3)
        modes = []
        lab.stream_rate(ep, "stub/model", "story", modes=modes)
        self.assertEqual(modes, ["estimate", "estimate"])


if __name__ == "__main__":
    unittest.main()
