import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lab


class Engine(BaseHTTPRequestHandler):
    """A model server that answers like an engine and has no tokenizer endpoint, so the lab has to
    count tokens by characters. It also answers in a fraction of a second, so the speed window the
    lab measures is shorter than the 30 s it asks for."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        last = ""
        for m in body.get("messages") or []:
            if m.get("content"):
                last = m["content"]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for piece in ["a ", "short ", "answer"]:
                chunk = {"id": "c", "object": "chat.completion.chunk", "model": "fake-model",
                         "choices": [{"index": 0, "delta": {"content": piece}}]}
                payload = ("data: " + json.dumps(chunk) + "\n\n").encode()
                self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if "three primary colors" in last:
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Red, green, blue."},
                                          "finish_reason": "stop"}]})
        elif "17 * 23" in last:
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "reasoning_content": "391",
                                                                  "content": "391"}, "finish_reason": "stop"}]})
        elif "weather in Paris" in last:
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                "finish_reason": "tool_calls"}]})
        elif "access code" in last:
            # a model that answers in a sentence puts the code in the middle of it
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content":
                "The access code for the vault is 58213. I found it in the archive notes, near the end of the log, "
                "where the shipment records are listed, and I am reporting it exactly as it was written there."},
                "finish_reason": "stop"}]})
        else:
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "17 degrees, overcast."},
                                          "finish_reason": "stop"}]})


class SpeedEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        self.endpoint = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()

    def test_the_speed_evidence_says_what_was_measured(self):
        ok, proof, ev = lab.gates(self.endpoint, {"ctx": 4096, "port": 12434})
        speed = ev["speed"]
        # the window the rate was measured over is the one that was really measured
        self.assertNotEqual(speed["window_s"], 30)
        self.assertAlmostEqual(speed["window_s"], speed["tokens_in_window"] / proof["tps"], delta=0.5)
        # and the evidence says where the token count came from
        self.assertEqual(speed["counted"], "estimate")

    def test_the_context_evidence_keeps_the_answer(self):
        ok, proof, ev = lab.gates(self.endpoint, {"ctx": 4096, "port": 12434})
        # the code has to be readable in the record: a reader checks it there
        self.assertIn("58213", ev["context"]["content"])

    def test_a_rate_counted_by_characters_does_not_pass_the_gate(self):
        ok, proof, ev = lab.gates(self.endpoint, {"ctx": 4096, "port": 12434})
        self.assertFalse(ev["ok"]["speed"])
        self.assertNotIn("speed", proof["gates"].split())


if __name__ == "__main__":
    unittest.main()
