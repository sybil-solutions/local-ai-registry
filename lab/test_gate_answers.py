"""The gates must check the answer itself: the reasoning gate the number, the tools gate the tool result."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lab


class Engine(BaseHTTPRequestHandler):
    """A model that answers the gates' questions; `config` decides what it answers and whether it uses the tool result."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send({"data": [{"id": "qwen3.5-9b-exl3"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/token/encode":
            return self.send({"length": len(body["text"]) // 4})
        if self.path != "/v1/chat/completions":
            return self.reply({"error": "no such path"}, 404)
        messages = body["messages"]
        said = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")
        if body.get("stream"):  # the speed gate: a short answer, streamed
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(20):
                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":"word{i} "}}}}]}}\n\n'.encode())
                time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if any(m.get("role") == "tool" for m in messages):  # the tool result is in the conversation
            return self.reply(self.answer(messages))
        if "17 * 23" in said:
            return self.reply({"content": self.config["reasoning"], "reasoning_content": "17 times 23"})
        if "weather in Paris" in said:
            call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            return self.reply({"content": "", "tool_calls": [call]})
        if "access code for the vault" in said:
            return self.reply({"content": "58213"})
        return self.reply({"content": "Red, green, blue"})

    def answer(self, messages):
        """The reply after the tool result: it uses the result, or it answers from its own knowledge."""
        result = json.loads([m for m in messages if m.get("role") == "tool"][0]["content"])
        asked = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")
        if self.config["tools"] == "use":
            # a model that was asked for the code repeats it; one that was not asked does not
            code = f", code {result.get('code', '')}" if "repeat the code" in asked else ""
            return {"content": f"{result['city']} is {result['temp_c']} C and {result['sky']}{code}"}
        return {"content": "I could not retrieve live conditions; Paris typically averages around 17 C in September."}

    def reply(self, message, status=200):
        self.send({"choices": [{"message": message, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 40000, "completion_tokens": 8}}, status)

    def send(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class GateAnswerTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reasoning": "391", "tools": "use"}
        owner = self

        class Handler(Engine):
            config = owner.config

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def gates(self):
        return lab.gates(f"http://127.0.0.1:{self.server.server_port}", {"ctx": 32768})[2]

    def test_reasoning_gate_needs_the_number_itself(self):
        for answer in ("3910", "13915", "The 391st shipment"):
            with self.subTest(answer=answer):
                self.config["reasoning"] = answer
                self.assertFalse(self.gates()["ok"]["reasoning"])

    def test_reasoning_gate_accepts_the_number(self):
        for answer in ("391", "391.", "17 * 23 = 391"):
            with self.subTest(answer=answer):
                self.config["reasoning"] = answer
                self.assertTrue(self.gates()["ok"]["reasoning"])

    def test_tools_gate_needs_the_tool_result(self):
        self.config["tools"] = "ignore"  # the model answers from its own knowledge: the temperature by heart
        evidence = self.gates()
        self.assertIn("17", evidence["tools"]["reply"])  # the old gate passed this reply
        self.assertFalse(evidence["ok"]["tools"])

    def test_tools_gate_accepts_a_reply_that_used_the_result(self):
        self.config["tools"] = "use"
        self.assertTrue(self.gates()["ok"]["tools"])


if __name__ == "__main__":
    unittest.main()
