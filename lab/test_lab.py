import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from lab.lab import check_served_weights, gates


class TestCheckServedWeights(unittest.TestCase):
    def test_missing_or_empty_served_rejected(self):
        # 1. Fail-closed: rejection of empty or missing served ID, even for baked-into-image
        ok, reason = check_served_weights("baked-into-image", None)
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("baked-into-image", "")
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("baked-into-image", "   ")
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("Qwen/Qwen2.5-7B-Instruct@1234567890abcdef1234567890abcdef12345678", None)
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

    def test_fixture_and_trivial_identifiers_rejected(self):
        # 2. Rejection of fixture/dummy/mock identifiers
        for forbidden in ["fixture", "dummy", "mock", "test", "none", "null", "default", "model", "local"]:
            ok, reason = check_served_weights("baked-into-image", forbidden)
            self.assertFalse(ok)
            self.assertIn("rejected fixture-like or trivial identifier", reason)

        # Less than 3 normalized chars rejected
        ok, reason = check_served_weights("baked-into-image", "ab")
        self.assertFalse(ok)
        self.assertIn("rejected fixture-like or trivial identifier", reason)

    def test_short_substring_injection_rejected(self):
        # 3. Short substring injections like "a" or "qwen" should not match when size or canonical tokens mismatch
        weights = "meta-llama/Llama-3.1-8B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights, "a")
        self.assertFalse(ok)

        ok, reason = check_served_weights(weights, "llama")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token", reason)

    def test_parameter_size_mismatch_rejected(self):
        # 4. Size/quant confusion (e.g. 0.5b vs 72b or 8b vs 70b)
        weights_72b = "Qwen/Qwen2.5-72B-Instruct-AWQ@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights_72b, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token '72b'", reason)

        weights_70b = "meta-llama/Llama-3.1-70B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights_70b, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token '70b'", reason)

    def test_valid_pinned_revision_prefix_accepted(self):
        # 5. Acceptance of valid pinned revision prefix (>= 7 chars)
        weights = "Qwen/Qwen2.5-7B-Instruct@abcdef0123456789abcdef0123456789abcdef01"
        ok, reason = check_served_weights(weights, "models--qwen--7b--abcdef012345")
        self.assertTrue(ok)
        self.assertIn("pinned revision prefix", reason)

    def test_valid_canonical_model_slug_accepted(self):
        # 6. Acceptance of canonical model slug and core tokens
        weights = "Qwen/Qwen2.5-7B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights, "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(ok)
        self.assertIn("canonical model slug", reason)

        # Core tokens attested with different casing or separators
        ok, reason = check_served_weights(weights, "qwen2.5-7b-instruct-vllm")
        self.assertTrue(ok)

    def test_baked_into_image_valid_served(self):
        ok, reason = check_served_weights("baked-into-image", "deepseek-ai/DeepSeek-V3")
        self.assertTrue(ok)
        self.assertIn("baked-into-image verified", reason)

        # Explicit --served mismatch
        ok, reason = check_served_weights("baked-into-image", "deepseek-ai/DeepSeek-V3", explicit_served="llama-3")
        self.assertFalse(ok)


class _MockModelsServer:
    def __init__(self, models_payload):
        self.models_payload = models_payload
        self.chat_calls = 0
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/v1/models":
                    body = json.dumps(parent.models_payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                parent.chat_calls += 1
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.port}", self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()


class TestGatesServedWeightsFailClosed(unittest.TestCase):
    def test_gates_rejects_mismatched_served_model_before_chat(self):
        launch = {
            "weights": {
                "repo": "Qwen/Qwen2.5-72B-Instruct-AWQ",
                "revision": "1234567890abcdef1234567890abcdef12345678",
                "at": "/models/qwen72b",
            },
            "ctx": 8192,
        }
        ev = {}
        with _MockModelsServer({"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}) as (endpoint, srv):
            passed, proof, evidence = gates(endpoint, launch, ev)
        self.assertFalse(passed)
        self.assertFalse(evidence["ok"]["load"])
        self.assertFalse(evidence["load"]["verified"])
        self.assertEqual(evidence["load"]["served"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertIn("missing required parameter size token '72b'", evidence["load"]["reason"])
        self.assertEqual(proof["gates"], "")
        self.assertEqual(srv.chat_calls, 0)

    def test_gates_rejects_empty_models_data_and_fixture_id(self):
        launch = {"weights": "baked-into-image", "ctx": 4096}
        with _MockModelsServer({"data": []}) as (endpoint, srv):
            passed, proof, evidence = gates(endpoint, launch, {})
        self.assertFalse(passed)
        self.assertFalse(evidence["ok"]["load"])
        self.assertEqual(srv.chat_calls, 0)

        with _MockModelsServer({"data": [{"id": "fixture"}]}) as (endpoint, srv):
            passed, proof, evidence = gates(endpoint, launch, {})
        self.assertFalse(passed)
        self.assertFalse(evidence["ok"]["load"])
        self.assertIn("rejected fixture-like", evidence["load"]["reason"])
        self.assertEqual(srv.chat_calls, 0)

    def test_gates_passes_load_gate_when_served_matches_weights(self):
        launch = {
            "weights": {
                "repo": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "abcdef0123456789abcdef0123456789abcdef01",
                "at": "/models/qwen7b",
            },
            "ctx": 4096,
        }
        ev = {}
        with _MockModelsServer({"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}) as (endpoint, srv):
            passed, _, evidence = gates(endpoint, launch, ev)
        self.assertFalse(passed)  # Fails at mocked HTTP 404 on /v1/chat/completions
        self.assertTrue(evidence["ok"]["load"])
        self.assertTrue(evidence["load"]["verified"])
        self.assertEqual(srv.chat_calls, 1)


if __name__ == "__main__":
    unittest.main()
