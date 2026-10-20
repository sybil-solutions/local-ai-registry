"""A run keeps its evidence: a failed gate, a refused prompt and a box that cannot be destroyed."""
import argparse
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import lab
import rent


WEIGHTS = "TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938"
ENGINE = "tabbyapi-exl3@0f83e6198dc3"


class Provider:
    """A rented box that answers /v1/models and whose destroy may fail."""

    def __init__(self, ns=None):
        self.failing = False

    def create(self, spec, exclude):
        return {"id": "fake-1", "endpoint": "http://127.0.0.1:1", "gpu": "RTX 3070 Ti", "cost": 1.5}

    def poll(self, handle):
        return True, "running"

    def destroy(self, handle):
        if self.failing:
            raise SystemExit("vastai destroy instance failed: connection reset by peer")


class Engine(BaseHTTPRequestHandler):
    """An engine that refuses the context gate's prompt with a 400, the way vLLM does with an over-long prompt."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_json({"data": [{"id": "qwen3.5-9b-exl3"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        said = "\n".join(m.get("content") or "" for m in body["messages"])
        if "access code for the vault" in said:
            self.server.refused += 1
            return self.send_json({"error": {"message": "maximum context length"}}, 400)
        self.send_json({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 4}})

    def send_json(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root, RUNS=self.root / "lab/runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, **kw):
        given = dict(weights=WEIGHTS, model="qwen3.5-9b", engine=ENGINE, card="rtx-3070-ti-8gb", set=[], on="vast",
                     endpoint=None, gpu=None, proxy_gpu=None, proxy_vram=None, disk=60, dry_run=False, min_inet=500,
                     min_cuda=None, max_price=1.5, any_host=False, min_ram=None, min_disk_bw=None, min_pcie=0,
                     max_inet_cost=0.02, label=None, ready_timeout=5400, machine=None, exclude_machine=None)
        given.update(kw)
        return argparse.Namespace(**given)

    def try_with(self, provider, gates):
        """A run on the fake box; returns the error it raised."""
        with patch.object(rent, "PROVIDERS", {"vast": lambda ns: provider}), \
                patch.object(rent, "http_probe", lambda url: "ok"), patch.object(lab, "gates", gates), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(BaseException) as raised:
                lab.cmd_try(self.args())
        return raised.exception, errors.getvalue()

    def records(self):
        return sorted((self.root / "lab/runs").glob("*.json"))

    def test_a_failed_gate_still_leaves_the_run_record(self):
        error, _ = self.try_with(Provider(), lambda *a: (_ for _ in ()).throw(RuntimeError("the context gate exploded")))
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(len(self.records()), 1)
        self.assertIn("the context gate exploded", self.records()[0].read_text())

    def test_a_box_that_cannot_be_destroyed_does_not_mask_the_error_or_lose_the_record(self):
        provider = Provider()
        provider.failing = True
        error, errors = self.try_with(provider, lambda *a: (_ for _ in ()).throw(RuntimeError("the context gate exploded")))
        self.assertIsInstance(error, RuntimeError)  # not the destroy failure
        self.assertEqual(len(self.records()), 1)
        self.assertIn("the context gate exploded", self.records()[0].read_text())
        self.assertIn("could not destroy", errors)
        self.assertIn("fake-1", errors)

    def test_a_refused_prompt_is_not_a_dropped_connection(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        server.refused = 0
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with self.assertRaises(SystemExit) as raised:
            lab.gates(f"http://127.0.0.1:{server.server_port}", {"ctx": 32768})
        self.assertIn("the server refused the prompt", str(raised.exception))
        self.assertEqual(server.refused, 1)  # a refusal is the server saying no: it is not retried


if __name__ == "__main__":
    unittest.main()
