"""A proxy proof must not claim a memory cap, or a card the run did not have."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lab


WEIGHTS = "TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938"
ENGINE = "tabbyapi-exl3@0f83e6198dc3"


def recipe(card_id, proxy=None):
    proof = {"at": "2026-10-20", "on": "vast", "gpu": "RTX 4060", "gates": "load chat reasoning tools context speed",
             "tps": 60.0, "prefill": 900, "served": "Qwen3.5-9B-exl3-22ef1303"}
    if proxy:
        proof["proxy"] = proxy
    return {"model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {}, "card": card_id, "proof": [proof]}


class ProxyCardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root, RUNS=self.root / "lab/runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, r):
        out = lab.recipe_path(r, lab.render(r))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, separators=(",", ":")) + "\n")
        return out

    def test_proxy_refuses_a_sibling_with_another_memory(self):
        self.write(recipe("rtx-4090-24gb"))
        with self.assertRaises(SystemExit) as raised:
            lab.cmd_proxy(argparse.Namespace(card="rtx-3070-ti-8gb", sibling="rtx-4090-24gb"))
        self.assertIn("same memory", str(raised.exception))
        self.assertFalse((self.recipes / "nvidia/rtx-3070-ti-8gb").exists())

    def test_proxy_copies_a_sibling_with_the_same_memory(self):
        self.write(recipe("rtx-4060-8gb"))
        lab.cmd_proxy(argparse.Namespace(card="rtx-3070-8gb", sibling="rtx-4060-8gb"))
        copied = json.loads(lab.recipe_path(recipe("rtx-3070-8gb"), lab.render(recipe("rtx-3070-8gb"))).read_text())
        self.assertEqual(copied["card"], "rtx-3070-8gb")
        self.assertEqual(copied["proof"][0]["proxy"], "rtx-4060-8gb")

    def test_check_refuses_a_proxy_with_another_memory(self):
        self.write(recipe("rtx-3070-ti-8gb", proxy="rtx-4090-24gb"))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(lab.cmd_check(None), 1)
        self.assertIn("same memory", printed.getvalue())

    def args(self, **kw):
        given = dict(weights=WEIGHTS, model="qwen3.5-9b", engine=ENGINE, card="rtx-3070-ti-8gb", set=[], on="vast",
                     endpoint=None, gpu=None, proxy_gpu="RTX 4060", proxy_vram=None, disk=60, dry_run=False,
                     min_inet=500, min_cuda=None, max_price=1.5, any_host=False, min_ram=None, min_disk_bw=None,
                     min_pcie=0, max_inet_cost=0.02, label=None, ready_timeout=5400)
        given.update(kw)
        return argparse.Namespace(**given)

    def try_on(self, args):
        proof = {"gates": " ".join(lab.GATES), "tps": 60.0, "prefill": 900, "served": "Qwen3.5-9B-exl3-22ef1303"}
        evidence = {"ok": {gate: True for gate in lab.GATES}}
        with patch.object(lab, "rented", return_value=("http://127.0.0.1:1", {"on": "vast", "gpu": "RTX 4060"}, lambda: None)), \
                patch.object(lab, "gates", return_value=(True, proof, evidence)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lab.cmd_try(args), 0)
        return json.loads(lab.recipe_path(recipe("rtx-3070-ti-8gb"), lab.render(recipe("rtx-3070-ti-8gb"))).read_text())

    def test_rented_proxy_does_not_claim_a_memory_cap(self):
        proof = self.try_on(self.args())["proof"][0]
        self.assertIn("not memory-capped", proof["proxy"])
        self.assertNotIn("memory capped", proof["proxy"])

    def test_rented_proxy_refuses_a_card_with_more_memory(self):
        with patch.object(lab, "rented", side_effect=AssertionError("the run must not start")):
            with self.assertRaises(SystemExit) as raised:
                lab.cmd_try(self.args(proxy_vram=16))
        self.assertIn("nothing caps a rented card's memory", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
