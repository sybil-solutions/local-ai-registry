"""Boundary: null and empty inputs are rejected with clear diagnostics."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path as _P

import lab.lab as lab_module
from lab.lab import card, cmd_check


def make_recipe(**kw):
    r = {
        "model": "qwen3.5-9b",
        "weights": "TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938",
        "engine": "tabbyapi-exl3@0f83e6198dc3",
        "set": {},
        "card": "rtx-3070-ti-8gb",
        "proof": [{"at": "2026-10-20", "on": "vast", "gpu": "RTX 3070 Ti",
                   "gates": "load chat reasoning tools context speed",
                   "tps": 60.0, "prefill": 900, "served": "Qwen3.5-9B-exl3"}],
    }
    r.update(kw)
    return r


class BoundaryTests(unittest.TestCase):
    OUT = None  # set in setUp

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recipes = _P(self.temp.name) / "registry/recipes"
        self.recipes.mkdir(parents=True)
        self.out = self.recipes / "nvidia" / "rtx-3070-ti-8gb" / "qwen3.5-9b.tabbyapi.64k.json"
        self.patcher = patch.multiple(
            lab_module,
            RECIPES=self.recipes,
            ROOT=_P(self.temp.name),
            RUNS=_P(self.temp.name) / "lab" / "runs",
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def run_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return cmd_check(None)

    def write(self, recipe):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(recipe, separators=(",", ":")) + "\n", encoding="utf-8")

    def test_null_proof_is_rejected(self):
        self.write(make_recipe(proof=None))
        self.assertEqual(self.run_check(), 1)

    def test_empty_proof_list_is_rejected(self):
        self.write(make_recipe(proof=[]))
        self.assertEqual(self.run_check(), 1)

    def test_proof_with_null_entry_is_rejected(self):
        self.write(make_recipe(proof=[None]))
        self.assertEqual(self.run_check(), 1)

    def test_empty_set_collection_is_rejected(self):
        self.write(make_recipe(set=""))
        self.assertEqual(self.run_check(), 1)

    def test_null_set_is_rejected(self):
        self.write(make_recipe(set=None))
        self.assertEqual(self.run_check(), 1)

    def test_empty_weights_is_rejected(self):
        self.write(make_recipe(weights=""))
        self.assertEqual(self.run_check(), 1)

    def test_whitespace_weights_is_rejected(self):
        self.write(make_recipe(weights="   "))
        self.assertEqual(self.run_check(), 1)

    def test_unknown_card_is_rejected(self):
        with self.assertRaises(SystemExit):
            card("no-such-card")
