"""Regression for issue #219: the registry exposes a tested rx-7900-xtx-24gb recipe.

The recipe (Qwen3.8-27B, 128 K context) must appear in the rendered catalog with all six
gates (load, chat, reasoning, tools, context, speed) and at least 15 tok/s.

On unpatched source this recipe is absent from the catalog, so the assertion below fails
with the message used as `expected_failure` in /tmp/fleet-artifacts/regression.json.
"""
import json
import unittest
from pathlib import Path

from lab.lab import GATES, MIN_TPS

CARD = "rx-7900-xtx-24gb"
KEY = "amd/rx-7900-xtx-24gb/qwen3.8-27b.exllamav3.128k.full"
MISSING_MSG = f"{KEY}: missing from {CARD} catalog recipes"


class RegressionTests(unittest.TestCase):
    def test_catalog_exposes_rx7900_xtx_24gb_recipe_meeting_all_gates(self):
        root = Path(__file__).resolve().parent.parent.parent
        with open(root / "dist/catalog.json", encoding="utf-8") as fh:
            catalog = json.load(fh)
        card = catalog["cards"][CARD]
        recipes = [key for setup in card["setups"] for key in setup["picks"] + setup["more"]]
        self.assertIn(KEY, recipes, MISSING_MSG)
        recipe = catalog["recipes"][KEY]
        gates = set(recipe["proof"][0]["gates"].split())
        self.assertTrue(set(GATES) <= gates, f"missing gates: {set(GATES) - gates}")
        self.assertGreaterEqual(recipe["proof"][0]["tps"], MIN_TPS,
                                f"below {MIN_TPS} tok/s")
        self.assertEqual(recipe["launch"]["ctx"], 131072)  # 128K or more
