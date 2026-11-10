"""Happy path: a valid rx-7900-xtx-24gb recipe renders and validates end to end."""
import json
import unittest
from pathlib import Path

from lab.lab import RECIPES, render, recipe_path, GATES, MIN_TPS

CARD = "rx-7900-xtx-24gb"
RECIPE_PATH = RECIPES / "amd" / CARD / "qwen3.8-27b.exllamav3.128k.full.json"
KEY = "amd/rx-7900-xtx-24gb/qwen3.8-27b.exllamav3.128k.full"


class HappyPathTests(unittest.TestCase):
    def test_recipe_file_exists_and_is_valid_json(self):
        self.assertTrue(RECIPE_PATH.exists(), f"{RECIPE_PATH} missing")
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(recipe), {"model", "weights", "engine", "set", "card", "proof"})
        self.assertEqual(recipe["model"], "qwen3.8-27b")
        self.assertEqual(recipe["card"], CARD)
        self.assertIsInstance(recipe["set"], dict)
        self.assertIsInstance(recipe["proof"], list)
        self.assertTrue(recipe["proof"], "recipe has no proof")

    def test_recipe_renders_to_launch_with_128k_context(self):
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        launch = render(recipe)
        self.assertEqual(launch["kind"], "host")
        self.assertEqual(launch["ctx"], 131072)  # 128K or more
        self.assertEqual(launch.get("variant"), "full")
        # the file name is derived from the rendered launch, so it cannot collide or drift
        self.assertEqual(recipe_path(recipe, launch), RECIPE_PATH)

    def test_recipe_passes_all_six_gates_at_or_above_min_tps(self):
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        proof = recipe["proof"][0]
        gates = set(proof["gates"].split())
        self.assertTrue(set(GATES) <= gates, f"missing gates: {set(GATES) - gates}")
        self.assertGreaterEqual(proof["tps"], MIN_TPS, f"below {MIN_TPS} tok/s")

    def test_lab_check_accepts_the_new_recipe(self):
        import contextlib, io
        from lab.lab import cmd_check
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = cmd_check(None)
            lines = buf.getvalue().splitlines()
        bad = [l for l in lines if "rx-7900-xtx-24gb" in l]
        self.assertEqual(bad, [], f"recipe rejected: {bad}")
