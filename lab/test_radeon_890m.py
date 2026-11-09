"""Regression test for issue #223: recipe for Ryzen AI 890M (Radeon 890M iGPu).

Verifies the card definition, launch profile, and recipe all exist and are
mutually consistent, so `lab.py check` accepts the new card.
"""
import json
import unittest
from pathlib import Path

import lab


class Radeon890MTests(unittest.TestCase):
    CARD = "radeon-890m-48gb"

    def test_card_file_exists(self):
        found = list(lab.CARDS.glob(f"*/{self.CARD}.json"))
        self.assertEqual(len(found), 1, f"card {self.CARD} must have exactly one card file")
        c = json.loads(found[0].read_text(encoding="utf-8"))
        self.assertEqual(c["id"], self.CARD)
        self.assertEqual(c["vendor"], "amd")
        self.assertEqual(c["vram_gb"], 48)
        self.assertIn("unifiedMemory", c["match"])
        self.assertTrue(c["match"]["unifiedMemory"])

    def test_launch_file_exists(self):
        # the launch file referenced by the recipe
        launch = Path(lab.LAUNCHES) / "llama.cpp-qwen3.5-9b-q4-k-m-64k-gfx1150.json"
        self.assertTrue(launch.exists(), f"launch {launch} must exist")
        d = json.loads(launch.read_text(encoding="utf-8"))
        self.assertEqual(d["engine"], "llama.cpp")
        self.assertEqual(d["backend"], "amd-vulkan")
        self.assertEqual(d["ctx"], 65536)
        self.assertEqual(d["cards"], 1)
        self.assertIn("plugin", d)
        self.assertGreaterEqual(d["plugin"]["serving"]["kvTokens"], lab.KV_FLOOR_TOKENS)

    def test_recipe_file_exists_and_passes_check(self):
        recipe_dir = lab.RECIPES / "amd" / self.CARD
        recipes = list(recipe_dir.glob("*.json"))
        self.assertEqual(len(recipes), 1, f"exactly one recipe for {self.CARD}")
        r = json.loads(recipes[0].read_text(encoding="utf-8"))
        self.assertEqual(r["model"], "qwen3.5-9b")
        self.assertEqual(r["card"], self.CARD)
        self.assertEqual(r["engine"], "llama.cpp-qwen3.5-9b-q4-k-m-64k-gfx1150@1086b8815273")
        self.assertIsInstance(r["set"], dict)
        self.assertIsInstance(r["proof"], list)
        self.assertTrue(r["proof"])
        p = r["proof"][0]
        gates = set(p["gates"].split())
        self.assertTrue({"load", "chat", "reasoning", "tools", "context", "speed"} <= gates)
        # launch hash must match the launch file
        launch = json.loads((lab.LAUNCHES / "llama.cpp-qwen3.5-9b-q4-k-m-64k-gfx1150.json").read_text(encoding="utf-8"))
        launch_hash = lab.launch_hash(launch)
        self.assertEqual(p["launch_sha256"], launch_hash)
        # recipe path must match render output
        from lab import recipe_path, render
        expected = recipe_path(r, render(r))
        self.assertEqual(recipes[0], expected)

    def test_check_accepts_radeon_890m_recipe(self):
        # running cmd_check should not report our recipe as bad
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = lab.cmd_check(None)
            lines = printed.getvalue().splitlines()
        bad_890m = [line for line in lines if self.CARD in line]
        self.assertEqual(bad_890m, [], f"lab.py check reported errors for {self.CARD}: {bad_890m}")


if __name__ == "__main__":
    unittest.main()