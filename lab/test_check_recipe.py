"""A malformed recipe is one bad recipe: the check reports it and goes on with the rest."""
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


def recipe(model="qwen3.5-9b", **kw):
    r = {"model": model, "weights": WEIGHTS, "engine": ENGINE, "set": {}, "card": "rtx-3070-ti-8gb",
         "proof": [{"at": "2026-10-20", "on": "vast", "gpu": "RTX 3070 Ti", "gates": "load chat reasoning tools context speed",
                    "tps": 60.0, "prefill": 900, "served": "Qwen3.5-9B-exl3-22ef1303"}]}
    r.update(kw)
    return r


class CheckRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root, RUNS=self.root / "lab/runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, r, text=None):
        good = {**r, "set": {}} if not isinstance(r.get("set"), dict) else r
        out = lab.recipe_path(good, lab.render(good))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text if text is not None else json.dumps(r, separators=(",", ":")) + "\n")
        return out

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = lab.cmd_check(None)
        return code, printed.getvalue().splitlines()

    def test_a_malformed_recipe_is_reported_and_the_rest_is_checked(self):
        self.write(recipe())
        self.write(recipe("qwen3.8-27b", proof=[]))
        self.write(recipe("gemma-4-12b", proof=[None]))
        self.write(recipe("qwen3.6-35b", proof=[{"gates": 5}]))
        self.write(recipe("qwen3.8-flash-next", set="ctx=1"))
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 4)  # one line per bad recipe, and the good one is not among them
        self.assertTrue(all("qwen3.5-9b" not in line for line in lines))

    def test_a_recipe_that_is_not_json_is_reported(self):
        self.write(recipe(), text="{not json\n")
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 1)
        self.assertIn("not a JSON recipe", lines[0])


if __name__ == "__main__":
    unittest.main()
