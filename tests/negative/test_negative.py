"""Negative: malformed data and invalid references are refused with typed errors."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from lab.lab import card, profile, render


class NegativeTests(unittest.TestCase):
    def test_unknown_card_raises_typed_error(self):
        with self.assertRaises(SystemExit):
            card("no-such-card")

    def test_unknown_engine_raises(self):
        with self.assertRaises(FileNotFoundError):
            profile("no-such-engine@0000000000000000000000000000000000000000")

    def test_stale_engine_digest_is_refused(self):
        # a recipe pinned to a digest the profile no longer matches must not validate silently
        with self.assertRaises(SystemExit):
            profile("tabbyapi-exl3@0000000000000000000000000000000000000000")

    def test_malformed_weights_cannot_be_rendered(self):
        recipe = {
            "model": "qwen3.8-27b",
            "weights": "not-a-pinned-reference",
            "engine": "tabbyapi-exl3@0f83e6198dc3",
            "set": {},
            "card": "rtx-3070-ti-8gb",
            "proof": [],
        }
        with self.assertRaises((ValueError, SystemExit)):
            render(recipe)

    def test_recipe_with_extra_fields_is_rejected(self):
        temp = tempfile.TemporaryDirectory()
        try:
            from lab.lab import RECIPES, ROOT, RUNS, cmd_check
            recipes = Path(temp.name) / "registry/recipes"
            recipes.mkdir(parents=True)
            patcher = patch.multiple(
                lab_lab := __import__("lab.lab", fromlist=["x"]),
                RECIPES=recipes, ROOT=Path(temp.name),
                RUNS=Path(temp.name) / "lab/runs",
            )
            patcher.start()
            try:
                bad = {
                    "model": "qwen3.5-9b",
                    "weights": "TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938",
                    "engine": "tabbyapi-exl3@0f83e6198dc3",
                    "set": {},
                    "card": "rtx-3070-ti-8gb",
                    "bogus": "extra",
                    "proof": [{"at": "2026-10-20", "on": "vast", "gpu": "RTX 3070 Ti",
                               "gates": "load chat reasoning tools context speed",
                               "tps": 60.0, "prefill": 900, "served": "Qwen3.5-9B-exl3"}],
                }
                out = recipes / "nvidia" / "rtx-3070-ti-8gb" / "qwen3.5-9b.tabbyapi.64k.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(bad, separators=(",", ":")) + "\n", encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    code = cmd_check(None)
            finally:
                patcher.stop()
        finally:
            temp.cleanup()
        self.assertEqual(code, 1)
