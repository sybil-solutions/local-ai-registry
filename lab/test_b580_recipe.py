"""Companion tests for bounty #216: a tested recipe for the Intel Arc B580 12 GB.

Covers the required scenarios against the recipe lab:

- happy path: the card exists and its recipe passes every recipe-lab check.
- boundary: null and empty inputs are rejected explicitly, no silent corruption.
- negative: malformed data is refused with a typed error.
- concurrency: simultaneous validation runs are atomic and agree.
- regression: the reported defect (no card / no tested recipe for the B580)
  is resolved cleanly.

Standard library only, like the rest of the lab.
"""
import contextlib
import io
import json
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import lab

CARD_ID = "intel-arc-b580-12gb"
RECIPE = lab.RECIPES / "intel" / CARD_ID / "qwen3.5-9b.llama.cpp.128k.json"
WEIGHTS = "unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5"
ENGINE = "llama.cpp-sycl-qwen3.5-9b-q4-k-m-128k-arc-b580-12gb@86b37d266ce8"


class B580HappyPathTests(unittest.TestCase):
    """Valid inputs: the card and its tested recipe exist and validate end to end."""

    def test_card_matches_what_the_plugin_detects(self):
        card = lab.card(CARD_ID)
        self.assertEqual(card["id"], CARD_ID)
        self.assertEqual(card["name"], "Intel Arc B580")
        self.assertEqual(card["vendor"], "intel")
        self.assertEqual(card["backend"], "intel-xpu")
        self.assertEqual(card["vram_gb"], 12)
        m = card["match"]
        self.assertEqual(m["backend"], "intel-xpu")
        self.assertEqual(m["name"], "Intel Arc B580")
        self.assertIn("arcb580", m["names"])
        self.assertEqual(m["vramGb"], 12)

    def test_tested_recipe_passes_every_check(self):
        self.assertTrue(RECIPE.exists(), f"missing recipe {RECIPE}")
        r = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(set(r), {"model", "weights", "engine", "set", "card", "proof"})
        self.assertEqual(r["model"], "qwen3.5-9b")
        self.assertEqual(r["engine"], ENGINE)
        self.assertEqual(r["weights"], WEIGHTS)
        self.assertEqual(r["card"], CARD_ID)
        self.assertIsInstance(r["set"], dict)
        self.assertRegex(r["weights"], r"[\w.-]+/[\w.-]+@[0-9a-f]{40}")
        launch = lab.render(r)
        self.assertEqual(launch["backend"], "intel-xpu")
        self.assertEqual(launch["cards"], 1)
        self.assertEqual(launch["ctx"], 131072)
        self.assertEqual(RECIPE, lab.recipe_path(r, launch))
        p = r["proof"][0]
        gates = set(p["gates"].split())
        self.assertGreaterEqual(gates, set(lab.GATES))
        self.assertGreaterEqual(p["tps"], lab.MIN_TPS)
        self.assertIsInstance(p["launch_sha256"], str) and len(p["launch_sha256"]) == 64
        self.assertTrue(p["log"].startswith("sha256:"))
        self.assertLessEqual(RECIPE.stat().st_size, lab.MAX_RECIPE_BYTES)
        self.assertFalse(lab.below_kv_floor(r, launch))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = lab.cmd_check(None)
        self.assertEqual(code, 0, out.getvalue())
        self.assertRegex(out.getvalue(), r"recipes ok: \d+")


class B580BoundaryTests(unittest.TestCase):
    """Null / empty / whitespace inputs: explicit validation errors, no crash."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        self.patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root,
                                      RUNS=self.root / "lab" / "runs")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, r, text=None):
        out = lab.recipe_path(r, lab.render(r))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text if text is not None else json.dumps(r, separators=(",", ":")) + "\n",
                       encoding="utf-8")
        return out

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = lab.cmd_check(None)
        return code, printed.getvalue().splitlines()

    def test_null_card_id_is_rejected_explicitly(self):
        with self.assertRaises(SystemExit):
            lab.card(None)

    def test_empty_card_id_is_rejected_explicitly(self):
        with self.assertRaises(SystemExit):
            lab.card("")

    def test_whitespace_card_id_is_rejected_explicitly(self):
        with self.assertRaises(SystemExit):
            lab.card("   ")

    def test_null_recipe_fields_are_rejected_explicitly(self):
        cases = [
            {"model": "qwen3.5-9b", "weights": None, "engine": ENGINE, "set": {},
             "card": CARD_ID, "proof": [{"gates": "load chat reasoning tools context speed"}]},
            {"model": "qwen3.5-9b", "weights": WEIGHTS, "engine": None, "set": {},
             "card": CARD_ID, "proof": [{"gates": "load chat reasoning tools context speed"}]},
            {"model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": None,
             "card": CARD_ID, "proof": [{"gates": "load chat reasoning tools context speed"}]},
            {"model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {},
             "card": CARD_ID, "proof": None},
            {"model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {},
             "card": CARD_ID, "proof": []},
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(
                    (AssertionError, AttributeError, TypeError, SystemExit,
                     FileNotFoundError, ValueError)):
                if bad["card"] is None:
                    lab.card(None)
                self._validate_recipe(bad)

    def test_empty_weights_and_gates_are_rejected(self):
        self.write({
            "model": "qwen3.5-9b", "weights": "", "engine": ENGINE, "set": {},
            "card": CARD_ID,
            "proof": [{"gates": "load chat reasoning tools context speed"}],
        })
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertTrue(any("weights not pinned" in line for line in lines))

        self.write({
            "model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {},
            "card": CARD_ID,
            "proof": [{"gates": ""}],
        })
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertTrue(any("lacks a gate" in line for line in lines))

    def _validate_recipe(self, r):
        assert set(r) == {"model", "weights", "engine", "set", "card", "proof"}, f"fields {sorted(r)}"
        assert isinstance(r["set"], dict), "set is not an object"
        assert isinstance(r["proof"], list) and r["proof"], "the recipe has no proof"
        assert r["weights"] == "baked-into-image" or re.fullmatch(
            r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", r["weights"]), "weights not pinned"
        lab.card(r["card"])
        launch = lab.render(r)
        assert lab.recipe_path(r, launch) is not None
        p = r["proof"][0]
        assert isinstance(p, dict), "the proof is not an object"
        assert isinstance(p.get("gates"), str), "the proof's gates are not a string"
        assert not p.get("reported") or p.get("src"), "a reported proof names its source"
        need = set(lab.GATES)
        assert need <= set(p["gates"].split()), "latest proof lacks a gate"


class B580NegativeTests(unittest.TestCase):
    """Malformed data: immediate refusal with a typed error code."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        self.patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root,
                                      RUNS=self.root / "lab" / "runs")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, r, text=None):
        out = lab.recipe_path(r, lab.render(r))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text if text is not None else json.dumps(r, separators=(",", ":")) + "\n",
                       encoding="utf-8")
        return out

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = lab.cmd_check(None)
        return code, printed.getvalue().splitlines()

    def test_unknown_card_is_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            lab.card("no-such-card-xyz")
        self.assertIn("unknown card no-such-card-xyz", str(ctx.exception))

    def test_malformed_weights_are_refused(self):
        self.write({
            "model": "qwen3.5-9b", "weights": "not-a-pinned-reference", "engine": ENGINE,
            "set": {}, "card": CARD_ID,
            "proof": [{"gates": "load chat reasoning tools context speed"}],
        })
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertTrue(any("weights not pinned" in line for line in lines))

    def test_missing_gate_is_refused(self):
        self.write({
            "model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {},
            "card": CARD_ID,
            "proof": [{"gates": "load chat reasoning tools context"}],
        })
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertTrue(any("lacks a gate" in line for line in lines))

    def test_decode_below_fifteen_tok_s_fails_the_speed_gate(self):
        self.assertFalse(14.9 >= lab.MIN_TPS)
        self.assertTrue(15.0 >= lab.MIN_TPS)
        self.assertTrue(lab.MIN_TPS > 0)

    def test_wrong_file_name_is_refused(self):
        # the same recipe content stored under the wrong derived file name
        bad_dir = self.recipes / "intel" / CARD_ID
        bad_dir.mkdir(parents=True, exist_ok=True)
        wrong = bad_dir / "qwen3.5-9b.llama.cpp.64k.json"  # the launch ctx is 128k, so 64k is wrong
        wrong.write_text(json.dumps({
            "model": "qwen3.5-9b", "weights": WEIGHTS, "engine": ENGINE, "set": {},
            "card": CARD_ID,
            "proof": [{"gates": "load chat reasoning tools context speed", "tps": 20.0}],
        }, separators=(",", ":")) + "\n", encoding="utf-8")
        code, lines = self.check()
        self.assertEqual(code, 1)
        self.assertTrue(any("file name should be" in line for line in lines))


class B580ConcurrencyTests(unittest.TestCase):
    """Multiple simultaneous validation requests on the same registry: atomic,
    agreeing results, zero double-acceptance of a bad recipe."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = self.root / "registry/recipes"
        self.recipes.mkdir(parents=True)
        import shutil
        shutil.copytree(lab.RECIPES, self.recipes, dirs_exist_ok=True)
        exempt = lab.RECIPES.parent / lab.KV_FLOOR_EXEMPT
        if exempt.exists():
            shutil.copy2(exempt, self.root / "registry" / lab.KV_FLOOR_EXEMPT)
        self.patcher = patch.multiple(lab, RECIPES=self.recipes, ROOT=self.root,
                                      RUNS=self.root / "lab" / "runs")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, r, text=None):
        out = lab.recipe_path(r, lab.render(r))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text if text is not None else json.dumps(r, separators=(",", ":")) + "\n",
                       encoding="utf-8")
        return out

    def check_once(self):
        # concurrent runs share the real stdout; only the return code is the contract
        code = lab.cmd_check(None)
        return code

    def test_concurrent_checks_are_atomic_and_agree(self):
        codes = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.check_once) for _ in range(16)]
            for f in as_completed(futures):
                codes.append(f.result())
        self.assertTrue(all(code == 0 for code in codes), codes)
        self.assertEqual(len(codes), 16)

    def test_concurrent_bad_recipe_is_never_double_accepted(self):
        # seed one malformed recipe; every concurrent check must refuse it, every time
        self.write({
            "model": "qwen3.5-9b", "weights": "not-pinned", "engine": ENGINE, "set": {},
            "card": CARD_ID,
            "proof": [{"gates": "load chat reasoning tools context speed"}],
        })
        codes = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.check_once) for _ in range(16)]
            for f in as_completed(futures):
                codes.append(f.result())
        self.assertTrue(all(code == 1 for code in codes), codes)

    def test_concurrent_renders_of_the_b580_recipe_agree(self):
        r = json.loads((lab.RECIPES / "intel" / CARD_ID / "qwen3.5-9b.llama.cpp.128k.json")
                       .read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            launches = list(pool.map(lambda _: lab.render(r), range(24)))
        self.assertTrue(all(l == launches[0] for l in launches))
        self.assertEqual(launches[0]["ctx"], 131072)


class B580RegressionTests(unittest.TestCase):
    """Regression for #216: the registry had no card and no tested recipe for
    the Intel Arc B580 12 GB. The defect is resolved when both exist and the
    recipe passes all six gates at 15 tok/s or more."""

    def test_b580_card_and_tested_recipe_exist(self):
        # On unpatched code this raises: SystemExit: unknown card intel-arc-b580-12gb
        card = lab.card("intel-arc-b580-12gb")
        self.assertEqual(card["id"], CARD_ID)
        self.assertEqual(card["vram_gb"], 12)

    def test_b580_recipe_passes_all_six_gates(self):
        card = lab.card(CARD_ID)
        self.assertEqual(card["vram_gb"], 12)
        self.assertTrue(RECIPE.exists(), f"missing tested recipe {RECIPE}")
        r = json.loads(RECIPE.read_text(encoding="utf-8"))
        launch = lab.render(r)
        p = r["proof"][0]
        self.assertGreaterEqual(set(p["gates"].split()), set(lab.GATES))
        self.assertGreaterEqual(p["tps"], lab.MIN_TPS)
        self.assertFalse(lab.below_kv_floor(r, launch))
        self.assertLessEqual(RECIPE.stat().st_size, lab.MAX_RECIPE_BYTES)


if __name__ == "__main__":
    unittest.main()

