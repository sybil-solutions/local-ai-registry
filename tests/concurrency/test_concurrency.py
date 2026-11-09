"""Concurrency: repeated concurrent renders of the same recipe are deterministic."""
import json
import unittest
from concurrent.futures import ThreadPoolExecutor

from lab.lab import RECIPES, launch_hash, render

RECIPE_PATH = RECIPES / "amd" / "rx-7900-xtx-24gb" / "qwen3.8-27b.exllamav3.128k.full.json"


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_renders_are_identical(self):
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        expected = launch_hash(render(recipe))

        def render_once(_):
            return launch_hash(render(recipe))

        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = list(pool.map(render_once, range(64)))

        # every concurrent render re-derives the same pinned launch (no shared mutation)
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(hashes[0], expected)

    def test_repeated_renders_stable(self):
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        first = render(recipe)
        for _ in range(20):
            self.assertEqual(render(recipe), first)
