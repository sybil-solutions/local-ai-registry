import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk" / "python"))
import local_ai_registry as L


class CardDetectMobileSeparationTest(unittest.TestCase):
    def test_untested_laptop_gpus_do_not_match_desktop_cards(self):
        cat = json.loads((ROOT / "dist" / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(L.detect(cat, "NVIDIA GeForce RTX 5060", 8), "rtx-5060-8gb")
        self.assertIsNone(L.detect(cat, "NVIDIA GeForce RTX 5060 Laptop GPU", 8))
        self.assertEqual(L.detect(cat, "NVIDIA GeForce RTX 3070 Ti", 8), "rtx-3070-ti-8gb")
        self.assertIsNone(L.detect(cat, "NVIDIA GeForce RTX 3070 Ti Laptop GPU", 8))

    def test_laptop_cards_resolve_when_present_in_catalog(self):
        card_5060_m = json.loads(
            (ROOT / "registry" / "cards" / "nvidia" / "rtx-5060-laptop-8gb.json").read_text(encoding="utf-8")
        )
        card_3070ti_m = json.loads(
            (ROOT / "registry" / "cards" / "nvidia" / "rtx-3070-ti-laptop-8gb.json").read_text(encoding="utf-8")
        )
        cat = {
            "cards": {
                "rtx-5060-8gb": {"name": "GeForce RTX 5060", "vram_gb": 8, "match": {"names": ["rtx5060"]}},
                "rtx-5060-laptop-8gb": card_5060_m,
                "rtx-3070-ti-8gb": {"name": "GeForce RTX 3070 Ti", "vram_gb": 8, "match": {"names": ["rtx3070ti"]}},
                "rtx-3070-ti-laptop-8gb": card_3070ti_m,
            }
        }
        cases = [
            ("NVIDIA GeForce RTX 5060", 8, "rtx-5060-8gb"),
            ("NVIDIA GeForce RTX 5060 Laptop GPU", 8, "rtx-5060-laptop-8gb"),
            ("NVIDIA GeForce RTX 3070 Ti", 8, "rtx-3070-ti-8gb"),
            ("NVIDIA GeForce RTX 3070 Ti Laptop GPU", 8, "rtx-3070-ti-laptop-8gb"),
        ]
        for gpu, vram, expected in cases:
            self.assertEqual(L.detect(cat, gpu, vram), expected)

        js_out = subprocess.run(
            [
                "node",
                "--input-type=module",
                "-e",
                'import {readFileSync} from "node:fs"; import {detect} from "./sdk/js/index.js"; '
                'const {cat, cases} = JSON.parse(readFileSync(0, "utf8")); '
                'console.log(JSON.stringify(cases.map(([g, v]) => detect(cat, g, v))));',
            ],
            cwd=ROOT,
            input=json.dumps({"cat": cat, "cases": cases}),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(json.loads(js_out), [c[2] for c in cases])


if __name__ == "__main__":
    unittest.main()
