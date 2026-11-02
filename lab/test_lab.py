import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lab

COMMIT = "22ef1303062e0f6d0b282440f8c1f685947f4938"
PROFILE = {
    "id": "tabbyapi-exl3",
    "engine": "tabbyapi",
    "image": "ghcr.io/test/tabbyapi:latest",
    "port": 8080,
    "entrypoint": ["serve"],
    "args": [],
    "defaults": {"ctx": 32768},
    "ctx": 32768,
    "seqs": 1,
    "backend": "nvidia",
    "cards": 1,
}


class TestVerifyEndpointServedWeights(unittest.TestCase):
    def test_check_served_weights_logic(self):
        w = f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}"
        # 1. Direct fraud (as reported in Issue #331): gpt-4o-mini
        ok, reason = lab.check_served_weights(w, "gpt-4o-mini")
        self.assertFalse(ok)
        self.assertIn("does not match", reason)

        # 2. Authentic tabbyapi served id carrying revision prefix
        ok, reason = lab.check_served_weights(w, f"Qwen3.5-9B-exl3-{COMMIT[:8]}")
        self.assertTrue(ok)
        self.assertIn("revision prefix", reason)

        # 3. Engines not exposing revision (e.g. sglang, flashnext, gguf paths) with explicit --served attestation
        ok, reason = lab.check_served_weights(w, "custom-flashnext-id", explicit_served="custom-flashnext-id")
        self.assertTrue(ok)
        self.assertIn("matches explicit served", reason)

        # 4. Explicit --served mismatch must be rejected
        ok, reason = lab.check_served_weights(w, "wrong-served-id", explicit_served="expected-served-id")
        self.assertFalse(ok)

        # 5. Baked into image
        ok, _ = lab.check_served_weights("baked-into-image", "any-id")
        self.assertTrue(ok)

    def test_try_on_endpoint_rejects_fraudulent_served_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "launches").mkdir()
            (root / "launches/tabbyapi-exl3.json").write_text(json.dumps(PROFILE))
            args = argparse.Namespace(
                weights=f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}",
                model="qwen3.5-9b",
                engine="tabbyapi-exl3",
                card="rtx-3070-8gb",
                set=[],
                dry_run=False,
                on="endpoint",
                endpoint="http://127.0.0.1:8099",
                gpu="NVIDIA GeForce RTX 4090",
                proxy_gpu=None,
                served=None,
            )
            proof = {"gates": " ".join(lab.GATES), "tps": 30.1, "prefill": 300, "served": "gpt-4o-mini"}
            evidence = {"ok": {g: True for g in lab.GATES}}
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "RUNS", root / "runs"), patch.object(lab, "LAUNCHES", root / "launches"),                     patch.object(lab, "gates", return_value=(True, proof, evidence)),                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = lab.cmd_try(args)
                self.assertEqual(ret, 1)
            # Verify no recipe was written
            recipes = list((root / "recipes").rglob("*.json"))
            self.assertEqual(len(recipes), 0)

    def test_try_on_endpoint_accepts_valid_served_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "launches").mkdir()
            (root / "launches/tabbyapi-exl3.json").write_text(json.dumps(PROFILE))
            args = argparse.Namespace(
                weights=f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}",
                model="qwen3.5-9b",
                engine="tabbyapi-exl3",
                card="rtx-3070-8gb",
                set=[],
                dry_run=False,
                on="endpoint",
                endpoint="http://127.0.0.1:8099",
                gpu="NVIDIA GeForce RTX 4090",
                proxy_gpu=None,
                served=None,
            )
            proof = {"gates": " ".join(lab.GATES), "tps": 30.1, "prefill": 300, "served": f"Qwen3.5-9B-exl3-{COMMIT[:8]}"}
            evidence = {"ok": {g: True for g in lab.GATES}}
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "RUNS", root / "runs"), patch.object(lab, "LAUNCHES", root / "launches"),                     patch.object(lab, "gates", return_value=(True, proof, evidence)),                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = lab.cmd_try(args)
                self.assertEqual(ret, 0)
            recipes = list((root / "recipes").rglob("*.json"))
            self.assertEqual(len(recipes), 1)

    def test_cmd_check_rejects_mismatched_served_model_in_proof(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rec_dir = root / "recipes/nvidia/rtx-3070-8gb"
            rec_dir.mkdir(parents=True)
            bad_recipe = {
                "model": "qwen3.5-9b",
                "weights": f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}",
                "engine": "tabbyapi-exl3@123456789012",
                "set": {},
                "card": "rtx-3070-8gb",
                "proof": [{
                    "at": "2026-10-20",
                    "on": "owner",
                    "gpu": "NVIDIA GeForce RTX 4090",
                    "launch_sha256": "dummy",
                    "gates": " ".join(lab.GATES),
                    "tps": 30.0,
                    "prefill": 300,
                    "served": "gpt-4o-mini",
                    "log": "sha256:dummy",
                }],
            }
            recipe_file = rec_dir / "qwen3.5-9b.tabbyapi.32k.json"
            recipe_file.write_text(json.dumps(bad_recipe))
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "kv_floor_exempt", return_value=set()),                     patch.object(lab, "render", return_value={"ctx": 32768, "seqs": 1, "backend": "nvidia", "cards": 1}),                     patch.object(lab, "recipe_path", return_value=recipe_file),                     contextlib.redirect_stdout(io.StringIO()):
                ret = lab.cmd_check(None)
                self.assertEqual(ret, 1)


    def test_try_on_endpoint_with_explicit_served_attestation_and_cmd_check_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "launches").mkdir()
            (root / "launches/tabbyapi-exl3.json").write_text(json.dumps(PROFILE))
            args = argparse.Namespace(
                weights=f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}",
                model="qwen3.5-9b",
                engine="tabbyapi-exl3",
                card="rtx-3070-8gb",
                set=[],
                dry_run=False,
                on="endpoint",
                endpoint="http://127.0.0.1:8099",
                gpu="NVIDIA GeForce RTX 4090",
                proxy_gpu=None,
                served="custom-unexposed-served-id",
            )
            proof = {"gates": " ".join(lab.GATES), "tps": 30.1, "prefill": 300, "served": "custom-unexposed-served-id"}
            evidence = {"ok": {g: True for g in lab.GATES}}
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "RUNS", root / "runs"), patch.object(lab, "LAUNCHES", root / "launches"),                     patch.object(lab, "gates", return_value=(True, proof, evidence)),                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = lab.cmd_try(args)
                self.assertEqual(ret, 0)
            recipes = list((root / "recipes").rglob("*.json"))
            self.assertEqual(len(recipes), 1)
            # Verify recipe proof recorded expected_served
            recipe_data = json.loads(recipes[0].read_text())
            self.assertEqual(recipe_data["proof"][0].get("expected_served"), "custom-unexposed-served-id")

            # Verify cmd_check passes this recipe because expected_served is recorded
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "kv_floor_exempt", return_value=set()),                     patch.object(lab, "render", return_value={"ctx": 32768, "seqs": 1, "backend": "nvidia", "cards": 1}),                     patch.object(lab, "recipe_path", return_value=recipes[0]),                     contextlib.redirect_stdout(io.StringIO()):
                chk = lab.cmd_check(None)
                self.assertEqual(chk, 0)

    def test_try_on_endpoint_rejects_when_explicit_served_mismatches_endpoint_served(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "launches").mkdir()
            (root / "launches/tabbyapi-exl3.json").write_text(json.dumps(PROFILE))
            args = argparse.Namespace(
                weights=f"TheMelonGod/Qwen3.5-9B-exl3@{COMMIT}",
                model="qwen3.5-9b",
                engine="tabbyapi-exl3",
                card="rtx-3070-8gb",
                set=[],
                dry_run=False,
                on="endpoint",
                endpoint="http://127.0.0.1:8099",
                gpu="NVIDIA GeForce RTX 4090",
                proxy_gpu=None,
                served="expected-served-name",
            )
            # Endpoint returned something else
            proof = {"gates": " ".join(lab.GATES), "tps": 30.1, "prefill": 300, "served": "different-served-name"}
            evidence = {"ok": {g: True for g in lab.GATES}}
            with patch.object(lab, "ROOT", root), patch.object(lab, "RECIPES", root / "recipes"),                     patch.object(lab, "RUNS", root / "runs"), patch.object(lab, "LAUNCHES", root / "launches"),                     patch.object(lab, "gates", return_value=(True, proof, evidence)),                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = lab.cmd_try(args)
                self.assertEqual(ret, 1)
            recipes = list((root / "recipes").rglob("*.json"))
            self.assertEqual(len(recipes), 0)


if __name__ == "__main__":
    unittest.main()
