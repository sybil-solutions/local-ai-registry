import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab import lab


class TestProofLogBinding(unittest.TestCase):
    def setUp(self):
        # Base recipe using tabbyapi where ctx overrides work via set
        self.base_recipe_path = lab.RECIPES / "nvidia" / "rtx-3070-ti-8gb" / "qwen3.5-9b.tabbyapi.64k.json"
        self.base_recipe = json.loads(self.base_recipe_path.read_text(encoding="utf-8"))
        launch = lab.render(self.base_recipe)
        self.base_recipe["proof"][0]["launch_sha256"] = lab.launch_hash(launch)

        self.sample_run_rec = {
            "recipe": self.base_recipe,
            "launch": launch,
            "evidence": {
                "ok": {g: True for g in lab.GATES},
                "context": {"prompt_tokens": int(launch["ctx"] * 0.85)},
            },
            "proof": self.base_recipe["proof"][0],
        }
        self.sample_run_raw = json.dumps(self.sample_run_rec, indent=2)
        valid_hash = "sha256:" + hashlib.sha256(self.sample_run_raw.encode("utf-8")).hexdigest()[:16]
        self.base_recipe["proof"][0]["log"] = valid_hash

    def _run_check_in_sandbox(self, recipe_dict, run_file_content=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rec_dir = root / "registry" / "recipes" / "nvidia" / "rtx-3070-ti-8gb"
            rec_dir.mkdir(parents=True, exist_ok=True)

            launch = lab.render(recipe_dict)
            rec_name = lab.recipe_path(recipe_dict, launch).name
            target_recipe = rec_dir / rec_name
            target_recipe.write_text(json.dumps(recipe_dict, separators=(",", ":")) + "\n", encoding="utf-8")

            runs_dir = root / "lab" / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            if run_file_content:
                (runs_dir / "run.json").write_text(run_file_content, encoding="utf-8")

            with patch.object(lab, "ROOT", root), \
                 patch.object(lab, "RECIPES", root / "registry" / "recipes"), \
                 patch.object(lab, "RUNS", runs_dir), \
                 patch.object(lab, "kv_floor_exempt", return_value=set()), \
                 contextlib.redirect_stdout(io.StringIO()):
                return lab.cmd_check(None)

    def test_check_rejects_inflated_ctx(self):
        # Reproduction (a) from Issue #334: inflating ctx without re-proving launch_sha256
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["set"]["ctx"] = 262144  # inflated from 64k to 256k
        rc = self._run_check_in_sandbox(tampered, run_file_content=self.sample_run_raw)
        self.assertEqual(rc, 1)

    def test_check_rejects_tps_below_min(self):
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["proof"][0]["tps"] = 12.0  # < MIN_TPS (15.0)
        rc = self._run_check_in_sandbox(tampered)
        self.assertEqual(rc, 1)

    def test_check_rejects_missing_log_hash(self):
        tampered = json.loads(json.dumps(self.base_recipe))
        del tampered["proof"][0]["log"]
        rc = self._run_check_in_sandbox(tampered)
        self.assertEqual(rc, 1)

    def test_check_rejects_invalid_log_hash_format(self):
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["proof"][0]["log"] = "md5:invalidhash1234"
        rc = self._run_check_in_sandbox(tampered)
        self.assertEqual(rc, 1)

    def test_check_rejects_unmatched_log_hash_anti_bypass(self):
        # Closing the exact bypass identified by @0xSero:
        # A well-formed but forged log hash (e.g. sha256:deadbeef...) must not bypass run comparison
        # when a run record for the recipe exists on disk.
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["proof"][0]["log"] = "sha256:deadbeef12345678"
        tampered["proof"][0]["tps"] = 9999.0
        rc = self._run_check_in_sandbox(tampered, run_file_content=self.sample_run_raw)
        self.assertEqual(rc, 1)

    def test_check_rejects_tampered_tps_against_run_record(self):
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["proof"][0]["tps"] = 9999.0  # valid log hash, but forged tps
        rc = self._run_check_in_sandbox(tampered, run_file_content=self.sample_run_raw)
        self.assertEqual(rc, 1)

    def test_check_rejects_tampered_served_against_run_record(self):
        tampered = json.loads(json.dumps(self.base_recipe))
        tampered["proof"][0]["served"] = "totally-different-model"
        rc = self._run_check_in_sandbox(tampered, run_file_content=self.sample_run_raw)
        self.assertEqual(rc, 1)

    def test_check_accepts_valid_recipe_bound_to_run_record(self):
        rc = self._run_check_in_sandbox(self.base_recipe, run_file_content=self.sample_run_raw)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
