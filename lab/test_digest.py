import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import digest

class DigestTest(unittest.TestCase):
    def test_historical_and_org_pins_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            (root / "plugin/v2").mkdir(parents=True)
            (root / "plugin/v2/recipes.json").write_text("{}", encoding="utf-8")
            (root / "registry/test.json").write_text(json.dumps({"images": [
                "ghcr.io/0xsero/gateway@sha256:" + "a" * 64,
                "ghcr.io/sybil-solutions/gateway:new@sha256:" + "b" * 64]}), encoding="utf-8")
            with patch.object(digest, "ROOT", root):
                images, weights = digest.pins()
            self.assertEqual(images, {("0xsero/gateway", "a" * 64), ("sybil-solutions/gateway", "b" * 64)})
            self.assertEqual(weights, set())

    def test_owner_delivery_items_are_not_hidden(self):
        item = {"number": 1, "user": {"login": "0xSero", "type": "User"},
                "created_at": "2026-10-07T00:00:00Z", "title": "D141 acceptance", "html_url": "https://example.invalid/1"}
        def gh(path):
            if "/actions/" in path: return {"workflow_runs": []}
            return [item] if "issues?" in path else []
        with patch.object(digest, "gh", side_effect=gh):
            result = digest.community({"dsv41-flash-offload": "main"})
        self.assertTrue(any("D141 acceptance" in line and "owner delivery item" in line for line in result))

    def test_image_check_preserves_the_captured_namespace(self):
        with patch.object(digest, "fetch", side_effect=[(200, {"token": "test"}), (200, None)]) as fetch:
            self.assertIsNone(digest.image_missing(("sybil-solutions/gateway", "a" * 64)))
        self.assertIn("scope=repository:sybil-solutions/gateway:pull", fetch.call_args_list[0].args[0])
        self.assertIn("/v2/sybil-solutions/gateway/manifests/", fetch.call_args_list[1].args[0])

if __name__ == "__main__": unittest.main()
