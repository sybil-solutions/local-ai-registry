import unittest
from lab.lab import check_served_weights


class TestCheckServedWeights(unittest.TestCase):
    def test_missing_or_empty_served_rejected(self):
        # 1. Fail-closed: rejection of empty or missing served ID, even for baked-into-image
        ok, reason = check_served_weights("baked-into-image", None)
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("baked-into-image", "")
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("baked-into-image", "   ")
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

        ok, reason = check_served_weights("Qwen/Qwen2.5-7B-Instruct@1234567890abcdef1234567890abcdef12345678", None)
        self.assertFalse(ok)
        self.assertIn("missing or empty", reason)

    def test_fixture_and_trivial_identifiers_rejected(self):
        # 2. Rejection of fixture/dummy/mock identifiers
        for forbidden in ["fixture", "dummy", "mock", "test", "none", "null", "default", "model", "local"]:
            ok, reason = check_served_weights("baked-into-image", forbidden)
            self.assertFalse(ok)
            self.assertIn("rejected fixture-like or trivial identifier", reason)

        # Less than 3 normalized chars rejected
        ok, reason = check_served_weights("baked-into-image", "ab")
        self.assertFalse(ok)
        self.assertIn("rejected fixture-like or trivial identifier", reason)

    def test_short_substring_injection_rejected(self):
        # 3. Short substring injections like "a" or "qwen" should not match when size or canonical tokens mismatch
        weights = "meta-llama/Llama-3.1-8B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights, "a")
        self.assertFalse(ok)

        ok, reason = check_served_weights(weights, "llama")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token", reason)

    def test_parameter_size_mismatch_rejected(self):
        # 4. Size/quant confusion (e.g. 0.5b vs 72b or 8b vs 70b)
        weights_72b = "Qwen/Qwen2.5-72B-Instruct-AWQ@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights_72b, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token '72b'", reason)

        weights_70b = "meta-llama/Llama-3.1-70B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights_70b, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertFalse(ok)
        self.assertIn("missing required parameter size token '70b'", reason)

    def test_valid_pinned_revision_prefix_accepted(self):
        # 5. Acceptance of valid pinned revision prefix (>= 7 chars)
        weights = "Qwen/Qwen2.5-7B-Instruct@abcdef0123456789abcdef0123456789abcdef01"
        ok, reason = check_served_weights(weights, "models--qwen--7b--abcdef012345")
        self.assertTrue(ok)
        self.assertIn("pinned revision prefix", reason)

    def test_valid_canonical_model_slug_accepted(self):
        # 6. Acceptance of canonical model slug and core tokens
        weights = "Qwen/Qwen2.5-7B-Instruct@1234567890abcdef1234567890abcdef12345678"
        ok, reason = check_served_weights(weights, "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(ok)
        self.assertIn("canonical model slug", reason)

        # Core tokens attested with different casing or separators
        ok, reason = check_served_weights(weights, "qwen2.5-7b-instruct-vllm")
        self.assertTrue(ok)

    def test_baked_into_image_valid_served(self):
        ok, reason = check_served_weights("baked-into-image", "deepseek-ai/DeepSeek-V3")
        self.assertTrue(ok)
        self.assertIn("baked-into-image verified", reason)

        # Explicit --served mismatch
        ok, reason = check_served_weights("baked-into-image", "deepseek-ai/DeepSeek-V3", explicit_served="llama-3")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
