import unittest
from unittest.mock import patch

import lab


class DiffusionSamplingTest(unittest.TestCase):
    def test_identifies_diffusion_models(self):
        self.assertTrue(lab.is_diffusion_model("diffusiongemma-26b-a4b"))
        self.assertTrue(lab.is_diffusion_model("diffusiongemma-26B-A4B-it-f7f5b7f5"))
        self.assertFalse(lab.is_diffusion_model("qwen3.5-9b"))

    def test_strips_unsupported_sampling_params_for_diffusion(self):
        payload = {
            "model": "diffusiongemma-26B-A4B-it-f7f5b7f5",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.6,
            "min_p": 0.05,
            "seed": 42,
            "min_tokens": 1,
            "logit_bias": {},
            "bad_words": ["x"],
            "allowed_token_ids": [1, 2],
            "stream": True,
        }
        cleaned = lab.strip_diffusion_sampling(payload)
        for forbidden in lab.DIFFUSION_UNSUPPORTED_SAMPLING:
            self.assertNotIn(forbidden, cleaned)
        self.assertTrue(cleaned["stream"])

    def test_chat_omits_temperature_for_diffusion_and_keeps_it_for_standard_models(self):
        sent_bodies = []

        def fake_call(endpoint, path, body=None, timeout=3600):
            sent_bodies.append(body)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}, 0.1

        with patch.object(lab, "call", side_effect=fake_call):
            lab.chat("http://localhost:8000", "diffusiongemma-26B-A4B-it-f7f5b7f5", [{"role": "user", "content": "hi"}])
            lab.chat("http://localhost:8000", "qwen3.5-9b", [{"role": "user", "content": "hi"}])

        self.assertNotIn("temperature", sent_bodies[0])
        self.assertEqual(sent_bodies[1].get("temperature"), 0.6)


if __name__ == "__main__":
    unittest.main()
