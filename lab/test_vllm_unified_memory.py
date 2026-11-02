import unittest
import lab


class VllmUnifiedMemorySafetensorsTest(unittest.TestCase):
    def test_unified_memory_nvidia_vllm_defaults_to_eager_safetensors(self):
        recipe = {
            "model": "qwen3.8-27b-awq-bf16-int4",
            "weights": "cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4@" + "a" * 40,
            "engine": "vllm",
            "set": {"ctx": 262144},
            "card": "dgx-spark-gb10-128gb",
        }
        launch = lab.render(recipe)
        self.assertIn("--safetensors-load-strategy", launch["args"])
        idx = launch["args"].index("--safetensors-load-strategy")
        self.assertEqual(launch["args"][idx + 1], "eager")

    def test_explicit_safetensors_strategy_is_not_duplicated(self):
        recipe = {
            "model": "qwen3.8-27b-awq-bf16-int4",
            "weights": "cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4@" + "a" * 40,
            "engine": "vllm",
            "set": {"ctx": 262144, "extra": "--safetensors-load-strategy eager"},
            "card": "rtx-spark-n1x-128gb",
        }
        launch = lab.render(recipe)
        self.assertEqual(launch["args"].count("--safetensors-load-strategy"), 1)

    def test_discrete_nvidia_gpu_does_not_inject_eager_safetensors(self):
        recipe = {
            "model": "qwen3.8-27b-awq-bf16-int4",
            "weights": "cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4@" + "a" * 40,
            "engine": "vllm",
            "set": {"ctx": 262144},
            "card": "rtx-3090-24gb",
        }
        launch = lab.render(recipe)
        self.assertNotIn("--safetensors-load-strategy", launch["args"])


if __name__ == "__main__":
    unittest.main()
