import json
import unittest

import lab
from export_plugin import build, exportable


class ExportFlagsTest(unittest.TestCase):
    def test_device_flag_launches_export(self):
        self.assertTrue(exportable({}))
        self.assertTrue(exportable({"flags": []}))
        self.assertTrue(exportable({"flags": ["--device /dev/kfd"]}))
        self.assertTrue(exportable({"flags": ["--device /dev/kfd", "--device /dev/dri"]}))

    def test_privileged_flag_launches_stay_out(self):
        for flag in ("--ipc host", "--security-opt seccomp=unconfined", "--ulimit memlock=-1:-1",
                     "--privileged", "--pid host", "--cap-add SYS_ADMIN", "--device /dev/mem"):
            self.assertFalse(exportable({"flags": ["--device /dev/kfd", flag]}), flag)

    def test_host_programs_and_multi_machine_stay_out(self):
        self.assertFalse(exportable({"kind": "host"}))
        self.assertFalse(exportable({"machines": 2}))
        self.assertFalse(exportable({"build": "x"}))


def recipe(path):
    return json.loads((lab.RECIPES / path).read_text(encoding="utf-8"))


class KvFloorTest(unittest.TestCase):
    def test_floor_is_128k_over_8gb(self):
        self.assertEqual(lab.KV_FLOOR_TOKENS, 131072)
        self.assertEqual(lab.KV_FLOOR_ABOVE_VRAM_GB, 8)

    def test_template_recipe(self):
        r = recipe("nvidia/rtx-3060-12gb/gemma-4-12b.tabbyapi.64k.json")  # 64k x 1 seq on 12 GB
        self.assertLess(lab.kv_tokens(r), lab.KV_FLOOR_TOKENS)
        self.assertTrue(lab.below_kv_floor(r))
        self.assertFalse(lab.below_kv_floor({**r, "card": "rtx-3060-ti-8gb"}))  # 8 GB is not over 8 GB
        self.assertFalse(lab.below_kv_floor({**r, "set": {**r["set"], "ctx": 131072}}))
        self.assertFalse(lab.below_kv_floor({**r, "set": {**r["set"], "ctx": 65536, "seqs": 2}}))  # 2 x (64k + 1k)

    def test_plugin_block_zero_kv_counts_context(self):
        r = recipe("intel/intel-arc-b390-64gb/qwen3.6-35b-a3b.llama.cpp.128k.json")  # kvTokens 0, ctxTokens 128k
        self.assertEqual(lab.profile(r["engine"])["plugin"]["serving"]["kvTokens"], 0)
        self.assertEqual(lab.kv_tokens(r), 131072)
        self.assertFalse(lab.below_kv_floor(r))
        r = recipe("amd/rx-6800-xt-16gb/qwen3.5-9b.llama.cpp.64k.json")  # 64k on 16 GB
        self.assertTrue(lab.below_kv_floor(r))

    def test_export_holds_the_floor(self):
        cat = json.loads((lab.ROOT / "dist" / "catalog.json").read_text(encoding="utf-8"))
        for version in (2, 3):
            out = json.loads(build(version))
            self.assertTrue(out["hardware"])
            for card, hw in out["hardware"].items():
                vram = cat["cards"][card]["vram_gb"] or 0
                for e in hw["recipes"]:
                    kv = e["serving"]["kvTokens"] or e["serving"]["ctxTokens"]
                    if vram * e["cards"] > lab.KV_FLOOR_ABOVE_VRAM_GB:
                        self.assertGreaterEqual(kv, lab.KV_FLOOR_TOKENS, f"v{version} {card} {e['id']}")


if __name__ == "__main__":
    unittest.main()
