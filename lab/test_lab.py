import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lab

BUILD = {
    "engine": "dummy",
    "model": "dummy",
    "weights": "dummy@12345678",
    "ctx": 32768,
}


class TestLabProofBindingAndContextGate(unittest.TestCase):
    def test_check_rejects_inflated_ctx_low_tps_missing_log_or_forged_run_values(self):
        base = json.loads((lab.RECIPES / 'nvidia/rtx-3080-10gb/qwen3.6-35b-a3b.sglang.256k.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Use a path not in DRIFTED_HISTORICAL_RECIPES to verify launch hash binding
            out128 = root / 'recipes/nvidia/rtx-4090/qwen3.6-35b-a3b.sglang.128k.json'
            out256 = root / 'recipes/nvidia/rtx-4090/qwen3.6-35b-a3b.sglang.256k.json'
            out128.parent.mkdir(parents=True, exist_ok=True)
            # (a) modified ctx without re-proving launch_sha256 fails check
            out128.write_text(json.dumps({**base, 'card': 'rtx-4090', 'set': {'ctx': 131072}}), encoding='utf-8')
            with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', root / 'recipes'), \
                    patch.object(lab, 'kv_floor_exempt', return_value=set()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab.cmd_check(None), 1)
            out128.unlink()
            # (b) tps below MIN_TPS (15.0) fails check
            out256.write_text(json.dumps({**base, 'card': 'rtx-4090', 'proof': [{**base['proof'][0], 'tps': 9.0}]}), encoding='utf-8')
            with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', root / 'recipes'), \
                    patch.object(lab, 'kv_floor_exempt', return_value=set()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab.cmd_check(None), 1)
            # (c) missing log hash fails check
            out256.write_text(json.dumps({**base, 'card': 'rtx-4090', 'proof': [{k: v for k, v in base['proof'][0].items() if k != 'log'}]}), encoding='utf-8')
            with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', root / 'recipes'), \
                    patch.object(lab, 'kv_floor_exempt', return_value=set()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab.cmd_check(None), 1)
            # (d) forged tps that disagrees with the run record on disk fails check
            runs_dir = root / 'lab/runs'
            runs_dir.mkdir(parents=True, exist_ok=True)
            src_run = (lab.ROOT / 'data/lab-runs/rtx-3080-10gb.qwen3.6-35b-a3b.sglang-exl3-offload.256k.20261015T164302.json').read_text(encoding='utf-8')
            (runs_dir / 'run.json').write_text(src_run, encoding='utf-8')
            out256.write_text(json.dumps({**base, 'card': 'rtx-4090', 'proof': [{**base['proof'][0], 'tps': 9999.0}]}), encoding='utf-8')
            with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', root / 'recipes'), \
                    patch.object(lab, 'RUNS', runs_dir), patch.object(lab, 'kv_floor_exempt', return_value=set()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab.cmd_check(None), 1)

    def test_context_gate_places_code_early_and_records_counted_mode(self):
        prompts = []
        def fake_call(endpoint, path, body=None, timeout=1800):
            if path == '/v1/models':
                return {'data': [{'id': 'raw-' + 'b' * 8}]}, 0.01
            if path == '/v1/token/encode':
                return {'length': max(1, len((body or {}).get('text', '')) // 4)}, 0.01
            msgs = (body or {}).get('messages') or []
            text = '\n'.join(m.get('content') or '' for m in msgs)
            if 'vault' in text:
                prompts.append(text)
                kept = text[-512 * 4:]  # simulate a 512-token front-truncating server
                ans = '58213' if '58213' in kept else 'unknown'
                return {'choices': [{'message': {'content': ans, 'reasoning_content': 'ok'}, 'finish_reason': 'stop'}]}, 0.1
            return {'choices': [{'message': {'content': '391 17', 'reasoning_content': 'ok'}, 'finish_reason': 'stop'}]}, 0.01

        with patch.object(lab, 'call', side_effect=fake_call), \
                patch.object(lab, 'stream_rate', return_value=(30.0, 600, 20.0)):
            passed, proof, ev = lab.gates('http://fixture.invalid', {**BUILD, 'ctx': 32768})
        self.assertFalse(passed)
        self.assertFalse(ev['ok']['context'])
        self.assertEqual(ev['context'].get('counted'), 'tokenizer')
        self.assertLess(prompts[0].index('58213'), len(prompts[0]) // 3)


if __name__ == '__main__':
    unittest.main()
