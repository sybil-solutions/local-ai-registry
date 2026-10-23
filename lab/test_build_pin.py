import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lab

COMMIT = 'c' * 40
BUILD = {'id': 'fixture-4x', 'engine': 'fixture', 'image': None, 'backend': 'nvidia', 'port': 8888, 'entrypoint': ['serve'], 'args': [],
         'weights': [{'repo': 'test/raw', 'revision': 'b' * 40, 'at': '/models/raw', 'layout': 'dir'}], 'ctx': 1048576, 'seqs': 16,
         'machines': 4, 'build': {'repo': 'test/src', 'commit': COMMIT, 'dockerfile': 'Dockerfile'}}


class BuildPinTest(unittest.TestCase):
    def profile(self, ref, p=BUILD):
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'LAUNCHES', Path(d)):
            (Path(d) / f"{p['id']}.json").write_text(json.dumps(p))
            return lab.profile(ref)

    def test_a_build_launch_pins_to_its_commit(self):
        self.assertEqual(lab.pin(BUILD), COMMIT)

    def test_a_launch_without_image_or_build_commit_cannot_be_pinned(self):
        for p in ({**BUILD, 'build': None}, {**BUILD, 'build': {'repo': 'test/src'}}):
            with self.assertRaises(SystemExit):
                lab.pin(p)

    def test_the_commit_prefix_and_the_old_build_ref_are_both_accepted(self):
        for ref in ('fixture-4x@' + COMMIT[:12], 'fixture-4x@' + COMMIT, 'fixture-4x@build', 'fixture-4x'):
            self.assertEqual(self.profile(ref)['id'], 'fixture-4x', ref)

    def test_a_recipe_pinned_to_an_older_commit_must_be_rerun(self):
        with self.assertRaisesRegex(SystemExit, 'rerun the recipe'):
            self.profile('fixture-4x@' + 'd' * 12)

    def test_image_pins_are_unchanged(self):
        self.assertEqual(lab.pin({**BUILD, 'image': 'ghcr.io/test/engine@sha256:' + 'a' * 64}), 'a' * 64)

    def test_every_build_launch_in_the_registry_pins_and_keeps_its_recipes(self):
        launches = [p for p in (json.loads(f.read_text()) for f in sorted(lab.LAUNCHES.glob('*.json'))) if p.get('build') and not p.get('image')]
        self.assertTrue(launches)
        for p in launches:
            self.assertRegex(lab.pin(p), r'^[0-9a-f]{40}$', p['id'])

    def test_try_writes_a_recipe_for_a_build_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'launches').mkdir()
            (root / 'launches/fixture-4x.json').write_text(json.dumps(BUILD))
            args = argparse.Namespace(weights='test/raw@' + 'b' * 40, model='test-model', engine='fixture-4x', card='dgx-spark-gb10-128gb',
                                      set=[], dry_run=False, on='endpoint', endpoint='http://fixture.invalid', gpu='4x DGX Spark (GB10)', proxy_gpu=None)
            proof = {'gates': ' '.join(lab.GATES), 'tps': 30.1, 'prefill': 300, 'served': 'fixture'}
            evidence = {'ok': {g: True for g in lab.GATES}}
            with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', root / 'recipes'), patch.object(lab, 'RUNS', root / 'runs'), \
                    patch.object(lab, 'LAUNCHES', root / 'launches'), patch.object(lab, 'gates', return_value=(True, proof, evidence)), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(lab.cmd_try(args), 0)
            [out] = list((root / 'recipes').rglob('*.json'))
            self.assertEqual(json.loads(out.read_text())['engine'], 'fixture-4x@' + COMMIT[:12])
            self.assertTrue(out.name.endswith('.1024k.4x.json'), out.name)


if __name__ == '__main__':
    unittest.main()

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
