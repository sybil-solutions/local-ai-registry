import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_plugin as export
import lab


class OffloadContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'dist').mkdir(); (self.root / 'registry').mkdir()
        (self.root / 'registry/models.json').write_text('{}', encoding="utf-8")
        self.profile = {'image': 'ghcr.io/example/engine@sha256:' + 'a'*64, 'ids': {'test': 'fixture'},
            'args': [], 'port': 8000, 'ctx': 8192, 'weights': [{'repo': 'test/raw', 'revision': 'b'*40, 'at': '/models/raw'}],
            'plugin': {'name': 'Fixture', 'family': 'test', 'format': 'EXL3', 'engine': 'vllm', 'servedName': 'fixture',
                'sizeGb': 1, 'minDriver': '', 'weights': [{'sizeGb': 1, 'layout': 'dir', 'mountPath': '/models/raw', 'dir': '', 'files': ''}],
                'scratch': None, 'serving': {'ctxTokens': 8192}, 'capabilities': {}}}
        self.prepare = {'at': '/models', 'args': ['prepare'], 'verifyArgs': ['verify-pack'], 'gpu': True, 'sizeGb': 3}
        self.resources = {'memoryBytes': 59055800320, 'memorySwapBytes': 59055800320, 'memlockUnlimited': True, 'ipcLock': True}

    def build(self, version, stale=False):
        with patch.object(lab, 'profile', return_value=self.profile), patch.object(export, 'ROOT', self.root):
            r = {'engine': 'fixture', 'card': 'test', 'model': 'test', 'proof': [{'gates': 'load chat reasoning tools context speed'}]}
            L = lab.render(r)
            r['proof'][0]['launch_sha256'] = '0'*64 if stale else hashlib.sha256(json.dumps(L, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            cat = {'cards': {'test': {'picks': ['test/fixture'], 'more': [], 'match': {}}}, 'recipes': {'test/fixture': {**r, 'launch': L}}}
            (self.root / 'dist/catalog.json').write_text(json.dumps(cat), encoding="utf-8")
            return json.loads(export.build() if version == 2 else export.build(version))

    def test_v2_excludes_prepare_only_and_resources_only(self):
        for key, value in [('prepare', self.prepare), ('resources', self.resources)]:
            with self.subTest(key=key):
                self.profile[key] = value
                self.assertEqual(self.build(2)['hardware'], {})
                del self.profile[key]

    def test_v3_preserves_exact_mounts_preparation_and_resources(self):
        self.profile.update(prepare=self.prepare, resources=self.resources)
        r = self.build(3)['hardware']['test']['recipes'][0]
        self.assertEqual(r['prepare'], self.prepare)
        self.assertEqual(r['launch']['resources'], self.resources)
        self.assertEqual(r['weights'][0]['mountPath'], '/models/raw')

    def test_changed_typed_launch_needs_new_acceptance(self):
        self.profile['resources'] = self.resources
        self.assertEqual(self.build(3, stale=True)['hardware'], {})

    def test_withdrawal_overrides_historical_acceptance(self):
        with patch.object(lab, 'profile', return_value=self.profile):
            r = {'engine': 'fixture', 'card': 'test', 'proof': [
                {'withdrawn': True}, {'gates': 'load chat reasoning tools context speed'}]}
            r['launch'] = lab.render(r)
            for version in (2, 3):
                self.assertFalse(export.eligible(r, version))

    def test_passing_try_requalifies_withdrawn_recipes_within_size_limit(self):
        withdrawn = [(f, json.loads(f.read_text(encoding="utf-8"))) for f in lab.RECIPES.rglob('*.json')
                     if json.loads(f.read_text(encoding="utf-8"))['proof'][0].get('withdrawn')]
        self.assertTrue(withdrawn)
        for source, recipe in withdrawn:
            with self.subTest(card=recipe['card']), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipes = root / 'registry/recipes'
                launch = lab.render(recipe)
                profile = {**lab.profile(recipe['engine']), 'defaults': {}}
                output = recipes / source.relative_to(lab.RECIPES)
                output.parent.mkdir(parents=True)
                output.write_text(json.dumps(recipe, separators=(',', ':')) + '\n', encoding="utf-8")
                args = argparse.Namespace(weights=recipe['weights'], model=recipe['model'],
                    engine=recipe['engine'], card=recipe['card'], set=[], dry_run=False,
                    on='endpoint', endpoint='http://fixture.invalid', gpu='RTX 4070 Ti SUPER', proxy_gpu=None)
                proof = {'gates': ' '.join(lab.GATES), 'tps': 30.1, 'prefill': 300,
                         'served': 'Qwen3.8-27B-EXL3-SC3bpw-H4-V4'}
                evidence = {'ok': {gate: True for gate in lab.GATES}}
                with patch.object(lab, 'ROOT', root), patch.object(lab, 'RECIPES', recipes), \
                     patch.object(lab, 'RUNS', root / 'lab/runs'), \
                     patch.object(lab, 'profile', return_value=profile), \
                     patch.object(lab, 'render', return_value=launch), \
                     patch.object(lab, 'gates', return_value=(True, proof, evidence)), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(lab.cmd_try(args), 0)
                    self.assertEqual(lab.cmd_check(None), 0)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(result['proof'][1:], recipe['proof'][:2])
                self.assertNotIn('withdrawn', result['proof'][0])
                self.assertLessEqual(output.stat().st_size, lab.MAX_RECIPE_BYTES)

    def test_input_mount_cannot_mask_output(self):
        self.profile['prepare'] = {**self.prepare, 'at': '/models/raw/pack'}
        with self.assertRaisesRegex(AssertionError, 'masks'):
            self.build(3)

    def test_resource_contract_rejects_unbounded_or_unknown_options(self):
        for resource in [{'privileged': True}, {'memoryBytes': True}, {'memorySwapBytes': 1}, {'memoryBytes': 2, 'memorySwapBytes': 1}]:
            with self.subTest(resource=resource), self.assertRaises(AssertionError):
                self.profile['resources'] = resource
                self.build(3)

    def test_prepare_requires_verification_and_explicit_gpu_budget(self):
        self.profile['prepare'] = {'at': '/models', 'args': ['prepare']}
        with self.assertRaisesRegex(AssertionError, 'prepare requires'):
            self.build(3)

    def test_plugin_metadata_cannot_change_accepted_inputs(self):
        self.profile['prepare'] = self.prepare
        for key, value in [('mountPath', '/other'), ('dir', 'other'), ('files', 'partial.safetensors'), ('repository', 'other/repo'), ('revision', 'd'*40), ('layout', 'hub')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                original = dict(self.profile['plugin']['weights'][0])
                self.profile['plugin']['weights'][0][key] = value
                try:
                    self.build(3)
                finally:
                    self.profile['plugin']['weights'][0] = original

    def test_nul_or_nonstring_environment_cannot_inject_docker_options(self):
        self.profile['resources'] = self.resources
        for value in ['hello\0--privileged', 1]:
            with self.subTest(value=value), self.assertRaisesRegex(AssertionError, 'environment'):
                self.profile['env'] = {'NOTE': value}
                self.build(3)

    def test_intel_debug_environment_does_not_hide_validated_recipe(self):
        self.profile.update(resources=self.resources, env={
            'NEOReadDebugKeys': '1',
            'TreatNonUsmForTransfersAsSharedSystem': '0',
            'EnableSharedSystemUsmSupport': '0'})
        self.assertIn('test', self.build(3)['hardware'])

    def test_malformed_environment_names_cannot_be_exported(self):
        self.profile['resources'] = self.resources
        for key in ['1NOTE', 'NOTE --privileged', 'NOTE=other', 'NOTE\nother']:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.profile['env'] = {key: '1'}
                self.build(3)

    def test_export_cannot_truncate_input_mounts(self):
        self.profile['plugin']['weights'] = []
        with self.assertRaisesRegex(AssertionError, 'input mounts'):
            self.build(3)


if __name__ == '__main__':
    unittest.main()
