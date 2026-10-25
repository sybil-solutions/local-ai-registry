import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import import_reported
import lab

CARD = 'cmp-170hx-64gb'
HOST = {'kind': 'host', 'command': ['llama-server', '--split-mode', 'layer'], 'port': 8080, 'ctx': 131072, 'backend': 'nvidia'}
GATES = 'load chat reasoning tools context speed'


KEYS = ('command', 'install', 'pip', 'env', 'port', 'weights', 'ctx', 'seqs', 'vision', 'wired_limit_reserve_mb')
old_pin = lambda p: hashlib.sha256(json.dumps({**{k: p.get(k) for k in KEYS}, 'config': lab.config_text(p)}, sort_keys=True).encode()).hexdigest()  # before #203
single = lambda p: p.get('cards', 1) == 1 and p.get('machines', 1) == 1


def recipe(engine='fixture@host'):
    return {'model': 'test-model', 'weights': 'test/raw@' + 'b'*40, 'engine': engine, 'set': {}, 'card': CARD,
            'proof': [{'gates': GATES, 'tps': 20.0}]}


IMPORT = {'format': 'exl3', 'model_id': 'test-model', 'model': 'Test', 'hf_repo': 'test/raw', 'hf_revision': 'b'*40, 'engine': 'vllm',
          'ctx': 131072, 'card': CARD, 'repo': 'test/src', 'commit': 'c'*40, 'port': 8000, 'args': []}


class HostTopologyTest(unittest.TestCase):
    def test_pin_counts_cards_and_machines(self):
        self.assertNotEqual(lab.pin(HOST), lab.pin({**HOST, 'cards': 2}))
        self.assertNotEqual(lab.pin(HOST), lab.pin({**HOST, 'machines': 2}))
        self.assertEqual(lab.pin(HOST), lab.pin({**HOST, 'cards': 1, 'machines': 1}))

    def assert_legacy_pins(self, launches):
        """Every single-machine host profile in `launches` without a pinned runtime keeps the pin from before cards,
        machines and runtime counted. One with a runtime pins it on purpose (lab/test_host_runtime.py RUNTIME_LAUNCHES):
        its pin differs, and only because of the runtime."""
        hosts = [p for p in (json.loads(f.read_text()) for f in sorted(launches.glob('*.json'))) if p.get('kind') == 'host']
        self.assertTrue(hosts)
        for p in hosts:
            if not single(p):
                continue
            if 'runtime' in p:
                self.assertNotEqual(lab.pin(p), old_pin(p), p.get('id'))
                self.assertEqual(lab.pin({k: v for k, v in p.items() if k != 'runtime'}), old_pin(p), p.get('id'))
            else:
                self.assertEqual(lab.pin(p), old_pin(p), p.get('id'))

    def test_existing_host_pins_unchanged(self):
        self.assert_legacy_pins(lab.LAUNCHES)

    def test_pin_compatibility_on_a_mixed_registry(self):
        # a registry holding a legacy single-card host, a new two-card one and a single-card one with a pinned runtime:
        # only the first keeps the old pin; the runtime one moves only by its runtime
        rt = {'archive': {'url': 'https://example.invalid/python.tar.gz', 'sha256': '1' * 64, 'into': 'rt'}, 'python': 'rt/python/bin/python3'}
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'LAUNCHES', Path(d)):
            for name, p in (('one', HOST), ('two', {**HOST, 'cards': 2}), ('rt', {**HOST, 'runtime': rt})):
                (Path(d) / f'{name}.json').write_text(json.dumps({'id': name, **p}))
            got = {p['id']: (single(p), lab.pin(p) == old_pin(p)) for p in (json.loads(f.read_text()) for f in sorted(Path(d).glob('*.json')))}
            self.assert_legacy_pins(Path(d))
        self.assertEqual(got, {'one': (True, True), 'rt': (True, False), 'two': (False, False)})
        moved = {**rt, 'archive': {**rt['archive'], 'sha256': '2' * 64}}
        self.assertNotEqual(lab.pin({**HOST, 'runtime': rt}), lab.pin({**HOST, 'runtime': moved}))  # the runtime itself counts
        self.assertEqual(lab.pin({'id': 'x', **HOST}), old_pin(HOST))  # a legacy profile, unchanged

    def test_explicit_single_topology_renders_the_same_launch(self):
        hashes = []
        for p in (HOST, {**HOST, 'cards': 1, 'machines': 1}):
            with patch.object(lab, 'profile', return_value=p):
                hashes.append(lab.launch_hash(lab.render(recipe())))
        self.assertEqual(hashes[0], hashes[1])

    def test_render_and_name_keep_cards(self):
        with patch.object(lab, 'profile', return_value={**HOST, 'cards': 2}):
            L = lab.render(recipe())
            self.assertEqual((L['cards'], L.get('machines')), (2, None))
            self.assertTrue(lab.recipe_path(recipe(), L).name.endswith('.128k.2x.json'))

    def test_variant_names_the_recipe(self):
        with patch.object(lab, 'profile', return_value={**HOST, 'cards': 1, 'variant': 'ram28'}):
            L = lab.render(recipe())
            self.assertEqual(L['variant'], 'ram28')
            self.assertTrue(lab.recipe_path(recipe(), L).name.endswith('.128k.ram28.json'))
        with patch.object(lab, 'profile', return_value={**HOST, 'cards': 1, 'variant': 'Bad/../x'}):
            with self.assertRaises(AssertionError):
                lab.recipe_path(recipe(), lab.render(recipe()))

    def test_check_rejects_host_on_several_machines(self):
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'ROOT', Path(d)), patch.object(lab, 'RECIPES', Path(d) / 'recipes'), \
                patch.object(lab, 'profile', return_value={**HOST, 'machines': 2}):
            r = recipe()
            out = lab.recipe_path(r, lab.render(r)); out.parent.mkdir(parents=True); out.write_text(json.dumps(r))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(lab.cmd_check(None), 1)
            self.assertIn('multi-machine host launches are not supported', buf.getvalue())

    def test_import_names_match_recipe_path(self):
        base = {'format': 'exl3', 'model_id': 'test-model', 'model': 'Test', 'hf_repo': 'test/raw', 'hf_revision': 'b'*40,
                'engine': 'vllm', 'ctx': 131072, 'card': CARD, 'repo': 'test/src', 'commit': 'c'*40, 'port': 8000, 'args': []}
        variants = [{**base, 'entrypoint': 'serve', 'machines': 1, 'gpus_per_machine': 2},  # a program on the host, two cards
                    {**base, 'image': 'ghcr.io/test/engine@sha256:' + 'a'*64, 'gpus_per_machine': 2}]  # a container, two cards
        for v in variants:
            out, prof, r = import_reported.freeze(v, 'test')
            with patch.object(lab, 'profile', return_value=prof):
                self.assertEqual(out, lab.recipe_path(r, lab.render(r)), v.get('image', 'host'))
            self.assertTrue(out.name.endswith('.2x.json'))

    def test_import_keeps_one_and_two_card_setups_apart(self):
        # one source, the same launch on one card and on two: each recipe keeps its own profile and file name
        host = {**IMPORT, 'entrypoint': 'serve'}
        box = {**IMPORT, 'model_id': 'test-box', 'image': 'ghcr.io/test/engine@sha256:' + 'a'*64}
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'ROOT', Path(d)), patch.object(lab, 'RECIPES', Path(d) / 'recipes'), \
                patch.object(lab, 'LAUNCHES', Path(d) / 'launches'), contextlib.redirect_stdout(io.StringIO()):
            (Path(d) / 'launches').mkdir()
            src = Path(d) / 'src' / 'test.json'; src.parent.mkdir()
            src.write_text(json.dumps([{**v, 'gpus_per_machine': g} for v in (host, box) for g in (1, 2)]))
            import_reported.main([str(src)])
            files = sorted((Path(d) / 'recipes').rglob('*.json'))
            self.assertEqual(len(files), 4)
            self.assertEqual(len(list((Path(d) / 'launches').glob('*.json'))), 4)
            for f in files:
                r = json.loads(f.read_text()); L = lab.render(r)
                self.assertEqual(f, lab.recipe_path(r, L))
                self.assertEqual(L['cards'], 2 if f.name.endswith('.2x.json') else 1, f.name)

    def test_import_refuses_two_launches_under_one_id(self):
        made = [import_reported.freeze({**IMPORT, 'entrypoint': 'serve', 'port': port}, 'test') for port in (8000, 8001)]
        made[1] = (made[1][0].with_name('other.json'), *made[1][1:])
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'LAUNCHES', Path(d)), \
                patch.object(import_reported, 'freeze', side_effect=made), patch.object(Path, 'read_text', return_value='[{}, {}]'):
            with self.assertRaises(SystemExit):
                import_reported.main(['src/test.json'])
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
