"""The optional pinned host runtime (docs/design.md, "A pinned host runtime"): validation, pins, rendered steps, and the
steps themselves run in a sandbox with real shasum, tar, mktemp and pip and stand-ins for curl (no network). The whole
one-script command from both SDKs is run under bash, bash -e and sh."""
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import lab

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'sdk' / 'python'))
import local_ai_registry as sdk  # noqa: E402

REV = '0123456789abcdef0123456789abcdef01234567'
ARCHIVE = 'cpython-3.12.15+20261021-aarch64-apple-darwin-install_only.tar.gz'
ARCHIVE_URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20261021/' + ARCHIVE.replace('+', '%2B')
WHEEL_URL = 'https://files.example.invalid/packages/'
KEYS = ('command', 'install', 'pip', 'env', 'port', 'weights', 'ctx', 'seqs', 'vision', 'wired_limit_reserve_mb')
SHAS = ('1' * 64, '2' * 64, '3' * 64, '4' * 64)
SERVER = 'open("SERVER_STARTED", "w").write("ok")\n'


# Launches that adopted the pinned host runtime on purpose, each qualified afresh on it. Every other launch must not use
# the new fields, and its pin must stay exactly what it was before them.
RUNTIME_LAUNCHES = {'mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1'}


def before(p):
    """The pin before `runtime` counted (cards and machines count only when not 1)."""
    topology = {k: p[k] for k in ('cards', 'machines') if p.get(k, 1) != 1}
    return hashlib.sha256(json.dumps({**{k: p.get(k) for k in KEYS}, **topology, 'config': lab.config_text(p)}, sort_keys=True).encode()).hexdigest()


def wheel(name, version, sha):
    return {'name': name, 'version': version, 'url': f"{WHEEL_URL}{name.replace('-', '_')}-{version}-py3-none-any.whl", 'sha256': sha}


def runtime(shas=SHAS, root='rt'):
    return {'archive': {'url': ARCHIVE_URL, 'sha256': shas[0], 'into': root}, 'python': f'{root}/python/bin/python3',
            'packages': {'dir': f'{root}/wheels', 'target': f'{root}/site',
                         'wheels': [wheel('fixture-a', '1.0', shas[1]), wheel('fixture-b', '2.0', shas[2]), wheel('huggingface-hub', '1.0', shas[3])]}}


def profile(rt=None, root='rt', launcher='launcher'):
    """A host profile shaped like the Apple MLX ones: a pinned runtime, Hugging Face cache weights and a launcher."""
    return {'id': 'fixture-runtime', 'kind': 'host', 'engine': 'mlx-lm', 'backend': 'metal', 'port': 8080, 'ctx': 32768, 'seqs': 1,
            'vision': False, 'install': f'Task runtime: {root}/ (see runtime)', 'runtime': runtime(root=root) if rt is None else rt,
            'command': [f'{root}/python/bin/python3', '-I', f'{launcher}/serve.py', '--port', '8080'],
            'env': {'PYTHONPATH': f'{root}/site', 'HF_HUB_CACHE': f'{root}/hf-cache', 'HF_HUB_OFFLINE': '1'},
            'weights': [{'repo': 'org/model', 'revision': REV, 'at': f'{root}/hf-cache', 'layout': 'hub', 'files': None, 'ref': 'main'}],
            'config': {'at': f'{launcher}/serve.py', 'text': SERVER}}


def launch(rt=None, **kw):
    """The rendered launch, as the SDKs see it in dist/catalog.json."""
    p = profile(rt, **kw)
    with patch.object(lab, 'profile', return_value=p):
        return lab.render({'model': 'test-model', 'weights': 'org/model@' + REV, 'engine': 'fixture-runtime@host', 'set': {}, 'card': 'apple-m1-8gb'})


def edit(path, value):
    """A launch whose value at `path` (keys and indexes) is replaced; `...` deletes it."""
    def make():
        L = launch()
        *head, last = path
        node = L
        for k in head:
            node = node[k]
        if value is ...:
            del node[last]
        else:
            node[last] = value
        return L
    return make


def without_runtime(change):
    def make():
        L = launch()
        del L['runtime']
        change(L)
        return L
    return make


DIR_WEIGHTS = {'repo': 'org/other', 'revision': 'f' * 40, 'at': 'models/other', 'layout': 'dir', 'files': None}

# Rejected by lab.py check and by both SDKs (sdk/test.sh runs this table through the JS SDK too).
BAD = {
    'archive sha256 too short': edit(['runtime', 'archive', 'sha256'], 'ab' * 31),
    'archive sha256 upper case': edit(['runtime', 'archive', 'sha256'], 'A' * 64),
    'archive sha256 missing': edit(['runtime', 'archive', 'sha256'], ...),
    'archive extra key': edit(['runtime', 'archive', 'strip'], 1),
    'archive over http': edit(['runtime', 'archive', 'url'], ARCHIVE_URL.replace('https', 'http')),
    'archive url with a query': edit(['runtime', 'archive', 'url'], ARCHIVE_URL + '?x=1'),
    'archive url with a fragment': edit(['runtime', 'archive', 'url'], ARCHIVE_URL + '#sha256=' + '1' * 64),
    'archive url with credentials': edit(['runtime', 'archive', 'url'], 'https://user@github.com/a/' + ARCHIVE),
    'archive url with ;': edit(['runtime', 'archive', 'url'], 'https://github.com/a;rm/' + ARCHIVE),
    'archive url with $()': edit(['runtime', 'archive', 'url'], 'https://github.com/$(id)/' + ARCHIVE),
    'archive url with a quote': edit(['runtime', 'archive', 'url'], "https://github.com/a'b/" + ARCHIVE),
    'archive is a zip': edit(['runtime', 'archive', 'url'], ARCHIVE_URL.replace('.tar.gz', '.zip')),
    'archive file decodes to a path': edit(['runtime', 'archive', 'url'], 'https://github.com/a/..%2F..%2F' + ARCHIVE),
    'archive file decodes to a space': edit(['runtime', 'archive', 'url'], 'https://github.com/a/x%20' + ARCHIVE),
    'archive file decodes to a quote': edit(['runtime', 'archive', 'url'], 'https://github.com/a/x%27' + ARCHIVE),
    'into is absolute': edit(['runtime', 'archive', 'into'], '/usr/local'),
    'into is ..': edit(['runtime', 'archive', 'into'], '..'),
    'into climbs': edit(['runtime', 'archive', 'into'], 'rt/../../x'),
    'into has a . part': edit(['runtime', 'archive', 'into'], 'rt/./x'),
    'into has ;': edit(['runtime', 'archive', 'into'], 'rt;id'),
    'into is empty': edit(['runtime', 'archive', 'into'], ''),
    'python is absolute': edit(['runtime', 'python'], '/usr/bin/python3'),
    'python has $()': edit(['runtime', 'python'], '$(id)/python3'),
    'python is a bare name (PATH lookup)': edit(['runtime', 'python'], 'python3'),
    'python outside the archive': edit(['runtime', 'python'], 'other/bin/python3'),
    'python is the archive dir itself': edit(['runtime', 'python'], 'rt'),
    'python without an archive': edit(['runtime', 'archive'], ...),
    'packages without python': edit(['runtime', 'python'], ...),
    'packages extra key': edit(['runtime', 'packages', 'index'], 'https://pypi.org/simple'),
    'packages without target': edit(['runtime', 'packages', 'target'], ...),
    'target is absolute': edit(['runtime', 'packages', 'target'], '/Library/Python'),
    'dir climbs': edit(['runtime', 'packages', 'dir'], 'rt/../..'),
    'dir inside target': edit(['runtime', 'packages', 'dir'], 'rt/site/wheels'),
    'target inside dir': edit(['runtime', 'packages', 'target'], 'rt/wheels/site'),
    'dir is target': edit(['runtime', 'packages', 'dir'], 'rt/site'),
    'no wheels': edit(['runtime', 'packages', 'wheels'], []),
    'wheels not a list': edit(['runtime', 'packages', 'wheels'], {'a': 1}),
    'wheel extra key': edit(['runtime', 'packages', 'wheels', 0, 'extras'], 'cli'),
    'wheel without sha256': edit(['runtime', 'packages', 'wheels', 0, 'sha256'], ...),
    'wheel sha256 short': edit(['runtime', 'packages', 'wheels', 0, 'sha256'], '1' * 63),
    'wheel over http': edit(['runtime', 'packages', 'wheels', 0, 'url'], 'http://files.example.invalid/fixture_a-1.0-py3-none-any.whl'),
    'wheel file is a source archive': edit(['runtime', 'packages', 'wheels', 0, 'url'], WHEEL_URL + 'fixture_a-1.0.tar.gz'),
    'wheel file names another project': edit(['runtime', 'packages', 'wheels', 0, 'name'], 'other'),
    'wheel file names another version': edit(['runtime', 'packages', 'wheels', 0, 'version'], '1.1'),
    'wheel name has a space': edit(['runtime', 'packages', 'wheels', 0, 'name'], 'fixture a'),
    'wheel version has ;': edit(['runtime', 'packages', 'wheels', 0, 'version'], '1.0;id'),
    'wheel listed twice': edit(['runtime', 'packages', 'wheels', 1], wheel('fixture-a', '1.0', '2' * 64)),
    'wheel listed twice under another spelling': edit(['runtime', 'packages', 'wheels', 1], wheel('Fixture_A', '1.0', '2' * 64)),
    'weights without a pinned huggingface_hub': edit(['runtime', 'packages', 'wheels', 2], wheel('fixture-c', '1.0', '4' * 64)),
    'weights with a runtime that has no packages': edit(['runtime', 'packages'], ...),
    'directory weights with a runtime': edit(['weights', 0], DIR_WEIGHTS),
    'hub and directory weights mixed': without_runtime(lambda L: L['weights'].append(DIR_WEIGHTS)),
    'launcher path is absolute': edit(['config', 'at'], '/tmp/serve.py'),
    'launcher path climbs': edit(['config', 'at'], '../serve.py'),
    'runtime extra key': edit(['runtime', 'post'], 'echo hi'),
    'runtime empty': edit(['runtime'], {}),
    'runtime not an object': edit(['runtime'], 'rt/python/bin/python3'),
    'runtime with pip': edit(['pip'], ['mlx-lm==0.31.3']),
    'ref without layout hub': edit(['weights', 0, 'layout'], 'dir'),
    'ref is a path': edit(['weights', 0, 'ref'], 'heads/main'),
    'ref climbs': edit(['weights', 0, 'ref'], '../main'),
    'ref has $()': edit(['weights', 0, 'ref'], '$(id)'),
    'hub revision is a branch': edit(['weights', 0, 'revision'], 'main'),
    'hub cache dir is absolute': edit(['weights', 0, 'at'], '/Users/someone/.cache/huggingface/hub'),
    'hub cache dir has ;': edit(['weights', 0, 'at'], 'rt;id'),
    'hub repo has ;': edit(['weights', 0, 'repo'], 'org/model;id'),
    'hub files climb': edit(['weights', 0, 'files'], ['../config.json']),
}



def paths(root='rt', runtime=True, **kv):
    """The fixture with some destinations moved: target, dir, at (every hub cache) or config (the launcher)."""
    def make():
        L = launch(root=root)
        if not runtime:
            del L['runtime']
        for k, v in kv.items():
            if k in ('target', 'dir'):
                L['runtime']['packages'][k] = v
            elif k == 'at':
                for w in L['weights']:
                    w['at'] = v
            else:
                L['config']['at'] = v
        return L
    return make


# Destinations that collide (equal, or one inside the other, both ways); with root 'rt' the interpreter's archive entry is
# rt/python, and 'opt/rt' lets a destination hold runtime.archive.into.
COLLIDE = {
    'target is into (dir elsewhere)': paths(dir='wheels', target='rt'),
    'target holds into': paths('opt/rt', target='opt', dir='w/wheels'),
    'target is the interpreter entry': paths(target='rt/python'),
    'target inside the interpreter entry': paths(target='rt/python/site'),
    'dir is into': paths(dir='rt', target='site'),
    'dir holds into': paths('opt/rt', dir='opt', target='s/site'),
    'dir is the interpreter entry': paths(dir='rt/python'),
    'dir inside the interpreter entry': paths(dir='rt/python/wheels'),
    'cache is into': paths(at='rt'),
    'cache holds into': paths('opt/rt', at='opt'),
    'cache is the interpreter entry': paths(at='rt/python'),
    'cache inside the interpreter entry': paths(at='rt/python/hf'),
    'launcher is into': paths(config='rt'),
    'launcher holds into': paths('opt/rt', config='opt'),
    'launcher inside the interpreter entry': paths(config='rt/python/serve.py'),
    'target is cache': paths(at='rt/site'),
    'cache inside target': paths(at='rt/site/hf'),
    'target inside cache': paths(target='rt/hf-cache/site'),
    'dir is cache': paths(at='rt/wheels'),
    'cache inside dir': paths(at='rt/wheels/hf'),
    'dir inside cache': paths(dir='rt/hf-cache/wheels'),
    'launcher is target': paths(config='rt/site'),
    'launcher inside target': paths(config='rt/site/serve.py'),
    'target inside launcher': paths(target='launcher/serve.py/site'),
    'launcher is dir': paths(config='rt/wheels'),
    'launcher inside dir': paths(config='rt/wheels/serve.py'),
    'dir inside launcher': paths(dir='launcher/serve.py/wheels'),
    'launcher is cache': paths(config='rt/hf-cache'),
    'launcher inside cache': paths(config='rt/hf-cache/serve.py'),
    'cache inside launcher': paths(at='launcher/serve.py/hf'),
    'launcher inside cache, no runtime': paths(runtime=False, config='rt/hf-cache/serve.py'),
}
BAD.update({f'collision: {k}': v for k, v in COLLIDE.items()})


def two_hub(first, second, runtime=True):
    """The fixture with two hub weights entries: the first is the fixture's own with `first` applied, then `second`."""
    def make():
        L = launch()
        if not runtime:
            del L['runtime']
        w = L['weights'][0]
        L['weights'] = [{**w, **first}, {**w, 'ref': 'main', **second}]
        return L
    return make


DRAFT = {'repo': 'org/draft', 'revision': 'c' * 40}
# r4: hub caches may be shared only when identical, and a repo may appear once per cache (each publish replaces its
# whole model directory, so a nested cache or a second entry would erase the other)
CACHES = {
    'second hub cache inside the first (in its model dir)': two_hub({}, {**DRAFT, 'at': 'rt/hf-cache/models--org--model/blobs'}),
    'first hub cache inside the second': two_hub({'at': 'rt/hf-cache/inner'}, {**DRAFT, 'at': 'rt/hf-cache'}),
    'hub caches nested, differing in case': two_hub({'at': 'rt/HF-cache'}, {**DRAFT, 'at': 'rt/hf-cache/inner'}),
    'hub caches nested, no runtime': two_hub({}, {**DRAFT, 'at': 'rt/hf-cache/models--org--model/blobs'}, runtime=False),
    'same repo twice in one cache, two revisions': two_hub({'ref': 'v1'}, {'revision': 'd' * 40, 'ref': 'v2'}),
    'same repo twice in one cache, file subsets': two_hub({'files': ['config.json']}, {'files': ['model.safetensors']}),
    'same repo twice in one cache, repo case differs': two_hub({}, {'repo': 'Org/Model', 'revision': 'd' * 40}),
    'same repo twice, the cache path differs in case': two_hub({}, {'at': 'rt/HF-cache', 'revision': 'd' * 40}),
    'same repo twice in one cache, no runtime': two_hub({}, {'revision': 'd' * 40}, runtime=False),
    'macos is a string': edit(['runtime', 'macos'], '15'),
    'macos is a bool': edit(['runtime', 'macos'], True),
    'macos below 11': edit(['runtime', 'macos'], 10),
    'macos above 99': edit(['runtime', 'macos'], 100),
    'macos is a dotted version': edit(['runtime', 'macos'], '15.0'),
    'macos is null': edit(['runtime', 'macos'], None),
}
BAD.update({f'cache: {k}': v for k, v in CACHES.items()})
# r5: a hub launch without a runtime is confined too, so its launcher path is checked like any other
BAD.update({
    'hub only: launcher path is absolute': paths(runtime=False, config='/tmp/local-ai-r5/serve.py'),
    'hub only: launcher path climbs': paths(runtime=False, config='../serve.py'),
    'hub only: launcher path holds the cache': paths(runtime=False, config='rt'),
})

# Rejected by lab.py check; the SDKs never render these fields for a container, so there is nothing to refuse there.
BAD_LAB = {
    'runtime on a container': lambda: {**launch(), 'kind': None},
    'ref on a container': lambda: {**launch(), 'kind': None, 'runtime': None},
    'hub weights extra key': edit(['weights', 0, 'sha256'], '1' * 64),
    'macos is a float': edit(['runtime', 'macos'], 15.0),  # JSON 15.0 reaches JS as 15; the registry never holds one
}


def lab_rejects(L):
    try:
        lab.check_host_runtime(L)
    except AssertionError:
        return True
    return False


NODE = shutil.which('node')
# bash (what the site's one command runs), sh, and dash (sh on Debian and Ubuntu) when installed
SHELLS = [('bash', '-c'), ('bash', '-ec'), ('sh', '-c')] + ([('dash', '-c')] if shutil.which('dash') else [])


def js(fn, L):
    """`steps` or `command` from the JS SDK for the same launch."""
    out = subprocess.run([NODE, '--input-type=module', '-e',
                          f'import {{readFileSync}} from "node:fs"; const m = await import(process.argv[1]); '
                          f'console.log(JSON.stringify(m.{fn}(JSON.parse(readFileSync(0, "utf8")))));', str(ROOT / 'sdk/js/index.js')],
                         input=json.dumps({'model': 'test-model', 'launch': L}), capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def renderings(L, fn='steps'):
    """Each SDK's output for one launch: Python always, JS when node is installed."""
    out = {'python': getattr(sdk, fn)({'model': 'test-model', 'launch': L})}
    if NODE:
        out['js'] = js(fn, L)
    return out


class ContractTest(unittest.TestCase):
    def test_fixture_passes(self):
        lab.check_host_runtime(launch())
        self.assertEqual(launch()['runtime'], runtime())

    def test_lab_rejects(self):
        for name, make in {**BAD, **BAD_LAB}.items():
            self.assertTrue(lab_rejects(make()), name)

    def test_sdk_refuses_to_render(self):
        for name, make in BAD.items():
            with self.assertRaises(ValueError, msg=name):
                sdk.steps({'model': 'test-model', 'launch': make()})

    def test_launches_without_the_fields_are_untouched(self):
        # only the launches named in RUNTIME_LAUNCHES use the fields; every rendered launch passes
        seen = set()
        for f in sorted(lab.RECIPES.rglob('*.json')):
            r = json.loads(f.read_text())
            L = lab.render(r)
            lab.check_host_runtime(L)
            if 'runtime' not in L:
                continue
            launch_id = r['engine'].split('@')[0]
            self.assertIn(launch_id, RUNTIME_LAUNCHES, f.name)
            seen.add(launch_id)
            # it renders as the confined runtime steps (after any "Set up first" note and its macOS check), and its one-script
            # command stops at the first failure
            x = {'model': r['model'], 'launch': L}
            titles = [s['title'] for s in sdk.steps(x) if s['title'] != 'Set up first']
            if 'macos' in L['runtime']:
                self.assertEqual(titles.pop(0), 'Check the macOS version', f.name)
            self.assertEqual(titles[:3], ['Install the runtime', 'Install the packages', 'Download the weights'], f.name)
            self.assertTrue(sdk.command(x).startswith('set -e\n\n'), f.name)
        self.assertEqual(seen, RUNTIME_LAUNCHES)

    def test_check_reports_a_bad_runtime(self):
        bad = profile({**runtime(), 'archive': {**runtime()['archive'], 'sha256': 'x' * 64}})
        r = {'model': 'test-model', 'weights': 'org/model@' + REV, 'engine': 'fixture-runtime@host', 'set': {}, 'card': 'apple-m1-8gb',
             'proof': [{'gates': 'load chat reasoning tools context speed', 'tps': 20.0}]}
        with tempfile.TemporaryDirectory() as d, patch.object(lab, 'ROOT', Path(d)), patch.object(lab, 'RECIPES', Path(d) / 'recipes'), \
                patch.object(lab, 'profile', return_value=bad):
            out = lab.recipe_path(r, lab.render(r)); out.parent.mkdir(parents=True); out.write_text(json.dumps(r))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(lab.cmd_check(None), 1)
            self.assertIn('runtime.archive.sha256', buf.getvalue())


class PinTest(unittest.TestCase):
    def test_every_existing_pin_unchanged(self):
        # a container pins to its image digest or build commit, which this never touches; every host profile keeps its hash
        ps = [json.loads(f.read_text()) for f in sorted(lab.LAUNCHES.glob('*.json'))]
        hosts = [p for p in ps if p.get('kind') == 'host']
        self.assertTrue(hosts)
        self.assertEqual({p['id'] for p in ps if 'runtime' in p}, RUNTIME_LAUNCHES)
        for p in hosts:
            if p['id'] in RUNTIME_LAUNCHES:  # its runtime is part of what runs, so it is in the pin, and it is all that moved it
                self.assertNotEqual(lab.pin(p), before(p), p['id'])
                rest = {k: v for k, v in p.items() if k != 'runtime'}
                self.assertEqual(lab.pin(rest), before(rest), p['id'])
            else:
                self.assertEqual(lab.pin(p), before(p), p.get('id'))

    def test_runtime_counts(self):
        p = profile()
        without = {k: v for k, v in p.items() if k != 'runtime'}
        self.assertEqual(lab.pin(without), before(without))
        self.assertNotEqual(lab.pin(p), lab.pin(without))
        self.assertNotEqual(lab.pin(p), lab.pin(profile(runtime(('1' * 64, '2' * 64, '5' * 64, '4' * 64)))))  # one wheel hash
        moved = copy.deepcopy(p); moved['weights'][0]['ref'] = 'v1'
        self.assertNotEqual(lab.pin(p), lab.pin(moved))  # the task-local ref


class StepsTest(unittest.TestCase):
    def setUp(self):
        self.steps = sdk.steps({'model': 'test-model', 'launch': launch()})
        self.by = {s['title']: s['code'] for s in self.steps}

    def test_titles(self):
        self.assertEqual([s['title'] for s in self.steps],
                         ['Install the runtime', 'Install the packages', 'Download the weights', 'Write the launcher', 'Start the server'])

    def test_every_setup_step_is_one_strict_subshell(self):
        for s in self.steps[:-1]:
            self.assertTrue(s['code'].startswith('(\nset -e\n') and s['code'].endswith('\n)'), s['title'])

    def test_command_stops_at_the_first_failure(self):
        for name, c in renderings(launch(), 'command').items():
            self.assertTrue(c.startswith('set -e\n\n'), name)

    def test_no_prose_no_flat_weights_no_ambient_tools(self):
        code = '\n'.join(s['code'] for s in self.steps)
        self.assertNotIn('Task runtime', code)  # the free-text install is not rendered when runtime is pinned
        self.assertNotIn('--local-dir', code)
        self.assertNotIn('hf download', code)
        self.assertIn("./rt/python/bin/python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from huggingface_hub import snapshot_download",
                      self.by['Download the weights'])
        self.assertIn('./rt/python/bin/python3 -I -m pip --isolated install', self.by['Install the packages'])

    def test_archive_is_checked_before_extracting(self):
        code = self.by['Install the runtime']
        self.assertIn(f'curl -q -fL -o "$s"/{ARCHIVE} {ARCHIVE_URL}', code)
        self.assertLess(code.index('shasum -a 256 -c'), code.index('tar -xzf'))

    def test_packages_are_hash_locked_offline_and_staged(self):
        code = self.by['Install the packages']
        for flag in ('--no-index', '--only-binary :all:', '--require-hashes', '--target "$s/t"'):
            self.assertIn(flag, code)
        self.assertLess(code.index('shasum -a 256 -c SHA256SUMS'), code.index('pip --isolated install'))
        self.assertLess(code.index('pip --isolated install'), code.index('mv "$s/t" rt/site'))

    def test_macos_check_comes_first_and_only_when_asked(self):
        L = launch({**runtime(), 'macos': 15})
        for name, st in renderings(L).items():
            self.assertEqual([s['title'] for s in st][:2], ['Check the macOS version', 'Install the runtime'], name)
            code = st[0]['code']
            self.assertTrue(code.startswith('(\nset -e\n') and code.endswith('\n)'), name)
            self.assertIn('sw_vers -productVersion', code)
            self.assertIn('-lt 15 ]', code)
            self.assertNotIn('curl', code)
        self.assertNotIn('Check the macOS version', [s['title'] for s in self.steps])

    def test_existing_shapes_unchanged(self):
        # without the new fields: the old install, pip, --local-dir and launcher rendering, and no `set -e` in command()
        L = launch()
        del L['runtime']
        L['pip'] = ['mlx-lm==0.31.3']
        L['weights'] = [{'repo': 'org/model', 'revision': REV, 'at': 'models/model', 'layout': 'dir', 'files': None}]
        st = {s['title']: s['code'] for s in sdk.steps({'model': 'test-model', 'launch': L})}
        self.assertEqual(st['Install the engine'], 'python3 -m pip install mlx-lm==0.31.3')
        self.assertEqual(st['Download the weights'], f'hf download org/model \\\n  --revision {REV} \\\n  --local-dir models/model')
        self.assertEqual(st['Write the launcher'], f"mkdir -p launcher\ncat > launcher/serve.py <<'EOF'\n{SERVER.rstrip()}\nEOF")
        self.assertFalse(sdk.command({'model': 'test-model', 'launch': L}).startswith('set -e'))


def make_wheel(d, name, version, requires=(), module='VERSION = 1\n'):
    """A minimal pure-Python wheel that pip will install."""
    dist = name.replace('-', '_')
    f = Path(d) / f'{dist}-{version}-py3-none-any.whl'
    info = f'{dist}-{version}.dist-info'
    files = {f'{dist}/__init__.py': module,
             f'{info}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n' + ''.join(f'Requires-Dist: {r}\n' for r in requires),
             f'{info}/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    digest = lambda v: base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b'=').decode()
    record = ''.join(f'{k},sha256={digest(v)},{len(v.encode())}\n' for k, v in files.items()) + f'{info}/RECORD,,\n'
    with zipfile.ZipFile(f, 'w') as z:
        for k, v in {**files, f'{info}/RECORD': record}.items():
            z.writestr(k, v)
    return f


# The pinned huggingface_hub stand-in: makes the snapshot (unless FAKE_HF_MISS) and records which copy ran.
HUB = '''import json, os
ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "HF_XET_CACHE", "HF_ENDPOINT", "HF_TOKEN_PATH", "HOME", "NETRC", "TMPDIR")
def snapshot_download(repo_id, revision=None, cache_dir=None, allow_patterns=None, token=None):
    open("HUB_USED", "w").write(os.path.abspath(__file__))
    json.dump({"token": token, "cache_dir": os.path.abspath(cache_dir), **{k: os.environ.get(k) for k in ENV}}, open("HUB_ENV", "w"))
    if os.environ.get("FAKE_HF_MISS") or os.environ.get("FAKE_HF_MISS_REPO") == repo_id:
        return cache_dir
    # the real layout: a blob, a snapshot file linking to it, and a lock under the cache root
    m = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snap = os.path.join(m, "snapshots", revision)
    blob = os.path.join(m, "blobs", "0f1e2d")
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    open(blob, "w").write("{}")
    os.makedirs(snap, exist_ok=True)
    os.symlink(os.path.relpath(blob, snap), os.path.join(snap, "config.json"))
    os.makedirs(os.path.join(cache_dir, ".locks", os.path.basename(m)), exist_ok=True)
    # a hostile cache, changed while the download ran
    hostile, out = os.environ.get("FAKE_HF_HOSTILE"), os.environ.get("FAKE_OUTSIDE")
    if hostile == "refs":  # refs a link out
        os.symlink(out, os.path.join(m, "refs"))
    elif hostile == "ref":  # refs/main a link out
        os.makedirs(os.path.join(m, "refs"), exist_ok=True)
        os.symlink(os.path.join(out, "victim"), os.path.join(m, "refs", "main"))
    elif hostile == "refdir":  # refs/main a directory
        os.makedirs(os.path.join(m, "refs", "main"), exist_ok=True)
    elif hostile == "escape":  # a snapshot file that is an absolute link out
        os.symlink(os.path.join(out, "victim"), os.path.join(snap, "model.safetensors"))
    elif hostile == "climb":  # a snapshot file that climbs out
        os.symlink(os.path.relpath(os.path.join(out, "victim"), snap), os.path.join(snap, "model.safetensors"))
    elif hostile == "fifo":
        os.mkfifo(os.path.join(snap, "pipe"))
    return snap
'''
# An ambient hf for a launch without a runtime: records what it sees and lays out the cache like the real one.
FAKE_HF = '''#!/bin/sh
repo=$2; rev=; dir=
while [ $# -gt 0 ]; do case "$1" in --revision) rev=$2; shift 2;; --cache-dir) dir=$2; shift 2;; *) shift;; esac; done
env > HF_ENV
m="$dir/models--$(printf '%s' "$repo" | sed 's#/#--#')"
mkdir -p "$m/blobs" "$m/snapshots/$rev" && echo '{}' > "$m/blobs/0f1e2d" && ln -s ../../blobs/0f1e2d "$m/snapshots/$rev/config.json"
'''
FAKE_CURL = '''#!/bin/sh
# curl -fL -o FILE URL: copy the URL's file from $FAKE_SERVER, or fail like curl -f on a 404. Like real curl, it reads
# $CURL_HOME/.curlrc or $HOME/.curlrc (here only a `trace = FILE` line) unless -q is its first argument.
if [ "$1" != "-q" ]; then
  for rc in "${CURL_HOME:-$HOME}/.curlrc" "$HOME/.curlrc"; do
    if [ -f "$rc" ]; then t=$(sed -n 's/^trace *= *//p' "$rc"); if [ -n "$t" ]; then echo "curl read $rc" >> "$t"; fi; fi
  done
fi
out=; url=
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2;; -*) shift;; *) url=$1; shift;; esac; done
name=$(printf '%s' "${url##*/}" | sed 's/%2B/+/g')
[ -f "$FAKE_SERVER/$name" ] || exit 22
cp "$FAKE_SERVER/$name" "$out"
'''
# mv, except that a publish onto $FAIL_MV_TO (a move out of a .local-ai- staging dir) fails, to exercise the rollback, and
# with $FAIL_RESTORE the restore of the previous copy (a move out of .../old) fails too
FAKE_MV = '''#!/bin/sh
for last; do :; done
if [ -n "$FAIL_MV_TO" ] && [ "$last" = "$FAIL_MV_TO" ]; then
  case "$1" in
    */old|*/old/*) if [ -n "$FAIL_RESTORE" ]; then echo "mv: injected restore failure" >&2; exit 1; fi;;
    *.local-ai-*) echo "mv: injected failure" >&2; exit 1;;
  esac
fi
# with $RETERM (parent or group) every restore move first sends the step two more TERMs, to its shell or to its whole
# process group (so the test must run the step in its own session)
if [ -n "$RETERM" ]; then case "$1" in */old|*/old/*)
  echo "$1" >> "$PWD/RETERMED"
  if [ "$RETERM" = group ]; then kill -TERM 0; kill -TERM 0; else kill -TERM "$PPID"; kill -TERM "$PPID"; fi;;
esac; fi
# with $FAIL_ANY_RESTORE every move out of .../old (putting a backup back) fails
if [ -n "$FAIL_ANY_RESTORE" ]; then case "$1" in */old|*/old/*) echo "mv: injected restore failure" >&2; exit 1;; esac; fi
/bin/mv "$@" || exit
# after a move from $KILL_FROM to $KILL_TO (glob patterns), signal the step's shell with $KILL_SIG (TERM by default)
if [ -n "$KILL_FROM" ]; then case "$1" in $KILL_FROM) case "$last" in $KILL_TO) echo "$1 -> $last" >> "$PWD/KILLED"; kill -"${KILL_SIG:-TERM}" "$PPID";; esac;; esac; fi
'''
# Anything looked up on PATH or planted in the working directory instead of the pinned runtime: record it and fail.
TRAP = '#!/bin/sh\necho "$0 $*" >> "$TRAPPED"\nexit 73\n'


def tree(p):
    """Every file and link under p with its content hash or target, so two trees compare exactly."""
    p = Path(p)
    if not p.exists() and not p.is_symlink():
        return None
    return {str(f.relative_to(p)): ('link:' + os.readlink(f)) if f.is_symlink() else
            hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else 'dir' for f in sorted(p.rglob('*'))}


def staging_left(p):
    return [str(f) for f in Path(p).rglob('*') if '.local-ai-' in f.name]


@unittest.skipUnless(shutil.which('shasum') and shutil.which('tar') and shutil.which('bash') and shutil.which('mktemp')
                     and subprocess.run([sys.executable, '-I', '-m', 'pip', '--version'], capture_output=True).returncode == 0,
                     'needs shasum, tar, bash, mktemp and pip')
class SandboxTest(unittest.TestCase):
    """Run the rendered steps for real in a temp directory: real shasum, tar, mktemp and pip; curl reads local files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = Path(self.tmp.name)
        self.server, self.bin, self.outside = d / 'server', d / 'bin', d / 'outside'
        for p in (self.server, self.bin, self.outside):
            p.mkdir()
        (self.outside / 'victim').write_text('untouched\n')
        (self.bin / 'curl').write_text(FAKE_CURL)
        (self.bin / 'mv').write_text(FAKE_MV)
        for name in ('hf', 'python3', 'python', 'pip', 'pip3', 'huggingface-cli'):
            (self.bin / name).write_text(TRAP)
        for f in self.bin.iterdir():
            f.chmod(0o755)
        # the "interpreter archive": python/bin/python3 hands off to this test's own interpreter by absolute path
        src = d / 'src' / 'python' / 'bin'; src.mkdir(parents=True)
        # (and records, for a pip run, the environment and arguments pip is started with)
        (src / 'python3').write_text('#!/bin/sh\ncase " $* " in *" -m pip "*) printf \'TMPDIR=%s\\nTMP=%s\\nTEMP=%s\\nHOME=%s\\nXDG_CACHE_HOME=%s\\n'
                                     'XDG_CONFIG_HOME=%s\\nPIP_CONFIG_FILE=%s\\nARGS=%s\\n\' "$TMPDIR" "$TMP" "$TEMP" "$HOME" "$XDG_CACHE_HOME" '
                                     '"$XDG_CONFIG_HOME" "$PIP_CONFIG_FILE" "$*" > PIP_ENV;; esac\n'
                                     f'exec "{sys.executable}" "$@"\n'); (src / 'python3').chmod(0o755)
        subprocess.run(['tar', '-czf', str(self.server / ARCHIVE), '-C', str(d / 'src'), 'python'], check=True)
        a = make_wheel(self.server, 'fixture-a', '1.0', requires=['fixture-b'])
        b = make_wheel(self.server, 'fixture-b', '2.0')
        h = make_wheel(self.server, 'huggingface-hub', '1.0', module=HUB)
        self.shas = tuple(hashlib.sha256(f.read_bytes()).hexdigest() for f in (self.server / ARCHIVE, a, b, h))
        self.archive_bytes = (self.server / ARCHIVE).read_bytes()
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def work(self):
        """A fresh working directory, with a planted python3 that must never run."""
        self.n += 1
        w = self.d / f'work{self.n}'
        w.mkdir()
        (w / 'python3').write_text(TRAP); (w / 'python3').chmod(0o755)
        return w

    def sh(self, code, work, shell=('sh', '-c'), session=False, **env):
        """Run the step; with session=True in its own session, whose every process is killed if it outlives the timeout."""
        p = subprocess.Popen([*shell, code], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=session,
                             env={'PATH': f'{self.bin}:/usr/bin:/bin', 'FAKE_SERVER': str(self.server), 'HOME': str(work),
                                  'TRAPPED': str(work / 'TRAPPED'), 'FAKE_OUTSIDE': str(self.outside), **env})
        try:
            out, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9) if session else p.kill()
            p.communicate()
            raise
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)

    def step(self, L, title, work, sdk_name='python', **env):
        code = next(s['code'] for s in renderings(L)[sdk_name] if s['title'] == title)
        return self.sh(code, work, **env)

    def L(self, shas=None, root='rt'):
        return launch(runtime(shas or self.shas, root=root), root=root)

    def assertNothingRan(self, work):
        self.assertFalse((work / 'TRAPPED').exists(), (work / 'TRAPPED').read_text() if (work / 'TRAPPED').exists() else '')

    # 1. the whole command: success, and each failure stops every later stage with a non-zero exit, in either SDK

    def test_whole_script_runs_on_the_pinned_runtime(self):
        for name, script in renderings(self.L(), 'command').items():
            for shell in SHELLS:
                w = self.work()
                r = self.sh(script, w, shell)
                self.assertEqual(r.returncode, 0, (name, shell, r.stderr[-2000:]))
                self.assertEqual((w / 'SERVER_STARTED').read_text(), 'ok')
                self.assertEqual((w / 'rt/hf-cache/models--org--model/refs/main').read_text(), REV)
                # 5. the weights came from the runtime's own huggingface_hub, not an hf on PATH
                self.assertEqual(Path((w / 'HUB_USED').read_text()).resolve(), (w / 'rt/site/huggingface_hub/__init__.py').resolve())
                # 4. nothing on PATH or planted in the working directory ran
                self.assertNothingRan(w)
                self.assertEqual(staging_left(w), [])

    def test_whole_script_stops_at_each_failure(self):
        bad = lambda i: tuple('0' * 64 if j == i else s for j, s in enumerate(self.shas))
        cases = {  # name: (launch, env, set up, what must not exist afterwards)
            'archive hash': (self.L(bad(0)), {}, None, ['rt/python', 'rt/site', 'rt/hf-cache', 'launcher', 'SERVER_STARTED']),
            'wheel hash': (self.L(bad(2)), {}, None, ['rt/site', 'rt/wheels', 'rt/hf-cache', 'launcher', 'SERVER_STARTED']),
            'missing snapshot': (self.L(), {'FAKE_HF_MISS': '1'}, None, ['rt/hf-cache/models--org--model/refs', 'launcher', 'SERVER_STARTED']),
            'launcher': (self.L(), {}, lambda w: (w / 'launcher').write_text('a file in the way\n'), ['SERVER_STARTED']),
        }
        for case, (L, env, setup, absent) in cases.items():
            for name, script in renderings(L, 'command').items():
                for shell in SHELLS:
                    w = self.work()
                    if setup:
                        setup(w)
                    r = self.sh(script, w, shell, **env)
                    self.assertNotEqual(r.returncode, 0, (case, name, shell))
                    for a in absent:
                        self.assertFalse((w / a).exists(), (case, name, shell, a))
                    self.assertFalse((w / 'SERVER_STARTED').exists(), (case, name, shell))
                    self.assertEqual(staging_left(w), [], (case, name, shell))

    def test_macos_check_stops_before_any_download(self):
        L = launch({**runtime(self.shas), 'macos': 15})
        for version, passes in (('14.6', False), ('11.7.10', False), ('not a version', False), (None, False), ('15.0', True), ('26.4.1', True)):
            (self.bin / 'sw_vers').write_text('#!/bin/sh\nexit 1\n' if version is None else f"#!/bin/sh\necho '{version}'\n")
            (self.bin / 'sw_vers').chmod(0o755)
            for name, script in renderings(L, 'command').items():
                for shell in SHELLS:
                    w = self.work()
                    r = self.sh(script, w, shell)
                    if passes:
                        self.assertEqual(r.returncode, 0, (version, name, shell, r.stderr[-2000:]))
                        self.assertEqual((w / 'SERVER_STARTED').read_text(), 'ok')
                    else:
                        self.assertNotEqual(r.returncode, 0, (version, name, shell))
                        self.assertIn('needs macOS 15 or later', r.stderr, (version, name, shell))
                        self.assertEqual(sorted(p.name for p in w.iterdir()), ['python3'], (version, name, shell))  # nothing made
                    self.assertNothingRan(w)
        (self.bin / 'sw_vers').unlink()

    # 2. packages: a fresh staged target, published only after the whole install succeeded

    def install(self, w, L=None):
        L = L or self.L()
        for title in ('Install the runtime', 'Install the packages'):
            r = self.step(L, title, w)
            self.assertEqual(r.returncode, 0, (title, r.stderr[-2000:]))

    def test_reinstall_replaces_a_dirty_target_exactly(self):
        ref = self.work(); self.install(ref)
        w = self.work(); self.install(w)
        (w / 'rt/site/fixture_a/__init__.py').write_text('TAMPERED = True\n')
        (w / 'rt/site/stale_pkg').mkdir(); (w / 'rt/site/stale_pkg/__init__.py').write_text('')
        (w / 'rt/wheels/stale-9-py3-none-any.whl').write_text('')
        r = self.step(self.L(), 'Install the packages', w)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(tree(w / 'rt/site'), tree(ref / 'rt/site'))
        self.assertEqual(tree(w / 'rt/wheels'), tree(ref / 'rt/wheels'))
        self.assertEqual(staging_left(w), [])

    def test_failed_reinstall_keeps_the_prior_target(self):
        w = self.work(); self.install(w)
        site, wheels = tree(w / 'rt/site'), tree(w / 'rt/wheels')
        rt = runtime(self.shas); rt['packages']['wheels'] = [rt['packages']['wheels'][0], rt['packages']['wheels'][2]]  # fixture-b missing
        for case, L in {'wrong wheel hash': self.L(self.shas[:2] + ('e' * 64,) + self.shas[3:]), 'incomplete lock': launch(rt)}.items():
            r = self.step(L, 'Install the packages', w)
            self.assertNotEqual(r.returncode, 0, case)
            self.assertEqual(tree(w / 'rt/site'), site, case)
            self.assertEqual(tree(w / 'rt/wheels'), wheels, case)
            self.assertEqual(staging_left(w), [], case)

    def test_reinstalled_runtime_replaces_only_the_archive_entries(self):
        w = self.work(); self.install(w)
        self.assertEqual(self.step(self.L(), 'Download the weights', w).returncode, 0)
        (w / 'rt/python/STALE').write_text('')
        site, cache = tree(w / 'rt/site'), tree(w / 'rt/hf-cache')
        r = self.step(self.L(), 'Install the runtime', w)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((w / 'rt/python/STALE').exists())
        self.assertEqual((tree(w / 'rt/site'), tree(w / 'rt/hf-cache')), (site, cache))

    def test_wrong_archive_hash_leaves_the_prior_runtime(self):
        w = self.work(); self.install(w)
        before_ = tree(w / 'rt')
        r = self.step(self.L(('f' * 64,) + self.shas[1:]), 'Install the runtime', w)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(tree(w / 'rt'), before_)

    def test_missing_wheel_stops(self):
        w = self.work(); self.step(self.L(), 'Install the runtime', w)
        (self.server / 'fixture_b-2.0-py3-none-any.whl').unlink()
        r = self.step(self.L(), 'Install the packages', w)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((w / 'rt/site').exists() or (w / 'rt/wheels').exists())

    # 3. no write through a symlink, at the destination or any directory above it, in either SDK's rendering

    def test_symlinks_are_refused(self):
        def link(rel, to=None):
            def make(w):
                (w / rel).parent.mkdir(parents=True, exist_ok=True)
                if (w / rel).is_dir() and not (w / rel).is_symlink():
                    shutil.rmtree(w / rel)
                elif (w / rel).exists():
                    (w / rel).unlink()
                (w / rel).symlink_to(to or self.outside)
            return make
        m = 'rt/hf-cache/models--org--model'
        cases = {  # name: (launch, prerequisite steps, symlink, step)
            'into leaf': (self.L(), [], link('rt'), 'Install the runtime'),
            'archive entry': (self.L(), [], link('rt/python'), 'Install the runtime'),
            'into parent': (self.L(root='opt/rt'), [], link('opt'), 'Install the runtime'),
            'wheel dir leaf': (self.L(), ['Install the runtime'], link('rt/wheels'), 'Install the packages'),
            'target leaf': (self.L(), ['Install the runtime'], link('rt/site'), 'Install the packages'),
            'packages parent': (self.L(root='opt/rt'), [], link('opt'), 'Install the packages'),
            'cache leaf': (self.L(), ['Install the runtime', 'Install the packages'], link('rt/hf-cache'), 'Download the weights'),
            'model dir': (self.L(), ['Install the runtime', 'Install the packages'], link(m), 'Download the weights'),
            'cache parent': (self.L(root='opt/rt'), [], link('opt'), 'Download the weights'),
            'launcher leaf': (self.L(), [], link('launcher/serve.py', self.outside / 'victim'), 'Write the launcher'),
            'launcher parent': (self.L(), [], link('launcher'), 'Write the launcher'),
        }
        for case, (L, pre, make, title) in cases.items():
            for name in renderings(L):
                w = self.work()
                for p in pre:
                    self.assertEqual(self.step(L, p, w, name).returncode, 0, (case, p))
                make(w)
                was = tree(self.outside)
                r = self.step(L, title, w, name)
                self.assertNotEqual(r.returncode, 0, (case, name))
                self.assertIn('is a symlink', r.stderr, (case, name))
                self.assertEqual(tree(self.outside), was, (case, name))

    # 5. the weights step needs nothing on PATH

    def test_weights_without_any_hf_on_path(self):
        w = self.work(); self.install(w)
        r = self.step(self.L(), 'Download the weights', w, PATH='/usr/bin:/bin')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((w / 'rt/hf-cache/models--org--model/refs/main').read_text(), REV)

    # r2/r3: the download goes into a fresh staging cache; links planted in the old cache are never followed, the staged
    # cache must hold only links that stay inside it, and the ref is written through an exclusive temp file

    def test_planted_links_in_the_cache_are_never_followed(self):
        m = 'rt/hf-cache/models--org--model'
        plants = {f'{m}/blobs': self.outside, f'{m}/refs': self.outside, f'{m}/snapshots/{REV}': self.outside,
                  f'{m}/.no_exist': self.outside, 'rt/hf-cache/.locks': self.outside}
        for rel, to in plants.items():
            for name in renderings(self.L()):
                w = self.work(); self.install(w)
                (w / rel).parent.mkdir(parents=True, exist_ok=True)
                (w / rel).symlink_to(to)
                was = tree(self.outside)
                r = self.step(self.L(), 'Download the weights', w, name)
                self.assertEqual(r.returncode, 0, (rel, name, r.stderr))
                self.assertEqual(tree(self.outside), was, (rel, name))
                self.assertFalse((w / m / 'blobs').is_symlink() or (w / m / 'refs').is_symlink(), (rel, name))
                self.assertEqual((w / m / 'refs/main').read_text(), REV, (rel, name))
                self.assertEqual((w / m / f'snapshots/{REV}/config.json').read_text(), '{}', (rel, name))  # the internal link kept
                self.assertEqual(staging_left(w), [], (rel, name))

    def test_failed_hub_publish_restores_the_cache(self):
        m = 'rt/hf-cache/models--org--model'
        for name in renderings(self.L()):
            w = self.work(); self.install(w)
            self.assertEqual(self.step(self.L(), 'Download the weights', w, name).returncode, 0, name)
            (w / m / 'KEEP').write_text('the previous cache\n')
            was = tree(w / m)
            r = self.step(self.L(), 'Download the weights', w, name, FAIL_MV_TO=m)
            self.assertNotEqual(r.returncode, 0, name)
            self.assertEqual(tree(w / m), was, name)
            self.assertEqual(staging_left(w), [], name)

    def test_planted_fixed_temp_link_is_never_written(self):
        for name in renderings(self.L()):
            w = self.work(); (w / 'launcher').mkdir()
            (w / 'launcher/.local-ai-main').symlink_to(self.outside / 'victim')
            r = self.step(self.L(), 'Write the launcher', w, name)
            self.assertEqual(r.returncode, 0, (name, r.stderr))
            self.assertEqual((self.outside / 'victim').read_text(), 'untouched\n', name)
            self.assertEqual((w / 'launcher/serve.py').read_text().rstrip(), SERVER.rstrip(), name)

    def test_hostile_cache_during_download(self):
        for hostile in ('refs', 'ref', 'refdir', 'escape', 'climb', 'fifo'):
            for name in renderings(self.L()):
                w = self.work(); self.install(w)
                was = tree(self.outside)
                r = self.step(self.L(), 'Download the weights', w, name, FAKE_HF_HOSTILE=hostile)
                self.assertNotEqual(r.returncode, 0, (hostile, name))
                self.assertEqual(tree(self.outside), was, (hostile, name))
                self.assertFalse((w / 'rt/hf-cache/models--org--model').exists(), (hostile, name))  # nothing published
                self.assertEqual(staging_left(w), [], (hostile, name))

    # r3: no ambient token, Hugging Face config, cache or .netrc reaches the download (synthetic values only)

    SYNTHETIC = {'HF_TOKEN': 'hf_synthetic_r3_marker', 'HUGGING_FACE_HUB_TOKEN': 'hf_synthetic_r3_marker',
                 'HF_ENDPOINT': 'https://synthetic.invalid', 'HF_TOKEN_PATH': '/nonexistent/synthetic-token'}

    def isolated(self, w, seen):
        for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_ENDPOINT', 'HF_TOKEN_PATH', 'NETRC'):
            self.assertIsNone(seen.get(k), k)
        hub = Path(w).resolve() / 'rt/hf-cache'
        for k in ('HF_HOME', 'HF_XET_CACHE', 'HOME', 'TMPDIR'):
            self.assertIsNotNone(seen.get(k), f'{k} is not set inside the staging cache')
            self.assertTrue(Path(seen[k]).resolve().is_relative_to(hub) and '.local-ai-hub.' in seen[k], (k, seen[k]))

    # r4: nested or duplicate hub caches would erase a model on a rerun; both SDKs refuse to render them. This also runs the
    # reviewer's reproductions on any rendering that does not refuse (an older SDK), so it fails there on the lost data.

    def rendered_or_none(self, L, name):
        try:
            return renderings(L)[name]
        except (ValueError, subprocess.CalledProcessError):
            return None

    def test_nested_or_duplicate_caches_never_lose_a_model(self):
        cases = {'nested': (CACHES['second hub cache inside the first (in its model dir)'](),
                            'rt/hf-cache/models--org--model/blobs/models--org--draft/snapshots/' + 'c' * 40),
                 'two revisions': (CACHES['same repo twice in one cache, two revisions'](), None)}
        for case, (L, second) in cases.items():
            for name in ('python', 'js') if NODE else ('python',):
                with self.subTest(case=case, sdk=name):
                    st = self.rendered_or_none(L, name)
                    if st is None:
                        continue  # refused: nothing can be lost
                    code = next(s['code'] for s in st if s['title'] == 'Download the weights')
                    w = self.work(); self.install(w)
                    r = self.sh(code, w)
                    self.assertEqual(r.returncode, 0, (case, name, r.stderr))
                    m = w / 'rt/hf-cache/models--org--model'
                    if second:  # rerun: the first publish succeeds, the second download fails; the old second must survive
                        self.assertTrue((w / second).exists(), (case, name))
                        self.sh(code, w, FAKE_HF_MISS_REPO='org/draft')
                        self.assertTrue((w / second).exists(), f'{case} ({name}): the previous org/draft model was deleted')
                    else:  # both requested revisions and both refs must be there
                        self.assertTrue((m / 'snapshots' / REV).exists() and (m / 'refs/v1').exists(), f'{case} ({name}): revision v1 was overwritten')

    def test_refused_by_both_sdks_and_lab(self):
        for case, make in CACHES.items():
            L = make()
            self.assertTrue(lab_rejects(L), case)
            for name in ('python', 'js') if NODE else ('python',):
                self.assertIsNone(self.rendered_or_none(L, name), (case, name))

    # r4: curl reads no ambient config (-q first), and pip reads no config file and keeps its temp, home and cache inside
    # the staging directory (synthetic files in the scratch directory only)

    def test_every_curl_starts_with_q(self):
        for name, st in renderings(self.L()).items():
            lines = [x.strip() for s in st for x in s['code'].splitlines() if 'curl ' in x]
            self.assertTrue(lines, name)
            self.assertTrue(all(x.startswith('curl -q ') for x in lines), (name, [x for x in lines if not x.startswith('curl -q ')]))

    def test_curl_reads_no_ambient_config(self):
        for name in renderings(self.L()):
            w = self.work()
            for d in ('curl-home', 'home'):
                (self.outside / d).mkdir(exist_ok=True)
                (self.outside / d / '.curlrc').write_text(f'trace = {self.outside}/curl.trace\n')
            env = {'CURL_HOME': str(self.outside / 'curl-home'), 'HOME': str(self.outside / 'home')}
            for title in ('Install the runtime', 'Install the packages'):
                r = self.step(self.L(), title, w, name, **env)
                self.assertEqual(r.returncode, 0, (name, title, r.stderr))
            self.assertFalse((self.outside / 'curl.trace').exists(), (name, (self.outside / 'curl.trace').read_text() if (self.outside / 'curl.trace').exists() else ''))

    def test_pip_reads_no_config_and_writes_nothing_outside(self):
        for name in renderings(self.L()):
            w = self.work()
            self.assertEqual(self.step(self.L(), 'Install the runtime', w, name).returncode, 0, name)
            o = self.outside / name; o.mkdir()
            for d in ('tmp', 'home', 'xdg-cache', 'xdg-config'):
                (o / d).mkdir()
            (o / 'pip.conf').write_text(f'[global]\nlog = {o}/pip.log\n')
            env = {'PIP_CONFIG_FILE': str(o / 'pip.conf'), 'PIP_LOG': str(o / 'pip-env.log'), 'TMPDIR': str(o / 'tmp'),
                   'TMP': str(o / 'tmp'), 'TEMP': str(o / 'tmp'), 'HOME': str(o / 'home'), 'XDG_CACHE_HOME': str(o / 'xdg-cache'),
                   'XDG_CONFIG_HOME': str(o / 'xdg-config')}
            r = self.step(self.L(), 'Install the packages', w, name, **env)
            self.assertEqual(r.returncode, 0, (name, r.stderr[-2000:]))
            self.assertTrue((w / 'rt/site/fixture_a/__init__.py').exists(), name)
            # pip deletes its temp trees after it runs, so check where it was told to put them
            seen = dict(line.split('=', 1) for line in (w / 'PIP_ENV').read_text().splitlines())
            self.assertEqual(seen['PIP_CONFIG_FILE'], '/dev/null', name)
            self.assertIn('--no-cache-dir', seen['ARGS'].split(), name)
            stage = Path(w).resolve() / 'rt'
            for k in ('TMPDIR', 'TMP', 'TEMP', 'HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME'):
                self.assertTrue(seen[k] and Path(seen[k]).resolve().is_relative_to(stage) and '.local-ai-packages.' in seen[k], (name, k, seen[k]))
            self.assertEqual(sorted(str(p.relative_to(o)) for p in o.rglob('*')), ['home', 'pip.conf', 'tmp', 'xdg-cache', 'xdg-config'], name)
            self.assertEqual(staging_left(w), [], name)

    def test_pinned_download_uses_no_ambient_credentials(self):
        for name in renderings(self.L()):
            w = self.work(); self.install(w)
            env = {**self.SYNTHETIC, 'HF_HOME': str(self.outside / 'hf-home'), 'HF_XET_CACHE': str(self.outside / 'xet'),
                   'NETRC': str(self.outside / 'netrc')}
            r = self.step(self.L(), 'Download the weights', w, name, **env)
            self.assertEqual(r.returncode, 0, (name, r.stderr))
            seen = json.loads((w / 'HUB_ENV').read_text())
            self.assertIs(seen['token'], False, name)
            self.isolated(w, seen)
            self.assertEqual(tree(self.outside), {'victim': hashlib.sha256(b'untouched\n').hexdigest()}, name)

    def test_ambient_hf_download_uses_no_ambient_credentials(self):
        L = {k: v for k, v in self.L().items() if k != 'runtime'}  # host hub weights without a runtime: hf from PATH
        hf = self.d / 'hf-bin'; hf.mkdir(exist_ok=True)
        (hf / 'hf').write_text(FAKE_HF); (hf / 'hf').chmod(0o755)
        for name in renderings(L):
            w = self.work()
            r = self.step(L, 'Download the weights', w, name, PATH=f'{hf}:{self.bin}:/usr/bin:/bin', **self.SYNTHETIC)
            self.assertEqual(r.returncode, 0, (name, r.stderr))
            seen = dict(line.split('=', 1) for line in (w / 'HF_ENV').read_text().splitlines() if '=' in line)
            self.isolated(w, seen)
            self.assertEqual((w / 'rt/hf-cache/models--org--model/refs/main').read_text(), REV, name)

    # r3: archive members and links are checked before anything runs or is published; safe internal links are kept

    def crafted(self, members, make=None):
        """Serve a .tar.gz holding `members` [(name, kind, value, mode)] and return the launch pinning it (`make(shas)`)."""
        f = self.server / ARCHIVE
        with tarfile.open(f, 'w:gz') as t:
            for name, kind, value, mode in members:
                i = tarfile.TarInfo(name); i.mode = mode
                if kind == 'file':
                    data = value.encode(); i.size = len(data); t.addfile(i, io.BytesIO(data))
                else:
                    i.type = {'sym': tarfile.SYMTYPE, 'hard': tarfile.LNKTYPE, 'fifo': tarfile.FIFOTYPE, 'dir': tarfile.DIRTYPE}[kind]
                    i.linkname = value or ''
                    t.addfile(i)
        shas = (hashlib.sha256(f.read_bytes()).hexdigest(),) + self.shas[1:]
        return make(shas) if make else self.L(shas)

    # r5: an archive entry may not replace another step's output even when the names differ only in case (macOS file
    # systems ignore case), whether the entry or a parent directory of the destination differs

    def test_archive_entry_never_replaces_output_differing_in_case(self):
        def variant(target='rt/site', at='rt/hf-cache', config='rt/launcher/serve.py'):
            def make(shas):
                L = launch(runtime(shas))
                L['runtime']['packages']['target'] = target
                L['env'] = {**L['env'], 'PYTHONPATH': target}
                for w in L['weights']:
                    w['at'] = at
                L['config']['at'] = config
                return L
            return make
        py = ('python/bin/python3', 'file', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755)
        cases = {  # name: (launch variant, protected dir, archive entry)
            'entry SITE, target rt/site': (variant(), 'rt/site', 'SITE'),
            'entry site, target rt/Site': (variant(target='rt/Site'), 'rt/Site', 'site'),
            'entry HF-CACHE, cache rt/hf-cache': (variant(), 'rt/hf-cache', 'HF-CACHE'),
            'entry LAUNCHER, launcher in rt/launcher': (variant(), 'rt/launcher', 'LAUNCHER'),
            'entry site, target under RT/': (variant(target='RT/site'), 'RT/site', 'site'),
            'entry hf-cache, cache under RT/': (variant(at='RT/hf-cache'), 'RT/hf-cache', 'hf-cache'),
            'entry launcher, launcher under RT/': (variant(config='RT/launcher/serve.py'), 'RT/launcher', 'launcher'),
        }
        for case, (make, protected, entry) in cases.items():
            for name in ('python', 'js') if NODE else ('python',):
                with self.subTest(case=case, sdk=name):
                    (self.server / ARCHIVE).write_bytes(self.archive_bytes)
                    good = make(self.shas)
                    w = self.work()
                    for title in ('Install the runtime', 'Install the packages', 'Download the weights', 'Write the launcher'):
                        r = self.step(good, title, w, name)
                        self.assertEqual(r.returncode, 0, (case, name, title, r.stderr[-1500:]))
                    (w / protected / 'KEEP').write_text('another step made this\n')
                    was = tree(w)
                    L = self.crafted([py, (f'{entry}/from-archive', 'file', 'x', 0o644)], make)
                    r = self.step(L, 'Install the runtime', w, name)
                    self.assertNotEqual(r.returncode, 0, (case, name))
                    self.assertIn('would replace', r.stderr, (case, name))
                    self.assertEqual(tree(w), was, (case, name))
                    self.assertEqual((w / protected / 'KEEP').read_text(), 'another step made this\n', (case, name))

    # r5: a hub launch without a runtime writes its launcher through the guarded writer, never through a link or outside

    def test_hub_only_launcher_never_writes_outside(self):
        def hub_only(config=None):
            L = {k: v for k, v in self.L().items() if k != 'runtime'}
            if config:
                L['config'] = {**L['config'], 'at': config}
            return L
        cases = {  # name: (launch for an outside dir, what to plant in the work dir)
            'launcher dir is a link out': (lambda o: hub_only(), lambda w, o: (w / 'launcher').symlink_to(o)),
            'launcher file is a link out': (lambda o: hub_only(), lambda w, o: ((w / 'launcher').mkdir(), (w / 'launcher/serve.py').symlink_to(o / 'victim'))),
            'launcher path is absolute, outside': (lambda o: hub_only(str(o / 'abs' / 'serve.py')), None),
        }
        for case, (make, plant) in cases.items():
            for name in ('python', 'js') if NODE else ('python',):
                with self.subTest(case=case, sdk=name):
                    o = self.d / f'outside-{self.n}'; self.n += 1  # a scratch dir of its own
                    o.mkdir(); (o / 'victim').write_text('untouched\n')
                    st = self.rendered_or_none(make(o), name)
                    if st is None:
                        self.assertIn('absolute', case)  # refused before anything runs
                        continue
                    w = self.work()
                    if plant:
                        plant(w, o)
                    code = next(s['code'] for s in st if s['title'] == 'Write the launcher')
                    r = self.sh(code, w)
                    self.assertEqual(tree(o), {'victim': hashlib.sha256(b'untouched\n').hexdigest()},
                                     f'{case} ({name}): the launcher was written outside the task')
                    self.assertNotEqual(r.returncode, 0, (case, name))

    def test_archive_links_and_members_are_checked(self):
        base = self.work(); self.install(base)
        before_ = tree(base / 'rt')
        py = ('python/bin/python3', 'file', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755)
        cases = {  # name: (members, the refusal it must give)
            'interpreter is an absolute link to an ambient python': ([('python/bin/python3', 'sym', sys.executable, 0o777)], 'is absolute'),
            'interpreter is a relative link out': ([('python/bin/python3', 'sym', '../../' * 12 + 'usr/bin/python3', 0o777)], 'leaves'),
            'interpreter is a dangling link': ([('python/bin/python3', 'sym', 'python3.12', 0o777)], 'dangling'),
            'a directory link out': ([py, ('python/lib', 'sym', '../..', 0o777)], 'leaves'),
            'a member that climbs': ([py, ('../evil', 'file', 'x', 0o644)], 'leaves the archive'),
            'an absolute member': ([py, ('/tmp/local-ai-evil', 'file', 'x', 0o644)], 'leaves the archive'),
            'a fifo': ([py, ('python/pipe', 'fifo', None, 0o644)], 'device, fifo or socket'),
            'a hard link out': ([py, ('python/h', 'hard', '/etc/hosts', 0o644)], ''),
        }
        for case, (members, says) in cases.items():
            L = self.crafted(members)
            for name in renderings(L):
                w = self.work(); shutil.rmtree(w); shutil.copytree(base, w, symlinks=True)
                was = tree(self.outside)
                r = self.step(L, 'Install the runtime', w, name)
                self.assertNotEqual(r.returncode, 0, (case, name))
                self.assertIn(says, r.stderr, (case, name))
                self.assertEqual(tree(w / 'rt'), before_, (case, name))
                self.assertEqual(tree(self.outside), was, (case, name))
                self.assertEqual(staging_left(w), [], (case, name))
                self.assertFalse(Path('/tmp/local-ai-evil').exists(), (case, name))

    def test_safe_internal_links_are_kept(self):
        L = self.crafted([('python/bin/python3.12', 'file', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755),
                          ('python/bin/python3', 'sym', 'python3.12', 0o777), ('python/lib', 'dir', None, 0o755),
                          ('python/lib/here', 'sym', '.', 0o777), ('python/lib/up', 'sym', '../bin/python3', 0o777),
                          ('python/bin/python3-hard', 'hard', 'python/bin/python3.12', 0o755)])
        for name in renderings(L):
            w = self.work()
            r = self.step(L, 'Install the runtime', w, name)
            self.assertEqual(r.returncode, 0, (name, r.stderr))
            self.assertEqual(os.readlink(w / 'rt/python/bin/python3'), 'python3.12', name)
            self.assertEqual(os.readlink(w / 'rt/python/lib/up'), '../bin/python3', name)
            ok = subprocess.run(['./rt/python/bin/python3', '-I', '-c', 'pass'], cwd=w)
            self.assertEqual(ok.returncode, 0, name)

    def test_launcher_over_a_directory_is_refused(self):
        for name in renderings(self.L()):
            w = self.work(); (w / 'launcher/serve.py').mkdir(parents=True)
            r = self.step(self.L(), 'Write the launcher', w, name)
            self.assertNotEqual(r.returncode, 0, name)
            self.assertEqual(staging_left(w), [], name)

    # r2: a bad archive or a failed publish leaves the whole previous runtime as it was, and usable

    def archive(self, entries):
        """Replace the served archive with one holding `entries` ({path: (text, mode)}); returns the launch pinning it."""
        src = self.d / f'arch{self.n}'; self.n += 1
        for rel, (text, mode) in entries.items():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_text(text); (src / rel).chmod(mode)
        subprocess.run(['tar', '-czf', str(self.server / ARCHIVE), '-C', str(src), *sorted({r.split('/')[0] for r in entries})], check=True)
        return self.L((hashlib.sha256((self.server / ARCHIVE).read_bytes()).hexdigest(),) + self.shas[1:])

    def test_bad_archive_or_failed_publish_keeps_the_runtime(self):
        base = self.work()
        for title in ('Install the runtime', 'Install the packages', 'Download the weights', 'Write the launcher'):
            self.assertEqual(self.step(self.L(), title, base).returncode, 0, title)
        before_ = tree(base / 'rt')
        good = (f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755)
        cases = {  # name: (archive entries, env)
            'no interpreter': ({'python/lib/x': ('', 0o644)}, {}),
            'interpreter not executable': ({'python/bin/python3': good[0:1] + (0o644,)}, {}),
            'interpreter that fails': ({'python/bin/python3': ('#!/bin/sh\nexit 3\n', 0o755)}, {}),
            'bad entry name': ({'python/bin/python3': good, 'bad name/x': ('', 0o644)}, {}),
            'reserved entry name': ({'python/bin/python3': good, '.local-ai-x/y': ('', 0o644)}, {}),
            'entry holds the packages target': ({'python/bin/python3': good, 'site/x': ('', 0o644)}, {}),
            'entry holds the hub cache': ({'python/bin/python3': good, 'hf-cache/x': ('', 0o644)}, {}),
            'publish fails after another entry was placed': ({'aaa/x': ('', 0o644), 'python/bin/python3': good}, {'FAIL_MV_TO': 'rt/python'}),
        }
        for case, (entries, env) in cases.items():
            L = self.archive(entries)
            for name in renderings(L):
                w = self.work(); shutil.rmtree(w); shutil.copytree(base, w, symlinks=True)
                r = self.step(L, 'Install the runtime', w, name, **env)
                self.assertNotEqual(r.returncode, 0, (case, name))
                self.assertEqual(tree(w / 'rt'), before_, (case, name))
                self.assertEqual(staging_left(w), [], (case, name))
                ok = subprocess.run(['rt/python/bin/python3', '-I', '-c', 'import sys; sys.path.insert(0, "rt/site"); import fixture_a, huggingface_hub'],
                                    cwd=w, capture_output=True, text=True)
                self.assertEqual(ok.returncode, 0, (case, name, ok.stderr))

    def test_failed_restore_keeps_the_previous_copy(self):
        base = self.work(); self.install(base)
        py = tree(base / 'rt/python')
        L = self.archive({'python/bin/python3': (f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755), 'zzz/x': ('', 0o644)})
        for name in renderings(L):
            w = self.work(); shutil.rmtree(w); shutil.copytree(base, w, symlinks=True)
            r = self.step(L, 'Install the runtime', w, name, FAIL_MV_TO='rt/python', FAIL_RESTORE='1')
            self.assertNotEqual(r.returncode, 0, name)
            kept = re.search(r'previous copy is kept in (\S+)', r.stderr)
            self.assertTrue(kept, (name, r.stderr))
            self.assertEqual(tree(w / kept.group(1) / 'python'), py, name)  # never deleted, so it can be put back by hand

    # r6: a signal in the middle of a publication restores the previous installation; if restoring fails, the backup is
    # kept and named. Every publication boundary of each stage, both SDKs, under sh and dash.

    BOUNDARIES = {  # name: (step, move after which the signal comes: from, to)
        'runtime: a new entry placed': ('Install the runtime', '*/x/aaa-new', 'rt/aaa-new'),
        'runtime: python backed up': ('Install the runtime', 'rt/python', '*/old/python'),
        'runtime: python placed': ('Install the runtime', '*/x/python', 'rt/python'),
        'packages: target backed up': ('Install the packages', 'rt/site', '*/old'),
        'packages: target placed': ('Install the packages', '*/t', 'rt/site'),
        'packages: wheel dir backed up': ('Install the packages', 'rt/wheels', '*/old'),
        'packages: wheel dir placed': ('Install the packages', '*.local-ai-wheels.*', 'rt/wheels'),
        'hub: model backed up': ('Download the weights', 'rt/hf-cache/models--org--model', '*/old'),
        'hub: model placed': ('Download the weights', '*/.local-ai-hub.*/models--org--model', 'rt/hf-cache/models--org--model'),
    }
    MARKED = ('rt/python', 'rt/site', 'rt/wheels', 'rt/hf-cache/models--org--model')

    def installed(self):
        """A complete installation, every output marked so a replaced or lost one shows; and the rerun's launch, whose
        archive adds an entry (aaa-new) so a half-published runtime shows too."""
        (self.server / ARCHIVE).write_bytes(self.archive_bytes)
        base = self.work()
        for title in ('Install the runtime', 'Install the packages', 'Download the weights', 'Write the launcher'):
            self.assertEqual(self.step(self.L(), title, base).returncode, 0, title)
        for rel in self.MARKED:
            (base / rel / 'KEEP').write_text('the previous installation\n')
        again = self.crafted([('aaa-new/x', 'file', 'new\n', 0o644),
                              ('python/bin/python3', 'file', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', 0o755)])
        return base, again

    def install_tree(self, w):
        """The installation itself (not the test's own records such as PIP_ENV in the work dir)."""
        return {'rt': tree(w / 'rt'), 'launcher': tree(w / 'launcher')}

    def interrupt(self, base, L, title, src, dst, name, shell, session=False, **env):
        w = self.work(); shutil.rmtree(w); shutil.copytree(base, w, symlinks=True)
        code = next(s['code'] for s in renderings(L)[name] if s['title'] == title)
        r = self.sh(code, w, shell, session, KILL_FROM=src, KILL_TO=dst, **env)
        self.assertTrue((w / 'KILLED').exists(), f'the boundary {src} -> {dst} was never reached')
        (w / 'KILLED').unlink()
        return w, r

    def test_signal_mid_publication_restores_the_previous_installation(self):
        base, again = self.installed()
        was = self.install_tree(base)
        for case, (title, src, dst) in self.BOUNDARIES.items():
            for name in ('python', 'js') if NODE else ('python',):
                for shell in [('sh', '-c')] + ([('dash', '-c')] if shutil.which('dash') else []):
                    with self.subTest(case=case, sdk=name, shell=shell[0]):
                        w, r = self.interrupt(base, again, title, src, dst, name, shell)
                        self.assertEqual(r.returncode, 143, (case, name, shell, r.stderr[-800:]))
                        self.assertEqual(self.install_tree(w), was, f'{case} ({name}, {shell[0]}): the previous installation was not restored')
                        self.assertEqual(staging_left(w), [], (case, name, shell))
                        ok = subprocess.run(['rt/python/bin/python3', '-I', '-c', 'import sys; sys.path.insert(0, "rt/site"); import fixture_a'], cwd=w)
                        self.assertEqual(ok.returncode, 0, (case, name, shell))

    def test_int_and_hup_mid_publication_restore_too(self):
        base, again = self.installed()
        was = self.install_tree(base)
        for sig, rc in (('INT', 130), ('HUP', 129)):
            for case in ('runtime: python backed up', 'packages: target placed', 'hub: model backed up'):
                title, src, dst = self.BOUNDARIES[case]
                for name in ('python', 'js') if NODE else ('python',):
                    for shell in [('sh', '-c')] + ([('dash', '-c')] if shutil.which('dash') else []):
                        with self.subTest(sig=sig, case=case, sdk=name, shell=shell[0]):
                            w, r = self.interrupt(base, again, title, src, dst, name, shell, KILL_SIG=sig)
                            self.assertEqual(r.returncode, rc, (sig, case, name, shell, r.stderr[-800:]))
                            self.assertEqual(self.install_tree(w), was, f'{sig} {case} ({name}, {shell[0]}): not restored')
                            self.assertEqual(staging_left(w), [], (sig, case, name, shell))

    def test_signal_with_a_failing_restore_keeps_the_backup(self):
        base, again = self.installed()
        cases = {'runtime: python backed up': 'rt/python', 'packages: target backed up': 'rt/site',
                 'packages: wheel dir backed up': 'rt/wheels', 'hub: model backed up': 'rt/hf-cache/models--org--model'}
        for case, live in cases.items():
            title, src, dst = self.BOUNDARIES[case]
            for name in ('python', 'js') if NODE else ('python',):
                with self.subTest(case=case, sdk=name):
                    w, r = self.interrupt(base, again, title, src, dst, name, ('sh', '-c'), FAIL_ANY_RESTORE='1')
                    self.assertNotEqual(r.returncode, 0, (case, name))
                    kept = re.findall(r'previous copy is kept in (\S+)', r.stderr)
                    self.assertTrue(kept, f'{case} ({name}): no backup named: {r.stderr[-600:]}')
                    # the named backup holds the previous output, byte for byte
                    held = [k for k in kept if tree(w / k / Path(live).name if case.startswith('runtime') else w / k) == tree(base / live)]
                    self.assertTrue(held, f'{case} ({name}): the previous {live} is not in {kept}')

    def test_repeated_term_during_rollback_cannot_cut_it_short(self):
        """A TERM mid-publication, then two more during the restore itself, sent to the step's shell alone or to its whole
        process group (its own session, so only test processes are signalled), under sh and dash: the previous installation
        comes back, or if restoring fails the exact backup path is printed and the backup is kept."""
        base, again = self.installed()
        was = self.install_tree(base)
        for case, live in (('runtime: python backed up', 'rt/python'), ('packages: target backed up', 'rt/site'),
                           ('hub: model backed up', 'rt/hf-cache/models--org--model')):
            title, src, dst = self.BOUNDARIES[case]
            for name in ('python', 'js') if NODE else ('python',):
                for shell in [('sh', '-c')] + ([('dash', '-c')] if shutil.which('dash') else []):
                    for to in ('parent', 'group'):
                        for fail in (False, True):
                            with self.subTest(case=case, sdk=name, shell=shell[0], to=to, restore_fails=fail):
                                w, r = self.interrupt(base, again, title, src, dst, name, shell, True, RETERM=to,
                                                      **({'FAIL_ANY_RESTORE': '1'} if fail else {}))
                                self.assertTrue((w / 'RETERMED').exists(), 'the rollback never reached its restore move')
                                (w / 'RETERMED').unlink()
                                if not fail:
                                    self.assertEqual(self.install_tree(w), was, r.stderr[-800:])
                                    self.assertEqual(staging_left(w), [])
                                    if to == 'parent':
                                        self.assertEqual(r.returncode, 143, r.stderr[-800:])
                                    continue
                                self.assertNotEqual(r.returncode, 0)
                                kept = re.findall(r'previous copy is kept in (\S+)', r.stderr)
                                self.assertTrue(kept, f'no backup named: {r.stderr[-600:]}')
                                self.assertTrue([k for k in kept if tree(w / k / Path(live).name if case.startswith('runtime') else w / k) == tree(base / live)],
                                                f'the previous {live} is not in {kept}')

    def test_failed_package_publish_restores_both(self):
        base = self.work(); self.install(base)
        site, wheels = tree(base / 'rt/site'), tree(base / 'rt/wheels')
        for dest in ('rt/site', 'rt/wheels'):
            for name in renderings(self.L()):
                w = self.work(); shutil.rmtree(w); shutil.copytree(base, w, symlinks=True)
                r = self.step(self.L(), 'Install the packages', w, name, FAIL_MV_TO=dest)
                self.assertNotEqual(r.returncode, 0, (dest, name))
                self.assertEqual((tree(w / 'rt/site'), tree(w / 'rt/wheels')), (site, wheels), (dest, name))
                self.assertEqual(staging_left(w), [], (dest, name))

    def test_missing_snapshot_writes_no_ref(self):
        w = self.work(); self.install(w)
        r = self.step(self.L(), 'Download the weights', w, FAKE_HF_MISS='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((w / 'rt/hf-cache/models--org--model/refs').exists())


if __name__ == '__main__':
    unittest.main()
