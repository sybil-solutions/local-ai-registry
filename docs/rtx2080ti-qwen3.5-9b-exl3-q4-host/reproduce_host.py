"""Reproduce the measured host launch; run on a Linux compute node, never login."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('action', choices=['setup', 'serve', 'gates'])
ap.add_argument('--run-dir', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--profile', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--registry', type=Path, help='Patched fixed registry checkout, for gates')
ap.add_argument('--python', default=sys.executable, help='Linux Python 3.12 for creating runtime')
ap.add_argument('--virtualenv-pyz', type=Path, help='Prefer the bundled exact virtualenv.pyz')
args = ap.parse_args()
assert os.name == 'posix', 'This launch uses the measured Linux CUDA wheel'
root = args.run_dir.resolve()
model_dir = args.model_dir.resolve()
profile = json.loads(args.profile.read_text())
manifest = json.loads(args.manifest.read_text())
assert profile['id'] == 'tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4'
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, **profile['env'])
for key, value in profile['env'].items():
    if key.endswith(('_DIR', '_HOME')) or key == 'TMPDIR':
        (root / value).mkdir(parents=True, exist_ok=True)
runtime = root / 'runtime/bin/python'
source_sha = '01f939c0ee10f10b5f5350047462effad01a4274'
tabby_url = 'https://github.com/theroyallab/tabbyAPI/archive/' + source_sha + '.tar.gz'
tabby_sha = '75ee0964ceb7e6086f7018d30f12c0dbac4d4c5977926a8edb5d2b5ff4928792'
virtualenv_sha = '215e29d6c8e2a4bf5f509828368629bbe7eddb847dcfee862db0f84e6df40047'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get(url, destination, sha):
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + '.part')
        with urllib.request.urlopen(url, timeout=180) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        assert digest(partial) == sha, (destination.name, 'download hash mismatch')
        partial.replace(destination)
    assert digest(destination) == sha, (destination.name, 'hash mismatch')


def run(command):
    print(json.dumps({'command': [str(x) for x in command]}), flush=True)
    subprocess.run([str(x) for x in command], cwd=root, env=env, check=True)


if args.action == 'setup':
    for row in manifest['files']:
        assert Path(row['name']).name == row['name']
        path = model_dir / row['name']
        assert path.stat().st_size == row['bytes'], path
        assert digest(path) == row['sha256'], path
    virtualenv = root / 'virtualenv.pyz'
    if args.virtualenv_pyz:
        assert digest(args.virtualenv_pyz) == virtualenv_sha
        if args.virtualenv_pyz.resolve() != virtualenv:
            shutil.copy2(args.virtualenv_pyz, virtualenv)
    get('https://bootstrap.pypa.io/virtualenv.pyz', virtualenv, virtualenv_sha)
    if not runtime.exists():
        run([args.python, virtualenv, '--app-data', 'cache/virtualenv',
             '--no-periodic-update', '--no-vcs-ignore', 'runtime'])
    run([runtime, '-m', 'pip', 'install', '--index-url',
         'https://download.pytorch.org/whl/cu128', 'torch==2.8.0+cu128'])
    archive = root / 'tabby.tar.gz'
    get(tabby_url, archive, tabby_sha)
    if not (root / 'tabby').exists():
        with tarfile.open(archive) as source:
            source.extractall(root, filter='data')
        (root / ('tabbyAPI-' + source_sha)).rename(root / 'tabby')
    lock = root / 'requirements-measured.txt'
    lock.write_text('\n'.join(profile['pip']) + '\n')
    run([runtime, '-m', 'pip', 'install', '--no-deps', '-r', lock])
    weight_records = profile['weights'] if isinstance(profile['weights'], list) else [profile['weights']]
    assert len(weight_records) == 1
    destination = root / weight_records[0]['at']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve() == model_dir
    elif destination.exists():
        assert destination.resolve() == model_dir, destination
    else:
        destination.symlink_to(model_dir, target_is_directory=True)
    config = root / profile['config']['at']
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(profile['config']['text'])
    (root / 'pip-freeze-reproduced.txt').write_text(subprocess.check_output(
        [str(runtime), '-m', 'pip', 'freeze'], cwd=root, env=env, text=True))
    print('SETUP_COMPLETE; serve uses the unchanged measured profile command/config')
elif args.action == 'serve':
    assert runtime.exists(), 'Run setup first'
    config = root / profile['config']['at']
    assert config.read_text() == profile['config']['text'], 'Measured config changed'
    os.chdir(root)
    os.execvpe(profile['command'][0], profile['command'], env)
else:
    assert args.registry, '--registry is required for gates'
    command = [str(runtime), str(args.registry.resolve() / 'lab/lab.py'), 'try',
        manifest['repo'] + '@' + manifest['revision'], '--model', 'qwen3.5-9b',
        '--engine', profile['id'], '--card', 'rtx-2080-ti-11gb', '--on', 'endpoint',
        '--endpoint', 'http://127.0.0.1:5000', '--gpu', 'NVIDIA GeForce RTX 2080 Ti']
    run(command)
