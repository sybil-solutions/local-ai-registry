#!/usr/bin/env python3
"""Qualify one configured Inkling profile. Authoring/check-only never starts a GPU job.

The default run is for the 128K recipe. All generation requests are uncapped.
Official registry gates and cross-run greedy equivalence remain separate gates.
"""
import argparse
import ast
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import random
import re
import shutil
import signal
import statistics
import subprocess as sp
import sys
import time
from pathlib import Path

import runner
import core

HERE = Path(__file__).resolve().parent
STAGES = ('panels', 'recall', 'marker', 'smoke', 'daily', 'soak')
GRADER_IMAGE = 'vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e'
MARKER = 'cobalt lantern 7319'
PROBE = 'Explain how a write-ahead log supports recovery after a crash, in about 400 words, using one concrete example.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def assert_uncapped(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ('max_tokens', 'max_completion_tokens', 'max_new_tokens'):
            raise ValueError('Output cap survived source adaptation')
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in ('max_tokens', 'max_completion_tokens', 'max_new_tokens'):
                    raise ValueError('Output cap survived source adaptation')


@contextlib.contextmanager
def supplied_prompt(prompt, expected=None, temperature=0):
    original, old_temp = core.prompt, core.SAMPLING_TEMP
    core.prompt = lambda *args, **kwargs: (prompt, expected)
    core.SAMPLING_TEMP = temperature
    try:
        yield
    finally:
        core.prompt, core.SAMPLING_TEMP = original, old_temp


def make_needles(size, seed):
    # core.prompt sizes the reference with the actual server tokenizer before timing.
    base, _ = core.prompt('prefill', size, seed)
    reference = base.split('\nEND OF REFERENCE DOCUMENT.', 1)[0]
    rng = random.Random(seed ^ 0x71C09)
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    codes = {f'code{i + 1}': ''.join(rng.choices(alphabet, k=12)) for i in range(3)}
    chunks, previous = [], 0
    for fraction, (key, value) in zip((.1, .5, .9), codes.items()):
        position = int(len(reference) * fraction)
        chunks.extend((reference[previous:position], f'\nThe exact lookup value for {key} is {value}.\n'))
        previous = position
    chunks.append(reference[previous:])
    text = ''.join(chunks)
    prompt = text + '\nEND OF REFERENCE DOCUMENT.\nReturn a JSON object with exactly the keys code1, code2 and code3 and their exact lookup values. No explanation, code fence or other text.'
    positions = {key: core.count(prompt[:prompt.index(value)]) for key, value in codes.items()}
    return prompt, codes, {'raw_prompt_tokens': core.count(prompt), 'needle_token_positions': positions,
                           'target_input': size, 'seed': seed, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}


def log_slice(start, path):
    since = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    result = sp.run(['docker', 'logs', '--timestamps', '--since', since, runner.NAME], capture_output=True, text=True)
    text = result.stdout + result.stderr
    path.write_text(text)
    # Keep raw lines: positive values during chunking are not automatically prefix-cache hits.
    return {'log_file': path.name, 'log_exit_code': result.returncode,
            'cached_token_log_values': [int(x) for x in re.findall(r'#cached-token:\s*(\d+)', text)]}


def direct_cached_tokens(row):
    details = (row.get('usage') or {}).get('prompt_tokens_details') or {}
    value = details.get('cached_tokens')
    return value if isinstance(value, (int, float)) else None


def needle_request(name, phase, size, seed, watch, out, warm_pair=False):
    prompt, expected, metadata = make_needles(size, seed)
    trials = []
    for label, cold in [('cold', True)] + ([('warm', False)] if warm_pair else []):
        start = time.time()
        with supplied_prompt(prompt, temperature=0):
            row = runner.sample(name, phase + '-' + label, 'multineedle', size, seed, watch, cold=cold)
        try:
            actual = json.loads(row.get('content', '').strip())
        except (ValueError, TypeError):
            actual = None
        valid_size = .95 * size <= (row.get('usage') or {}).get('prompt_tokens', 0) <= size + 128
        grade = {**metadata, 'label': label, 'phase': phase, 'expected': expected, 'actual': actual,
                 'exact_match': actual == expected, 'input_size_valid': valid_size,
                 'flags': row['flags'], 'finish': row.get('finish'), 'ttft': row.get('ttft'),
                 'seconds': row.get('seconds'), 'usage': row.get('usage'),
                 'output_sha256': row['output_sha256'], 'direct_cached_tokens': direct_cached_tokens(row),
                 **log_slice(start, out / f'{phase}-{size}-{seed}-{label}-server.log')}
        grade['passed'] = grade['exact_match'] and valid_size and not row['flags']
        core.append(out / 'recall-grades.jsonl', grade)
        trials.append(grade)
    if warm_pair:
        cold, warm = trials
        cold_values, warm_values = cold['cached_token_log_values'], warm['cached_token_log_values']
        direct = warm['direct_cached_tokens']
        if direct is not None:
            evidence = 'usage counter'; observed = direct > 0
        elif cold_values and warm_values and max(cold_values) == 0 and max(warm_values) > 0:
            evidence = 'warm positive/cold zero server counters'; observed = True
        else:
            evidence = 'unconfirmed; inspect raw counters'; observed = None
        core.append(out / 'cache-evidence.jsonl', {'target_input': size, 'seed': seed,
            'prompt_sha256': metadata['prompt_sha256'], 'cold_ttft': cold['ttft'], 'warm_ttft': warm['ttft'],
            'cache_hit_observed': observed, 'evidence': evidence,
            'cold_cached_log_values': cold_values, 'warm_cached_log_values': warm_values,
            'warm_direct_cached_tokens': direct})
    return trials


def freeze_sources(out, daily_root, smoke_path):
    frozen = out / 'frozen'; frozen.mkdir()
    manifest = {'files': {}, 'adaptations': []}
    for path in (HERE / 'runner.py', HERE / 'core.py', Path(__file__)):
        target = frozen / path.name
        shutil.copyfile(path, target)
        manifest['files'][str(target.relative_to(out))] = sha(target)
    smoke = smoke_path.read_text()
    assert smoke.count(', max_tokens=2048') == 1 and smoke.count(",'max_tokens':512") == 1
    adapted = smoke.replace('127.0.0.1:8101', '127.0.0.1:8102').replace(', max_tokens=2048', '').replace(",'max_tokens':512", '')
    assert_uncapped(adapted)
    (frozen / 'smoke-original.py').write_text(smoke)
    (frozen / 'smoke-uncapped.py').write_text(adapted)
    manifest['adaptations'].append('Smoke: endpoint8102 and removal of both generation caps; assertions unchanged.')
    for relative in ('benchmarks/daily/run.py', 'benchmarks/daily/grade_repairs.py', 'public/lab/daily/tasks.json'):
        target = frozen / 'daily-repo' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(daily_root / relative, target)
    original = (frozen / 'daily-repo/benchmarks/daily/run.py').read_text()
    cap_line = '            "max_tokens": args.max_tokens,\n'
    assert original.count(cap_line) == 1
    adapted = original.replace(cap_line, '')
    assert_uncapped(adapted)
    (frozen / 'daily-repo/benchmarks/daily/run-uncapped.py').write_text(adapted)
    manifest['adaptations'].append('Daily: removed the sole max_tokens request field; prompts, order, temperature and rubric unchanged. CLI budget option remains unused.')
    for path in frozen.rglob('*'):
        if path.is_file(): manifest['files'][str(path.relative_to(out))] = sha(path)
    if manifest['files']['frozen/runner.py'] != runner.RUNNER_HASH or manifest['files']['frozen/core.py'] != runner.CORE_HASH:
        raise RuntimeError('Harness source changed after import; preserve this attempt and restart with a new output directory')
    write_json(out / 'source-manifest.json', manifest)
    return frozen


def run_script(command, log_path, watch, timeout):
    before = runner.environment(runner.NAME)
    with log_path.open('w') as stream:
        result = sp.run(command, stdout=stream, stderr=sp.STDOUT, timeout=timeout)
    after = runner.environment(runner.NAME)
    return {'exit_code': result.returncode, 'environment_valid': runner.stable(before, after),
            'memory_cap_tripped': watch.tripped, 'watch_error': watch.error,
            'passed': result.returncode == 0 and runner.stable(before, after) and not watch.tripped and not watch.error}


def grade_daily(out, frozen):
    tasks = {t['id']: t for t in json.loads((frozen / 'daily-repo/public/lab/daily/tasks.json').read_text())}
    rows = records(out / 'daily.jsonl')
    repairs, review = {}, []
    for row in rows:
        task = tasks[row['task']]
        digest = hashlib.sha256((row['task'] + '\0' + row['answer']).encode()).hexdigest()
        if 'function' in task:
            repairs[digest] = {'digest': digest, 'fixture': task, 'answer': row['answer']}
        else:
            review.append({'task': row['task'], 'repeat': row['repeat'], 'rubric': task['rubric'],
                           'answer': row['answer'], 'finish': row.get('finish'), 'digest': digest,
                           'grade': None, 'status': 'pending_manual_review'})
    write_json(out / 'daily-manual-review.json', review)
    command = ['docker', 'run', '--rm', '--pull=never', '-i', '--network', 'none', '--read-only',
        '--memory', '256m', '--cpus', '1', '--pids-limit', '64', '--ulimit', 'cpu=10:10', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', '65534:65534',
        '-v', str(frozen / 'daily-repo/benchmarks/daily') + ':/grade:ro',
        '--entrypoint', 'python3', GRADER_IMAGE, '/grade/grade_repairs.py']
    result = sp.run(command, input=json.dumps(list(repairs.values())), text=True, capture_output=True, timeout=180)
    (out / 'daily-repair-grades.jsonl').write_text(result.stdout)
    (out / 'daily-grading.log').write_text(result.stderr)
    grades = {g['digest']: g for g in map(json.loads, result.stdout.splitlines())} if result.returncode == 0 else {}
    repair_rows = [r for r in rows if 'function' in tasks[r['task']]]
    passed = sum(grades.get(hashlib.sha256((r['task'] + '\0' + r['answer']).encode()).hexdigest(), {}).get('passed', False) for r in repair_rows)
    expected = {(t, repeat) for t in tasks for repeat in (1, 2)}
    result_summary = {'responses': len(rows), 'complete_task_set': len(rows) == len(expected) and {(r['task'], r['repeat']) for r in rows} == expected,
        'natural_stops': sum(r.get('finish') == 'stop' for r in rows), 'repair_attempts': len(repair_rows),
        'repair_passed': passed, 'grader_exit_code': result.returncode, 'manual_review_pending': len(review),
        'overall_score': None}
    write_json(out / 'daily-summary.json', result_summary)
    return result_summary


def soak(name, seconds, size, watch, out):
    start = time.monotonic(); cycles = 0
    core.PROMPTS['soak_probe'] = PROBE
    while time.monotonic() - start < seconds:
        needle_request(name, 'soak-recall', size, 80000 + cycles, watch, out)
        core.SAMPLING_TEMP = 0
        try:
            probe = runner.sample(name, 'soak-probe', 'soak_probe', 8192, 90000, watch)
        finally:
            core.SAMPLING_TEMP = .8
        core.append(out / 'soak-probes.jsonl', {'cycle': cycles, 'seed': 90000, 'temperature': 0,
            'unix_start': probe['unix_start'], 'prompt_sha256': probe['prompt_sha256'],
            'output_sha256': probe['output_sha256'], 'usage': probe.get('usage'),
            'usage_decode_tps': probe.get('usage_decode_tps'), 'seconds': probe['seconds'], 'flags': probe['flags']})
        cycles += 1
    rows = [r for r in records(out / 'ledger.jsonl') if r['phase'].startswith('soak-')]
    probes = [r for r in rows if r['phase'] == 'soak-probe']
    rate_probes = [r for r in probes if isinstance(r.get('usage_decode_tps'), (int, float)) and math.isfinite(r['usage_decode_tps']) and r['usage_decode_tps'] > 0]
    duration = time.monotonic() - start
    hashes = [r['output_sha256'] for r in probes]
    hash_counts = {h: hashes.count(h) for h in sorted(set(hashes))}
    hashes_equal = len(hash_counts) == 1 if len(hashes) >= 2 else None
    result = {'requested_seconds': seconds, 'wall_seconds': duration, 'cycles': cycles,
        'requests': len(rows), 'request_seconds': sum(r['seconds'] for r in rows),
        'probe_seed': 90000, 'probe_temperature': 0, 'probe_output_hashes_equal': hashes_equal,
        'probe_output_hash_counts': hash_counts,
        'probe_output_variation': [{'index': i, 'output_sha256': r['output_sha256'], 'usage': r.get('usage')} for i, r in enumerate(probes)],
        'request_flags': [{'phase': r['phase'], 'seed': r['seed'], 'flags': r['flags']} for r in rows if r['flags']],
        'recall_failures': sum(not r['passed'] for r in records(out / 'recall-grades.jsonl') if r['phase'] == 'soak-recall')}
    # Same probe/task/size in each comparison; report both ends even if a probe is flagged.
    if len(rate_probes) >= 3:
        n = max(1, len(rate_probes) // 3)
        first = statistics.median(r['usage_decode_tps'] for r in rate_probes[:n])
        last = statistics.median(r['usage_decode_tps'] for r in rate_probes[-n:])
        result['probe_drift'] = {'samples_per_end': n, 'probes_without_valid_rate': len(probes) - len(rate_probes), 'first_tps': first, 'last_tps': last,
            'change_percent': 100 * (last / first - 1), 'identical_outputs': hashes_equal,
            'attribution': 'matched deterministic outputs' if hashes_equal else 'output variation prevents attributing rate drift solely to runtime',
            'all_probe_flags': [r['flags'] for r in probes]}
    result['passed'] = duration >= seconds and not result['request_flags'] and not result['recall_failures'] and not watch.tripped and not watch.error
    write_json(out / 'soak-summary.json', result)
    return result


def qualify(args):
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = {'name': args.name, 'config': cfg, 'config_sha256': sha(args.config), 'sampling_temperature': .8,
        'panel_lengths': args.panel_lengths, 'recall_lengths': args.recall_lengths,
        'soak_context': args.soak_context, 'soak_seconds': args.soak_seconds, 'stages_requested': args.stages,
        'completed_stages': [], 'official_registry_gates': 'not_run_here', 'cross_run_greedy_equivalence': 'separate_gate',
        'status': 'waiting_for_gpu_lock'}
    write_json(out / 'result.json', result)
    frozen = freeze_sources(out, args.daily_root, args.smoke_source)
    lock = open('/tmp/inkling-gpu.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX)
    core.ROOT = out; runner.ROOT = out; core.SAMPLING_TEMP = .8
    watch = runner.Watch(out / 'telemetry.jsonl', cap_mib=args.memory_cap_mib)
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        core.ready(8101, 'inkling-mini'); runner.drain_stop('inkling-mini', 8101)
        result['environment_before'] = runner.environment()
        watch.start(); result['startup_seconds'] = runner.launch(cfg, out)
        core.SAMPLING_TEMP = .8
        runner.sample(args.name, 'warmup', 'lru', 8192, 70000, watch)
        result['status'] = 'running'; write_json(out / 'result.json', result)
        for stage in args.stages:
            if stage == 'panels':
                for size in args.panel_lengths:
                    for i, kind in enumerate(runner.PANEL):
                        core.SAMPLING_TEMP = .8
                        runner.sample(args.name, 'panel', kind, size, 71000 + size + i, watch)
            elif stage == 'recall':
                for size in args.recall_lengths:
                    for seed in (72001, 72002):
                        needle_request(args.name, 'recall', size, seed, watch, out, warm_pair=True)
            elif stage == 'marker':
                p = 'Return exactly the text between quotation marks, without the quotation marks: "' + MARKER + '"'
                with supplied_prompt(p, expected=MARKER, temperature=0):
                    row = runner.sample(args.name, 'marker', 'quoted_marker', 0, 73000, watch)
                result['marker_passed'] = row['content'].strip() == MARKER and not row['flags']
            elif stage == 'smoke':
                result['smoke'] = run_script([sys.executable, str(frozen / 'smoke-uncapped.py')], out / 'smoke.log', watch, 1200)
            elif stage == 'daily':
                result['daily_replay'] = run_script([sys.executable, str(frozen / 'daily-repo/benchmarks/daily/run-uncapped.py'),
                    '--base-url', runner.URL + '/v1', '--model', 'Inkling-2-Mini', '--output', str(out / 'daily.jsonl')], out / 'daily.log', watch, 3600)
            elif stage == 'soak':
                result['soak'] = soak(args.name, args.soak_seconds, args.soak_context, watch, out)
            result['completed_stages'].append(stage); write_json(out / 'result.json', result)
            print(json.dumps({'stage_completed': stage, 'qualification': args.name}), flush=True)
        result['environment_after'] = runner.environment(runner.NAME)
        result['environment_valid'] = runner.stable(result['environment_before'], result['environment_after'])
        result['status'] = 'measurements_complete'
    except BaseException as exc:
        result['status'] = 'failed'; result['error'] = repr(exc)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # A second TERM must not interrupt restoration.
        core.SAMPLING_TEMP = .8
        watch.end(); result['memory_cap_tripped'] = watch.tripped; result['watch_error'] = watch.error
        try:
            if runner.container_exists(runner.NAME):
                log = sp.run(['docker', 'logs', runner.NAME], capture_output=True, text=True, timeout=30)
                (out / 'server.log').write_text(log.stdout + log.stderr)
        except Exception as exc:
            result['log_capture_error'] = repr(exc)
        try:
            runner.drain_stop(runner.NAME, 8102)
        except Exception as exc:
            result['cleanup_error'] = repr(exc)
        finally:
            try:
                if runner.container_exists(runner.NAME) and sp.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', runner.NAME], text=True).strip() == 'true':
                    raise RuntimeError('Trial still running; refusing overlapping GPU owners')
                sp.run(['docker', 'start', 'inkling-mini'], check=True, stdout=sp.DEVNULL)
                core.ready(8101, 'inkling-mini'); result['restored'] = True
            except Exception as exc:
                result['restoration_error'] = repr(exc)
        write_json(out / 'result.json', result)
    # CPU-only sandbox grading happens after timed GPU work and safe restoration.
    if (out / 'daily.jsonl').exists():
        try: result['daily_grading'] = grade_daily(out, frozen)
        except Exception as exc: result['daily_grading_error'] = repr(exc)
    rows = records(out / 'ledger.jsonl')
    panel_review = [{'phase': r['phase'], 'kind': r['kind'], 'target_input': r['target_input'],
                     'seed': r['seed'], 'task': runner.PANEL[r['kind']], 'answer': r.get('content', ''),
                     'flags': r['flags'], 'grade': None, 'status': 'pending_manual_semantic_review'}
                    for r in rows if r['phase'] == 'panel']
    write_json(out / 'panel-manual-review.json', panel_review)
    result['request_flags'] = [{'phase': r['phase'], 'kind': r['kind'], 'seed': r['seed'], 'flags': r['flags']} for r in rows if r['phase'] != 'warmup' and r['flags']]
    result['recall_passed'] = all(r['passed'] for r in records(out / 'recall-grades.jsonl')) if 'recall' in result['completed_stages'] else False
    d = result.get('daily_grading', {})
    result['automated_gates_passed'] = bool(set(result['completed_stages']) == set(STAGES) and not result.get('error') and not result.get('cleanup_error')
        and result.get('restored') and result.get('environment_valid') and not watch.tripped and not watch.error
        and not result['request_flags'] and result['recall_passed'] and result.get('marker_passed')
        and result.get('smoke', {}).get('passed') and result.get('daily_replay', {}).get('passed')
        and d.get('complete_task_set') and d.get('natural_stops') == 24 and d.get('repair_passed') == d.get('repair_attempts') == 8
        and result.get('soak', {}).get('passed') and args.soak_seconds >= 900)
    result['qualified_for_publication'] = False
    result['remaining_review'] = ['Manual grading of nonrepair Daily responses', 'Cross-run greedy equivalence', 'Official registry gates on identical published launch', 'Semantic review of free-form panel and generated code responses']
    write_json(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['automated_gates_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel-lengths', type=int, nargs='+', default=[32768, 114688])
    parser.add_argument('--recall-lengths', type=int, nargs='+', default=[65536, 126000])
    parser.add_argument('--soak-context', type=int, default=126000)
    parser.add_argument('--soak-seconds', type=int, default=900)
    parser.add_argument('--memory-cap-mib', type=int, default=32512)
    parser.add_argument('--stages', choices=STAGES, nargs='+', default=list(STAGES))
    parser.add_argument('--daily-root', type=Path, default=Path.home() / 'Work/karimi.info')
    parser.add_argument('--smoke-source', type=Path, default=Path.home() / '.local/share/inkling/smoke.py')
    parser.add_argument('--check-only', action='store_true', help='Validate local inputs and source adaptations; do not use Docker, GPU, HTTP or the lock.')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    launch_args = list(runner.BASE['args']) + cfg.get('extra_args', [])
    for key, value in cfg.get('replace_args', {}).items(): launch_args[launch_args.index(key) + 1] = str(value)
    context = int([launch_args[i + 1] for i, value in enumerate(launch_args[:-1]) if value == '--context-length'][-1])
    sizes = args.panel_lengths + args.recall_lengths + [args.soak_context]
    if min(sizes) < 1024 or max(sizes) > context - 512: parser.error('Prompt targets must leave at least512 tokens of context for answers')
    if len(set(args.stages)) != len(args.stages): parser.error('Stages must not be repeated')
    if args.soak_seconds <= 0 or args.memory_cap_mib <= 0: parser.error('Duration and memory cap must be positive')
    for source in (args.smoke_source, args.daily_root / 'benchmarks/daily/run.py', args.daily_root / 'benchmarks/daily/grade_repairs.py', args.daily_root / 'public/lab/daily/tasks.json'):
        if not source.is_file(): parser.error('Missing local fixture: ' + str(source))
    if args.check_only:
        smoke = args.smoke_source.read_text().replace(', max_tokens=2048', '').replace(",'max_tokens':512", '')
        daily = (args.daily_root / 'benchmarks/daily/run.py').read_text().replace('            "max_tokens": args.max_tokens,\n', '')
        assert_uncapped(smoke); assert_uncapped(daily)
        print(json.dumps({'check_only': True, 'context': context, 'uncapped_sources_parse': True, 'stages': args.stages})); return 0
    return qualify(args)


if __name__ == '__main__':
    sys.exit(main())
