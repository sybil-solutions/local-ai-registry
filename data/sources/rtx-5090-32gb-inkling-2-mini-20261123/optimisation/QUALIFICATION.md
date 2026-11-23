# Qualification harness

`qualification.py` qualifies one selected **128K, single-request** profile using the existing runner's launch, environment checks, memory watchdog, GPU lock and safe drain. It does not edit the registry or the daily launch. The normal `inkling-mini` container is restored on exit; the harness refuses to start it if the trial still runs.

Authoring validation (no Docker, HTTP, GPU job or lock acquisition):

```sh
python3 qualification.py --config configs/control-a.json \
  --name input-check --output /tmp/inkling-input-check --check-only
```

Full run, after selecting a profile from the control/candidate results:

```sh
setsid nohup python3 qualification.py --config configs/cache-warm-b.json \
  --name retained-profile --output "$PWD/qualification/retained-profile" \
  > qualification-retained-profile.log 2>&1 < /dev/null &
```

The shared `/tmp/inkling-gpu.lock` makes this wait for the existing queue. Every output directory must be new. A failed or interrupted run is preserved, never overwritten. The `--config` may name a baseline configuration instead. The emitted `launch.json` and `profile.json` must match the profile intended for publication; this script does not choose a winner.

Default stages:

- Six fixed uncapped panel requests at approximately 32K and 112K input. Earlier 8K screening stays a separate artifact. Actual server token counts and completion/loop flags are retained.
- Two seeds at approximately 64K and 126K input. Each document has three exact random codes at roughly 10%, 50% and 90% depth; tokenizer-measured positions are recorded. A strict JSON response must contain exactly the expected codes.
- A cold and identical warm request for each document. `cache-evidence.jsonl` records direct usage counters if exposed, raw server counter values and TTFT. Faster warm timing alone never counts as proof of a prefix-cache hit. Ambiguous counters remain unconfirmed.
- An explicitly quoted marker with an exact visible-answer check.
- Frozen copies of the existing tools/vision/text/stream smoke and Daily fixtures. Only the local endpoint and output-cap fields are adapted. AST checks reject any surviving generation cap. Smoke assertions and Daily prompts, order and temperature stay unchanged. These uncapped Daily results must not be conflated with the published Daily4096 benchmark.
- A 900-second c1 soak. Each cycle sends a fresh cold near-full-context three-code retrieval, then the identical 8K write-ahead-log task and context at seed90000 and temperature0, with a cold cache. Infrastructure failure stops safely; semantic, loop and length failures remain in the evidence and fail qualification. Wall duration includes prompt preparation; summed request time is reported separately. Drift compares the deterministic probe in its first and last thirds, with sample counts, full reasoning-plus-content hash equality and any output variation. If outputs differ, the report explicitly refuses to attribute rate drift solely to runtime.

Sampling panels explicitly use temperature0.8. The fixed soak probe, exact retrieval and quoted marker use temperature0. The harness makes no concurrency claim beyond one request at a time. It leaves clocks, power limit, system profile and the daily4096 budget untouched.

`--stages`, lengths and soak duration can run a diagnostic subset, but incomplete stages or a soak below900 seconds cannot set `automated_gates_passed`. Custom shorter lengths can be supplied for a separate48K diagnostic; such a run does not qualify the128K recipe.

## Evidence and remaining review

Each run stores source copies and hashes, the exact launch, per-request raw responses and timing, telemetry, server logs, context grades, cache evidence, smoke output and Daily answers. The offline repair grader has no network/GPU access and runs read-only with CPU, memory and process limits. It grades the eight repair attempts. The other16 Daily answers are copied alongside their rubrics to `daily-manual-review.json`; they are deliberately **not** automatically declared correct.

`automated_gates_passed` is a strict aggregate of the implemented checks. `qualified_for_publication` stays false: the parent still needs to grade nonrepair answers, confirm cross-run full-output greedy equality, and run the official registry gates on the identical published launch. Official gates are left outside this script so it cannot silently change registry evidence or validate a different profile.

The provided smoke checks one small blue image. It is a basic vision smoke check, not a general image-quality benchmark. Loop detection and natural completion do not prove that free-form answers are semantically correct. Free-form panels and generated code are copied to `panel-manual-review.json` as ungraded; generated shell/code output is never executed by the performance harness. The saved outputs permit further review.

## CPU-only authoring checks performed

- Python syntax compilation and `--check-only` passed.
- Source freezing preserved nine files and removed request caps without altering the originals.
- Mocked exact-code grading and cold/warm requests preserved identical prompts.
- Ambiguous positive cache counters remained unconfirmed rather than being promoted to a hit.
- No GPU requests, Docker operations or changes to the running queue were made while authoring this harness.
