# Apple M1 8 GB: LFM2.5-2.6B on MLX (mlx-lm 0.31.3, uncapped), a 32K window and a 24,576-token prompt limit, on the SDK's pinned host runtime

**Depends on #361** (the pinned host runtime; head `ff00ce61`, which is SDK `7641917e`, the 2026-10-25 catalog date refresh `40bb285b`
and a test-only correction to `lab/test_host_topology.py` for launches that pin a runtime).
This launch uses #361's `runtime` field and hub-layout weights with `ref`, and every setup step below is the SDK's own rendering of it.
This commit adds one optional field to that contract, `runtime.macos` (below).

**A proposal that needs a maintainer ruling before merge.** `docs/data.md` says new recipes use EXL3, NVFP4, FP8 or AWQ, and it does
not name Apple MLX. This MLX (safetensors) recipe is sent at the hardware owner's direction ("Add them anyway"; "Send it in, but make a
note that you are not sure"). No exemption is claimed, and the policy text is not changed here. The precedent is the registry's existing
Apple MLX host launches: `registry/launches/mlx-vlm-qwen3.8-27b-3bit-mtp-200k.json`, `mlx-vlm-qwen3.8-27b-4bit-mtp-200k.json`,
`mlx-vlm-qwen3.8-27b-4bit-mtp-200k-m2-max.json`, `mlx-vlm-qwen3.8-27b-4bit-mtp-fused-200k.json`, `mtplx-qwen3.8-27b-4bit-mtp-200k.json`,
`omlx-glm-5.3-flash-4bit-mtp-1024k.json`, `omlx-qwen3.8-27b-oq4e-mtp-256k.json` and `omlx-qwen3.8-27b-oq4e-mtp3-200k.json`. Their
recipes include `registry/recipes/apple/apple-m2-max-96gb/qwen3.8-27b.mlx-vlm.200k.json` and
`registry/recipes/apple/apple-m4-pro-48gb/qwen3.8-27b.omlx.195k.json`.

**Catalog effect (part of the same ruling).** `lab/catalog.py` ranks a non-GGUF recipe above a GGUF one. That makes this recipe the
`apple-m1-8gb` pick and moves the CPU-container recipe (`lfm2.5-2.6b.llama.cpp.8k`) to `more`. That changes the card's pick engine
from llama.cpp to mlx-lm. `docs/data.md` asks for a `lab/compare.py` head-to-head before a pick changes engine. That section is about
ExLlamaV3 to SGLang-EXL3 and requires both arms to use the same weights (repo@revision), which a GGUF and an MLX build cannot. It was
not run.

**Before merge, all three:** #361 merges first; a maintainer approves the MLX format and the pick change; the site build passes in CI
(it was not run here: this checkout has no `node_modules`).

## What the launch holds
- **A 32,768-token window, prompt and answer together.** `ctx` is 32,768, and `--max-kv-size 32768` bounds each attention layer's KV
  cache to it. Pinned mlx-lm 0.31.3 ignores `max_kv_size` for a model with its own `make_cache`, as LFM2 has (a growing `KVCache` per
  attention layer), so the launcher patch builds a `RotatingKVCache` of 32,768 tokens (`keep=4`) in its place.
- **Prompts up to 24,576 tokens.** `--max-prompt-tokens 24576` answers a longer prompt with HTTP 400 before any prefill or generation.
  That leaves at least 8,192 tokens of answer that still see the whole prompt. An answer that runs past the window drops the oldest
  tokens (after the first 4), so from then on it no longer sees the start of the prompt. Nothing past the window is claimed (measured
  below).
- **One cached conversation.** `--prompt-cache-size 1` (stock mlx-lm) keeps at most one finished conversation's cache between requests,
  so memory holds at most that one plus the request being served. The stock default keeps 10. Stock `--prompt-cache-bytes` takes effect
  only on mlx-lm's batch path, which this launch never uses.
- **Uncapped.** `--max-tokens -1`; no request limit, stop or budget; every answer runs to EOS.
- **macOS 15 or later.** The mlx and mlx-metal wheels are tagged `macosx_15_0`. The launch's `runtime.macos` is 15, so the SDK's steps
  begin with "Check the macOS version", which stops before anything is downloaded on an older macOS or where `sw_vers` cannot be read.

## Measured (owner's Mac mini, 2026-10-25)
Macmini9,1, Apple M1, 8 GB, 8-core GPU, macOS 26.4.1, on AC. The registry lab ran `lab.py try --on endpoint` from a task directory built
from a clean `git archive` of the interim commit `91d36083` (see "Reproduction" for exactly which files match this commit). All six gates
passed. The recipe and run JSON are the lab's own output, byte for byte. The run is `q3`; `q3b` is a second server start in the same task
directory for the streamed long answers and the stock-server sampler check.

| Measurement | Result |
| --- | --- |
| Gates | load, chat, reasoning, tools, context, speed: all true |
| Speed gate (recipe `tps`, first 30 s) | **36.1, an estimate**: characters/4, because mlx_lm.server has no tokenize route. The proof says so with `tps_estimated: true` and its `note` |
| The speed answer, whole (134.1 s, ended by itself) | 4,407 text-carrying stream events = 32.9/s; re-encoded with the pinned tokenizer 4,486 = 33.5/s; characters/4 4,362 = 32.5/s. **All three are estimates** |
| Context gate (stock) | 22,242 prompt tokens (`usage.prompt_tokens`, at least 0.6 x 32,768), code recalled, 108.2 s, prefill 206 tok/s |
| Over the prompt limit | 25,011 and 30,331 tokens: **HTTP 400** in 0.0 s, "the prompt is N tokens; this server accepts at most 24576". No prefill or cache work was logged for either |
| Ten distinct near-limit prompts, back to back | 24,071 prompt tokens each, no shared prefix, each with its own code: **10 of 10 recalled**, natural stop, 116.9 to 119.4 s each |
| Retained prompt cache | the server's own log before each of the 24 accepted requests: **never more than 1 sequence**, at most 0.40 GB |
| A short answer after a near-limit prompt (q3b) | 24,086 + 721 tokens (re-encoded), inside the window: natural stop, code recalled, 143.1 s |
| A long story after a near-limit prompt (q3b, streamed) | 24,097 + 6,226 tokens (re-encoded) = 30,323, inside the window: natural stop, code recalled, 362.5 s |
| A long story after a near-limit prompt (q3) | 24,096-token prompt, then an answer that **had not ended after 40.5 min** (02:49:41 to 03:30:12); the run was stopped there and no answer exists. GPU memory in use rose for about 6 minutes (median 2.69 to 2.79 GB), then held at 2.80 to 2.81 GB to the end, which is what a full, rotating window looks like; that it passed the window is inferred from that, not measured. Available system memory drifted from 2.13 to 1.68 GB over the 40 minutes; swap did not move |
| Memory, whole run (q3) | lowest available about 0.88 GB; GPU in use at most 3.97 GB; swap 0.17 GB from before the run, unchanged throughout; the 10% memory guard's lowest reading 21% (778 polls) |
| Memory, the ten near-limit prompts | the first: 1.17 GB lowest available, 3.55 GB GPU; the other nine: 0.91 to 0.92 GB and 3.94 GB each. Flat, not growing |
| Thermal | `pmset -g therm` reports no thermal, performance or CPU-power warning before setup, before the run and after it |

**Why the speeds are estimates.** The lab counts tokens through the server's tokenize route. mlx_lm.server has none (all three routes
answered 404, in the capture), so the lab fell back to characters/4 and wrote `tps_estimated: true` into the proof, with the `note`.
The catalog carries both; the site marks such a speed with "≈" on the recipe page, its description and images, the configuration cards and
the hardware directory, and never names an estimated recipe "Fastest" over measured ones (`site/lib/speed.ts`). [`tools/count_stream_tokens.py`](tools/count_stream_tokens.py)
gives two more readings of the same answer: the stream's own text-carrying events (about one per generation step on mlx_lm's
sequential path), and a re-encoding of the decoded text with the snapshot's `tokenizer.json` (sha256 695be780…). Re-encoding can split
boundaries differently and does not see tokens the server strips (think delimiters, EOS), so it is an estimate too. Every reading is above
the 15 tokens/second gate.

**Memory.** The prompt cache holds one conversation: each near-limit request ran with the previous one's 0.40 GB cache kept and its own
being built, and that is the plateau the ten show. Prompts over 24,576 tokens no longer reach the model, so the earlier past-the-bound
case (a 30,322-token prompt that touched swap and lost recall at the previous head) is now a 400.

## Every generation ended on its own, except one we stopped (raw capture)
Byte-transparent relays ([`tools/observer.py`](tools/observer.py)) kept every connection's raw bytes, committed gzipped with `RAW.sha256` of
the uncompressed bytes. The readers read the `.gz` files directly; an empty capture fails ([`tools/test_capture_tools.py`](tools/test_capture_tools.py)).
- **Lab** ([`…-32k.observer/`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.observer/), 16 connections): `check_run.py` passes on
  the 6 chat requests (`tool_calls` for the tool request, `stop` for every other; the stream ends in one `data: [DONE]`). `audit_capture.py`
  passes: no body carries an output or stop limit, a budget or a sampler key.
- **Probes** ([`…-32k.observer-probe/`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.observer-probe/), 21 connections):
  `check_run.py` passes 17 of the 20 chat requests and fails exactly the three that were meant not to finish: `conn-0008` and `conn-0009`
  (the two HTTP 400s) and `conn-0020` (the long answer, stopped after 40.5 min, with no response). `audit_capture.py --allow-sampler seed`
  passes (seed only for the sampler comparison) and lists `conn-0020` as unanswered.
- **Long answers, q3b** ([`…-32k.observer-long/`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.observer-long/), 3 connections):
  `check_run.py` passes both streamed answers (`stop`, `data: [DONE]`); `audit_capture.py` passes.

## Sampler: what was observed, and why
On a short prompt distinct from the lab's ("Write one sentence about a red kite flying over a lake."), through this launch
([`…-32k.probe.log`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.probe.log)): temperature 0 twice gave identical answers;
temperature 1.0 differed from temperature 0; and at temperature 1.0, two unseeded requests, two with seed 7 and one with seed 8 all gave
the same answer (sha256 `f54c1e2d…`). **Seeds and repeats did not change sampled output.** The cause is upstream, not this patch:
- [`tools/seedprobe.py`](tools/seedprobe.py) ([`…-32k.seedprobe.json`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.seedprobe.json))
  draws from mlx-lm's own `make_sampler` on the task runtime (mlx 0.31.2). On the main thread, seeds 7 and 8 differ and unseeded draws
  differ. **In a worker thread, where mlx_lm.server samples, every draw is the same token whatever the seed**: mlx-lm's categorical
  sampler is `mx.compile`d with `mx.random.state`, and in that thread it uses a frozen key. Plain `mx.random.categorical` in the same
  thread honours seeds.
- A stock, unpatched mlx_lm.server 0.31.3 on the same runtime ([`tools/pristine_sampler.py`](tools/pristine_sampler.py),
  [`…-32k.pristine-sampler.json`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.pristine-sampler.json); each request sends
  `max_tokens` 512, stock's own default): two unseeded temperature-1.0 requests on its batch path gave identical answers, and seeds 7 and
  8 on its sequential path gave identical answers too.

What this patch does change: with a bound set, every request is served on the sequential path, so an unseeded request that stock would
serve on the batch path gets the sequential path's sampled output instead. **The patch does not change sampler code, and no parity with
stock sampled output is claimed.** On this runtime, sampling at temperature above 0 is repeatable and seed-insensitive on either path.

## Runtime: the SDK's own setup steps
The SDK renders this launch as seven steps (Python and JS agree). The run executed the first six, then the seventh's exact argv under
supervision.
- **Generated files.** `sdk-setup.sh` (sha256 7ab804ad…2e54) is an exact prefix of the SDK's one-script `command()` (4c8aee2c…2339). It
  ran under `bash -e` from the task directory, supervised by `groupwatch run --max-seconds 5400`, and exited 0.
  [`verify.txt`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.verify.txt) is the committed runner's record of what it made.
  1. **Set up first.** The launch's note: macOS 15 or later, the 24,576-token prompt limit and the 32,768-token window.
  2. **Check the macOS version.** `sw_vers -productVersion`; it stops below 15, before any download (it passed on 26.4.1).
  3. **Install the runtime.** CPython 3.12.15 (python-build-standalone 20261021, sha256 a18cc338…4584), hash-checked, every member and
     link confined, staged, then published to `rt312/python`.
  4. **Install the packages.** 34 hash-locked wheels, installed offline into a fresh target and published to `rt312/site`.
     Lock: [`runtime/requirements.lock.txt`](runtime/requirements.lock.txt) (ac028b35…c241) and [`runtime/wheels.lock.tsv`](runtime/wheels.lock.tsv).
     Installed: mlx-lm 0.31.3, mlx 0.31.2, mlx-metal 0.31.2, transformers 5.12.1, huggingface_hub 1.22.0.
  5. **Download the weights.** `LiquidAI/LFM2.5-2.6B-MLX-4bit@04efa23776ce61ec34ec95ec34c859854c89542b`, downloaded on the pinned
     interpreter with `token=False` into a fresh staging cache, then published to `rt312/hf-cache` with `refs/main`. All 10 files, and
     only those, match [`runtime/weights-04efa237.sha256`](runtime/weights-04efa237.sha256).
  6. **Write the launcher.** `rt312/mlx_lm_unlimited_serve.py`, sha256 94300d51…3943
     ([`registry/launches/mlx-lm-unlimited-max-tokens.py`](../../../registry/launches/mlx-lm-unlimited-max-tokens.py)). `--verify-only`
     confirms the imported mlx_lm is the pristine PyPI 0.31.3 (package digest 8089a1a4…), then applies the in-memory edits
     ([`runtime/mlx_lm-0.31.3-unlimited-max-tokens.patch`](runtime/mlx_lm-0.31.3-unlimited-max-tokens.patch), 7 hunks), giving patched
     server.py sha256 9158b98e…7f17.
- **Start the server.** The server ran the exact argv, recorded in `sdk-start.argv` (627bf6f8…2fc9), under `groupwatch run --max-seconds
  8000 --min-free-pct 10` ([`server-argv.txt`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.server-argv.txt)). Launch sha256
  fd3e3548…03ec, pin `fcfc3a6d18b2`:
  `HF_HUB_OFFLINE=1 HF_HUB_CACHE=rt312/hf-cache PYTHONPATH=rt312/site rt312/python/bin/python3.12 -s rt312/mlx_lm_unlimited_serve.py --model LiquidAI/LFM2.5-2.6B-MLX-4bit --max-tokens -1 --max-kv-size 32768 --max-prompt-tokens 24576 --prompt-cache-size 1 --host 127.0.0.1 --port 8080`
- **What the patch changes.**
  - Validation accepts `max_tokens` -1 (unlimited, mlx-lm's own generator convention) and rejects -1 with a draft model.
  - A -1 request, and with `--max-kv-size` or `--max-prompt-tokens` every request, is served on the sequential path (`stream_generate`).
  - `--max-kv-size` bounds the KV cache as above. `--max-prompt-tokens` rejects a longer tokenized prompt in the sequential path before its
    generation context exists, and the handler answers that one error with 400 (stock answers every error raised there with 404, and still
    does for all others).
  - EOS handling, the prompt and the sampler code are untouched; the routing change above is what moves unseeded requests off the batch path.
  - Concurrent requests are served one at a time.
- **CPU-only regression** on the target's runtime: [`runtime/regression-cpu.txt`](runtime/regression-cpu.txt) is that run's stdout and
  stderr, merged (`2>&1`): 36 PASS lines, `REGRESSION PASS`, `exit 0`, no mlx module loaded. Its first lines are the pristine copy's
  argparse usage and error, printed when the pristine parser rejects `--max-kv-size` and the launch's flags, as the test expects. It runs the
  copies' own `_serve_single` (a prompt one token over the limit is rejected before the generation context and before any generation;
  one at the limit is served), `handle_completion` (400 for that rejection, 404 for any other error), the argparse block on the launch's own
  flags, and mlx-lm's own `LRUPromptCache` at `--prompt-cache-size 1` (keeps 1 of 3 distinct near-bound conversations; the stock default
  keeps 3). On the previous launcher (`902875cc`) its new checks fail and it exits non-zero.

## The host
The Mac mini also runs a camera service and a LiveKit server. Neither was paused, restarted or reconfigured. Before the run the host was
quiet: GPU 5%, ports 8080-8083 free, no other model process; swap held 0.17 GB from an earlier run. In all 139 snapshots of q3 (135 at
30 s intervals during it) and all 21 of q3b, both services kept the same pids and LiveKit answered HTTP 200. The snapshots name the host's
services and account, so they are kept privately. After each teardown every `groupwatch` reported `clean=True`, and no task process or
listener remained.

## Memory log
[`…-32k.memwatch-native.csv`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.memwatch-native.csv) (q3) and
[`…-32k.long.memwatch-native.csv`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.long.memwatch-native.csv) (q3b), from
[`tools/memwatch_native.py`](tools/memwatch_native.py), one row a second. `mem_available_gb_approx` is (free + inactive + purgeable +
speculative pages) x page size, an approximation. `swap_used_gb` comes from `sysctl vm.swapusage`. `gpu_in_use_gb` is the AGXAccelerator's
"In use system memory". [`…-32k.memory-windows.txt`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.memory-windows.txt) cuts the rows
at each request's start in the server's own log ([`…-32k.server.log`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.server.log)),
whose "Prompt Cache: N sequences" lines are the retained-cache record. The supervisor logs
([`groupwatch-server.log`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.groupwatch-server.log)) hold the guard's polls.

## Reproduction (clean checkout)
Build a task directory from a checkout:
- `tools/` and `runtime/`: copies of this directory's `tools/` and `runtime/`;
- `labtree/`: the checkout itself;
- `sdk-setup.sh` and `sdk-start.argv`: generated from the SDK. `sdk.command(recipe)` up to "# Start the server" is `sdk-setup.sh`, and the
  launch's `env` assignments followed by `command`, one per line, are `sdk-start.argv`.

Then run `bash tools/macmini-run3.sh setup|start|lab|probe|stop TAG`, and for the streamed long answers and the stock-server sampler check
`start|long|stop|sampler TAG`. The host services it observes come from `OBSERVE_SERVICES` and `OBSERVE_HTTP`.

This run's task directory was built that way from `git archive` of the interim commit `91d36083` and checked against its manifest
(10,494 files) on the target. Against this commit, file by file:
- **byte-identical (10,470 files)**, among them everything that ran or rendered the run: `lab/lab.py`, `sdk/js/index.js`,
  `sdk/python/local_ai_registry.py`, the launch and launcher in `registry/launches/`, the patch, the regression harness
  (`runtime/test_unlimited_max_tokens.py`, `run_regression_cpu.py`), and the tools `observer.py`, `memwatch_native.py`,
  `probe_m1.py`, `seedprobe.py`, `pristine_sampler.py` and `check_run.py`;
- **generated for the run (4)**: `sdk-command.sh`, `sdk-setup.sh`, `sdk-start.argv`, `SOURCE_COMMIT`;
- **replaced during the run**: `tools/macmini-run3.sh` (it gained the `long` phase) and `tools/long_answer.py` (new) were copied in after
  the q3 probe, before q3b; their hashes then are this commit's
  ([`…-32k.tools-replaced.txt`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.tools-replaced.txt));
- **changed after the run**: `runtime/regression-cpu.txt` (now this run's output); `tools/audit_capture.py` and
  `tools/count_stream_tokens.py` (they now report an unanswered or cut-off connection instead of crashing; the committed analysis files were
  made with these versions); `tools/test_capture_tools.py` and `tools/test_runner.py` (a test for that, and files closed); `dist/catalog.json`
  (this recipe added); this README; and two `__pycache__` files the SDK generation step left in the task directory (never tracked);
- **changed after review**: `tools/groupwatch.py`, the run's supervisor, now fails closed. `run` exits 4 when a teardown is not clean (a
  failing command keeps its own code), and 5 when supervision fails after the command started (it still tears down what it owns, and kills its
  own direct child if even that cannot run). Its logging is best effort, so a full disk can no longer stop a teardown or its fallback, and an
  unwritable log turns an exit of 0 into 5. The signal handlers are installed before the command starts. One supervisor holds a state
  directory at a time (an exclusive lock; a second is refused before it starts anything), and each run's token heads its log lines and its
  `supervisor.id`, so `stop` exits 0 only on that run's own clean final line and never on another run's. A process table counts only when
  `ps` exits 0, every line parses and the supervisor itself is listed, so a failed query can never read as "all gone"; its own child's exit
  comes from its Popen handle; and an armed memory guard that gets no reading three polls in a row tears the run down
  ([`tools/test_groupwatch.py`](tools/test_groupwatch.py)). The run's provenance is unchanged: all 17 supervisors of q3 and q3b ended with an
  `EXIT` or `ABORT` line saying `clean=True`, none hit a supervision or logging error, none shared a state directory with another (one
  `gw-*` directory per phase and run), and every memory poll of theirs had both readings, so the fixed version gives every one of them the
  exit code the run recorded. Their logs, and
  `tools/groupwatch-options-test.txt`, predate the token prefix on log lines. `lab/test_host_topology.py` is #361's corrected test at `ff00ce61` (a unit test the
  run never used; this launch's pin, `fcfc3a6d`, is the same). `sdk/js/index.d.ts` gained the types `Proof.note` and `Proof.tps_estimated` (declarations only; nothing rendered changes).

The kit's Qwen-specific reference panel and its generic sweep were not run, and no kit result is claimed.

## Limitations
- The format and pick ruling is the maintainer's, and the site build has not run (CI runs it).
- Only macOS 26.4.1 was measured; the macOS 15 floor comes from the wheel tags.
- Speeds are estimates; mlx_lm.server reports no token usage on a stream.
- Sampling above temperature 0 is repeatable and ignores seeds on this runtime, in stock mlx-lm as here.
- An answer that runs past the 32,768-token window loses the start of its prompt. One near-limit long story ended inside the window; another
  had not ended after 40.5 min. Uncapped answers can run that long.
- Concurrent requests queue: the bounded configuration serves one at a time.

## Files
- Recipe: [`registry/recipes/apple/apple-m1-8gb/lfm2.5-2.6b.mlx-lm.32k.json`](../../../registry/recipes/apple/apple-m1-8gb/lfm2.5-2.6b.mlx-lm.32k.json)
- Run JSON: [`../../lab-runs/apple-m1-8gb.lfm2.5-2.6b.mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1.32k.20261025T062632.json`](../../lab-runs/apple-m1-8gb.lfm2.5-2.6b.mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1.32k.20261025T062632.json)
- Launch: [`registry/launches/mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1.json`](../../../registry/launches/mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1.json)
- Probes and long answers: [`…-32k.probe.log`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.probe.log) (the q3 probe; it was
  stopped in its last request, so it wrote no `probe.json`, and its other results are this log and the probe capture),
  [`…-32k.long-answer.json`](../../lab-runs/20261025-apple-m1-8gb-lfm25-mlx-32k.long-answer.json) (q3b).
- The previous runs (24K at `902875cc`, 32K unbounded at `1cdf672b`) and their evidence stay in those commits.
