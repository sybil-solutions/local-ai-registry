# RTX 4080 Laptop GPU TabbyAPI EXL3 container evidence

**Registry-only recipe.** As of 2026-10-25 this recipe and its card are in `registry/` with this evidence. They are not in `dist/catalog.json` and not exported to the plugin (`plugin/v2` or `v3`). Qwen3.5-9B was released 2026-02-26 and is now 241 days old, past `registry/models.json` `max_age_days` (240), so `lab/catalog.py` no longer picks any Qwen3.5-9B recipe. The generated files in this branch are exactly `origin/main` rebuilt on 2026-10-25. [generated-delta-20261025.json](generated-delta-20261025.json) lists every catalog and plugin entry that rebuild adds or removes, shows that this branch adds none of its own, and checks that v2 and v3 agree. [generated_delta.py](generated_delta.py) produces it.

The unchanged registry lab (`lab/lab.py` at c0604cff48078bfc13678c42b501bdced0c0fb81, SHA256 b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca) ran with `--on endpoint` against the owner's NVIDIA GeForce RTX 4080 Laptop GPU (12 GB, sm_89) on 2026-10-24 UTC. It used the existing frozen launch `tabbyapi-qwen3.5-9b-exl3-4bpw-128k`, with no changes. All six gates passed: load, chat, reasoning, tools, context and speed. The raw run JSON and the stock-written 128K recipe are copied byte for byte.

| Measurement | Actual result |
| --- | --- |
| Context configuration | 131,072 tokens (`max_seq_len` and `cache_size`), Q4 KV cache, one card, MTP draft |
| Context recall | 106,242 input tokens; correct code; 65.6 seconds |
| Streamed decode (lab speed gate) | 91.9 tokens/second |
| Complete uncapped streamed answer (lab speed gate) | 2,062 tokens in 22.6 seconds, natural `stop` |
| Independent repeat of the stock speed request | 107.7 tokens/second; 4,668 tokens in 41.5 seconds, natural `stop` |
| Lab-reported prefill | 1,620 tokens/second |
| Peak GPU memory during the run | 7,862 MiB of 12,282 MiB (52 MiB in use before launch) |
| Host MemAvailable low point / swap | 21.69 GB / 4.71 GB (4.66 GB at baseline) |

The card has more than 8 GB, so the registry's KV floor applies. The launch holds 131,072 tokens of KV, which meets the 131,072-token floor without an exemption.

## Every response ended naturally

The lab records `finish_reason` only for the chat answer and the tool follow-up. The lab was therefore pointed at a pass-through recorder on 127.0.0.1:15001, which forwarded each request unchanged to the server on 127.0.0.1:15000 and logged every response's finish reason. This recorder is a task tool, not part of the lab or the kit; it is published here as [finish_recorder.py](finish_recorder.py), and the checks as [check_postconditions.py](check_postconditions.py). Its log is [finish-recorder.jsonl](../../lab-runs/20261024-rtx-4080-laptop-12gb.finish-recorder.jsonl), and the checked postconditions are in [postconditions.json](../../lab-runs/20261024-rtx-4080-laptop-12gb.postconditions.json).

Every raw request body, every raw JSON response and the full SSE text of both streams are in [raw-recorder.tar.gz](../../lab-runs/20261024-rtx-4080-laptop-12gb.raw-recorder.tar.gz): 44 body files, `recorder.jsonl` and a `SHA256SUMS` list. Archive SHA256 d1e6f9da5e2e7802c931876256837e53e590bdad500e6e8a34c5b397f44b0e0c ([.sha256](../../lab-runs/20261024-rtx-4080-laptop-12gb.raw-recorder.tar.gz.sha256)). Verify with `sha256sum -c 20261024-rtx-4080-laptop-12gb.raw-recorder.tar.gz.sha256`, then `tar xzf … && sha256sum -c SHA256SUMS`. Its `recorder.jsonl` is byte-identical to the log above.

- The run made exactly seven `POST /v1/chat/completions` requests: the six gate requests plus the repeat. Every one returned HTTP 200 with no error.
- None of the requests set `max_tokens` or `max_completion_tokens`.
- Finish reasons: chat `stop`; reasoning `stop`; tool call `tool_calls` (with the `get_weather` call); tool result `stop`; context `stop`; speed stream `stop`; repeat stream `stop`.
- The [server log](../../lab-runs/20261024-rtx-4080-laptop-12gb.server.log) shows `max_tokens: 131046 (auto)` on the streams. That is TabbyAPI's own default (the context left after the prompt), not a cap set by the lab or the runner.

## Machine

- Lenovo laptop, Intel Core i9-14900HX (32 threads), 33.35 GB RAM, Linux 7.2.5, NVIDIA driver 610.57.04, PCIe gen 4 x16 ([kit probe](../../lab-runs/20261023-rtx-4080-laptop-12gb.probe.json); host RAM in the probe was taken while another job was running).
- Rootless Podman 6.1.1 with task-local storage. This host has no NVIDIA Container Toolkit or CDI spec, so the GPU reached the container by hand: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` as devices, and the host driver's `libcuda`, `libnvidia-ml`, `libnvidia-ptxjitcompiler`, `libnvidia-nvvm` and `libnvidia-gpucomp` (610.57.04) bind-mounted read-only into `/usr/local/nvidia/lib64`, which is already on the image's `LD_LIBRARY_PATH`. On a host with the toolkit, `--gpus all` is the usual equivalent. That route was not measured here.
- The kit's `tools/memwatch.sh` needs `bc`, which this host lacks. Memory was therefore recorded once a second by a separately labelled native watcher with the same columns plus GPU utilisation, power and SM clock: [memwatch-native.csv](../../lab-runs/20261024-rtx-4080-laptop-12gb.memwatch-native.csv).
- No clock, power, fan or GPU-limit change; no output cap; no thinking budget; no sampler or seed changes.

## Exact inputs and evidence

- Image: ghcr.io/0xsero/tabbyapi-exl3@sha256:0f83e6198dc3be2561652d8df2525a7d1a69733e12fc8e25cbc5bc319a8a3ad1 (amd64 manifest sha256:9506a96c89fdd98b24db4cde9b5a6b390bf649af502ba234e17b41da0ed478bb, config sha256:938e1c1768292e15c03b651d29a1266be8396fb3c9db6f26a19fbdf789dba6fe).
- Weights: TheMelonGod/Qwen3.5-9B-exl3@e98453aa143a24fab9eb14b718daed02e2fa6eef (EXL3 4.0 bpw, 6-bit head). `model.safetensors` is 7,307,469,398 bytes with SHA256 c43536c08cb54147a097937868f0791f637f98d1505c79475aa9b9b05589a74b, and `tokenizer.json` has SHA256 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42. All 13 files matched the Hub's LFS SHA256 or git blob ids before launch.
- Rendered launch SHA256: 72198e58f86afcf47cd0b7aae9d2c23dfc5f2af25440b260e2bca6a816e07097. Config file SHA256: d3efe710e28fe94a93a81a86bf52f74f95ab6740800890d78b6a4f9db7ee9d0c.
- [Raw lab run JSON](../../lab-runs/rtx-4080-laptop-12gb.qwen3.5-9b.tabbyapi-qwen3.5-9b-exl3-4bpw-128k.128k.20261024T000705.json); raw-file SHA256 72b7f651c70102c661f47c390de5220e507d4a722029d24b289d912bdf62ac23. Recipe log hash sha256:efc6b260bc176808 (the lab hashes the text before its final newline).
- [Speed repeat](../../lab-runs/20261024-rtx-4080-laptop-12gb.speed-repeat.json): one run of the lab's own `stream_rate` with the stock prompt, uncapped.
- The kit's published target and reference panel are for Qwen3.8-Flash-Next. Neither that panel nor the full kit speed sweep was run. This submission records the registry's six lab gates.

## Reproduction on a Linux host without the NVIDIA Container Toolkit

[repro.sh](repro.sh) is a self-contained, bounded reproduction of the whole measured procedure. It needs Linux, an NVIDIA driver, rootless Podman, `flock`, `curl`, `pgrep` (procps) and Python 3.11 or newer, and it changes no GPU, driver, clock, power or OS setting.

It writes only under `STORE`: the Podman storage, a lock file and one `STORE/run-<id>/` directory per run. The checkout is read and never written. The stock lab rewrites the recipe with its new proof and writes `lab/runs/`, so `repro.sh` runs it in a scratch copy of `lab/` and `registry/` at `STORE/run-<id>/lab-root`. It keeps the rewritten recipe as `stock-recipe.json` and the run JSON in the run directory.

`MODEL_DIR` must hold exactly the 13 files of [weights.manifest](weights.manifest). The one other entry allowed is a real `.cache` directory, which `huggingface-cli download --local-dir` leaves behind. A `.cache` file or symlink, another dot directory, or any extra file fails the check.

~~~bash
MODEL_DIR=/path/to/Qwen3.5-9B-EXL3-6hb-4bpw STORE=/path/to/roomy/dir \
  bash data/sources/rtx4080-laptop-tabbyapi-exl3-container/repro.sh
~~~

What it does, in order:

1. Pins. It refuses to run unless `lab/lab.py` is the stock lab (SHA256 b7d36ea4…5cca), the recipe renders to the measured launch (72198e58…7097) and config file (d3efe710…9d0c), all 13 weight files match [weights.manifest](weights.manifest), the image has the pinned digest, and the GPU is an `NVIDIA GeForce RTX 4080 Laptop GPU`. It takes one non-blocking lock per `STORE`, so only one run can use a store at a time.
2. Idle check, before anything starts and again immediately before `podman run` (a pull can take long). It takes three samples one second apart. Each needs GPU utilization of at most 15 % and memory use of at most 1,500 MiB (parish-media's thresholds), with no compute process. A query that fails, takes longer than 10 s or returns anything else counts as busy, so a game or other graphics load stops the run even with no compute process. On a machine with a battery, a mains or USB supply must be online. Where `parish-media` is installed, its `status` must also exit 0 and report `ac_online: true` and `not_ready: null`. Nothing is changed to make the GPU idle.
3. Server. It starts the rendered launch with the device and driver-library mapping described above. The container gets a per-run name, cidfile and `sybil.run` label.
4. Lab. It runs the stock `lab.py try --on endpoint` through the recorder, then one uncapped repeat of the stock speed request using the lab's own `stream_rate`.
5. Postconditions. [check_postconditions.py](check_postconditions.py) requires exactly seven chat completions, each HTTP 200 with no error and no `max_tokens` / `max_completion_tokens`, with `stop` (or `tool_calls` with calls) on every JSON answer and a final `stop` on both streams. It also requires the repeat to reach at least 15 tok/s.
6. Cleanup, always. It acts only on a container whose id and run label it verifies. It saves the server log and requires it to be non-empty, then removes the container (escalating from `rm` to `kill` and `rm`) and requires podman to report the container gone. It stops only the helper process it started, after checking that helper's PID and start time.

It exits 0 only if the six gates, the repeat and every postcondition pass, the server log was saved and the container and helper are positively gone. The exit codes are listed at the top of the script. Every stage is bounded (`REPRO_TIMEOUT` 7200 s overall, `HEALTH_TIMEOUT` 900, `LAB_TIMEOUT` 5400, `REPEAT_TIMEOUT` 1800), and cleanup also runs after a timeout or signal. Each long step (render, weights check, image check, pull and inspect, container start, lab, repeat, postconditions) runs in the background under its own `timeout`, in its own process group, while the script waits for it. A signal interrupts that wait at once, and cleanup first stops the stage: TERM to its `timeout`, which passes the TERM on and KILLs the stage's group after its grace (at most 30 s). Cleanup then KILLs anything left in that group and requires the group to be empty. A stage that ends on its own (normally, or on its own deadline with exit 124) gets the same check before the script moves on: if anything of its group is left, such as a grandchild that ignored TERM, it is KILLed, and if it cannot be confirmed empty the run fails with exit 9 and keeps the stage for cleanup. Only a `pgrep` that finds no match counts as empty: a `pgrep` error or a missing `pgrep` fails closed the same way, and cleanup then reports `ok=0`. So the deadline reaches a slow pull or a hung lab directly, not only once that stage's own bound runs out. When the overall bound expires, the outer `timeout` sends TERM and then allows 540 s before KILL. That covers the worst complete cleanup path, 507 s with every step running to its own bound; the script header shows the sum step by step. Before the recorder stops, the script waits up to 10 s for its seventh chat row. The recorder's TERM handler then stops taking requests and lets in-flight requests finish writing their rows. A signal that arrives while cleanup is already running, as the deadline can, is recorded and does not stop cleanup. Cleanup still saves the log, removes the container, confirms it is gone and reports its verdict, and the run exits non-zero. That includes a signal that arrives while the verdict line itself is being written: the line's text was already expanded and can still say `exit 0`, but the handler has set the exit status and appended a `signal … during cleanup` record. The process exit status is what counts. Each run writes its evidence to `STORE/run-<id>/`: the lab log and run JSON, `recorder.jsonl` with raw bodies, `postconditions.json`, `server.log` and the cleanup readback.

[test_repro.py](test_repro.py) runs `repro.sh` on CPU against a stub Podman and `nvidia-smi` and a fake TabbyAPI, with the real stock lab. It checks that a correct run exits 0, and that a run fails when an answer has the right text but ends with `length` (reasoning, tool call, context, speed or repeat). The stock lab passes that last case on its own. It also checks failure on a log that fails or is empty, a container that cannot be removed, a container with another run's label (left untouched), a busy or different GPU, a missing driver library, mismatched weights, a held lock, and a server that never becomes healthy, hangs, or fails to start. A container `rm` that fails once must be escalated and still end clean. No `rm`, `kill`, `stop` or `logs` call may name a container that is not verifiably this run's. A run that passes must leave `lab/`, `registry/` and this directory byte-for-byte unchanged. In the slow-pull fixture, the deadline lands in a pull that ignores TERM and has a second process in its group. The stage must be stopped, cleanup must reach its verdict well inside the reserve, and no process of the stage may survive. Other cases cover a stream that closes half a second after `[DONE]`, a recorder stopped right after `[DONE]` (the row must still be written, whole), and the weights contract above. The idle cases cover a GPU at 99 % with no compute process, a GPU that becomes busy after the pull, malformed, two-line, failed and timed-out queries, running on battery, and parish-media reporting not ready or failing. None of them starts a container. In the worst-case fixture the overall bound expires during the lab while the log, removal, kill and absence calls all hang past TERM and the GPU readback hangs. Cleanup still reaches its verdict after 165 s of the 540 s reserve, and the run ends by TERM, not KILL. A copy with a 60 s reserve fails that fixture. Two more fixtures land the deadline inside the server-log save, one after a passing run and one after a failing run. In both, the log is saved (and stays free of controller output), the container is removed and confirmed gone, the helper is gone, and cleanup reaches its verdict. The earlier TERM handler, which exited in the middle of cleanup, fails both. Four more fixtures send TERM, INT, or TERM then INT from inside the final verdict stamp, and once after a failing run. The process must exit 143, 130, 143 or keep the failure's own 8. A handler that only records the signal fails all four.

The first measured run above came from the task runner that `repro.sh` was built from: the same recorder, checks, mapping and launch. `repro.sh` itself then ran once on the same card, as recorded below.

## Confirmation run of repro.sh on the card

On 2026-10-25 UTC the published command ran once, unchanged, from a clean export of the committed tree: `repro.sh` SHA256 1cdb92c43c6ffc0e262a85b4b6122688cb90caec0e634c8569ff254193939224, with the same weights, image, launch and lab pins as above. parish-media reported the GPU ready immediately before the run. `repro.sh`'s own preflight and pre-launch idle, AC and parish-media checks passed. It exited 0.

| Measurement | Result |
| --- | --- |
| Six lab gates | all pass |
| Lab speed gate | 108.4 tokens/second; 3,187 tokens in 29.5 s, natural `stop` |
| Uncapped repeat of the stock speed request | 108.1 tokens/second; 3,984 tokens in 34.3 s, natural `stop` |
| Context recall | 106,242 input tokens; correct code; 66.9 seconds |
| Lab-reported prefill | 1,588 tokens/second |
| Chat completions | exactly 7; HTTP 200; none set `max_tokens`, `max_completion_tokens`, `stop` or `seed`; four JSON answers end with `stop`, the tool call with `tool_calls` (`get_weather`, Paris), and both streams with `stop` and `[DONE]` |
| Peak GPU memory | 7,862 MiB of 12,282 MiB (52 MiB in the quiet baseline) |
| GPU utilization / power, quiet 60 s baseline vs run | 8–13 % / 8.4 W vs up to 100 % / 198.1 W |
| Host MemAvailable low point / swap | 20.24 GB (24.53–24.81 GB at baseline) / 4.14–4.19 GB |
| Host CPU (vmstat us+sy), baseline vs run | mean 4.2 % vs 7.6 %, max 16 % |
| Cleanup | server log saved (78 lines), container removed and confirmed gone, helper gone, verdict after 2 s of the 540 s reserve |

Evidence, all byte for byte from the run:
- [run JSON](../../lab-runs/rtx-4080-laptop-12gb.qwen3.5-9b.tabbyapi-qwen3.5-9b-exl3-4bpw-128k.128k.20261025T060529.json): SHA256 8f0c4b5c8637b96af107f4063c25fc4fc0833cb5fb04a781b8c2a2720ea9362d, recipe log hash sha256:c5889d9fc4b219b7.
- The stock lab added this proof ahead of the first one in the recipe. The first proof is unchanged.
- [repro.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.repro.log), [server.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.server.log), [postconditions.json](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.postconditions.json) and [speed-repeat.json](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.speed-repeat.json).
- [raw-recorder.tar.gz](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.raw-recorder.tar.gz): 44 body files (every request and response, including both full SSE streams), `recorder.jsonl` and `SHA256SUMS`. Archive SHA256 c2c8823c3198ab6856b5c75729dc9cc5b36e2623701fc86e119b4924fbeafa52 ([.sha256](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.raw-recorder.tar.gz.sha256)).
- The quiet baseline and run metrics came from task watchers, not from `repro.sh` or the kit: [memwatch-native.csv](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.memwatch-native.csv) (1 s samples) and [vmstat.txt](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.vmstat.txt), started 60 s before `repro.sh` by a wrapper whose log is [confirm.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-confirm.confirm.log).

The run above used `repro.sh` at 1cdb92c4. After review the controller was revised: stages are owned and stopped on the deadline, the lab runs in a scratch copy, the recorder shuts down gracefully, and no bytecode lands in the checkout. That revised controller is the one in this directory.

## Confirmation run of the revised repro.sh on the card

On 2026-10-25 at 07:10 UTC the published command ran once more, unchanged, from a clean export of commit f1a5fa7380ea288807f1ae80a08add2e3f655ff0 (`repro.sh` SHA256 22eaf76f8143fc64da0580698f0118085a72afbfa0077fbe1b6b7e9a44173641). The weights, image, launch and lab pins were the same. parish-media reported the GPU ready immediately before (no compute process, 52 MiB, AC on), and `repro.sh`'s own idle, AC and parish-media checks passed. It exited 0.

| Measurement | Result |
| --- | --- |
| Six lab gates | all pass |
| Lab speed gate | 97.0 tokens/second; 2,624 tokens in 27.2 s, natural `stop` |
| Uncapped repeat of the stock speed request | 75.0 tokens/second; 1,891 tokens in 25.4 s, natural `stop` |
| Context recall | 106,242 input tokens; correct code; 71.8 seconds |
| Lab-reported prefill | 1,480 tokens/second |
| Chat completions | exactly 7; HTTP 200; no `max_tokens`, `max_completion_tokens`, `stop` or `seed`; four JSON answers end with `stop`, the tool call with `tool_calls` (`get_weather`, Paris), both streams with `stop` and `[DONE]`; every recorder line parses |
| Peak GPU memory | 7,862 MiB of 12,282 MiB (52 MiB in the quiet baseline) |
| Host MemAvailable low point / swap | 17.69 GB (23.31–25.35 GB at baseline) / 4.36–4.87 GB |
| Host CPU (vmstat us+sy), baseline vs run | mean 18.1 % (max 96 %) vs 43.2 % (max 100 %) |
| Checkout | all 10,472 exported files byte-identical before and after; no `__pycache__` |
| Cleanup | server log saved (78 lines), container removed and confirmed gone, helper gone, verdict after 5 s of the 540 s reserve |

The host was busier on CPU during this run than during the earlier ones (mean 4 % and 8 %). Some other host load ran alongside it; the GPU itself was idle at the baseline. That probably explains the slower repeat. Both speed measurements stayed far above the 15 tokens/second floor.

The recipe keeps its two committed proofs. The stock lab would put this run's proof on top, but three proofs make the recipe 1,122 bytes, and `lab/lab.py check` rejects any recipe over 1,024 bytes. The lab's own output is kept as evidence instead.

Evidence, all byte for byte from the run:
- [run JSON](../../lab-runs/rtx-4080-laptop-12gb.qwen3.5-9b.tabbyapi-qwen3.5-9b-exl3-4bpw-128k.128k.20261025T071310.json): SHA256 37229727cce45126c1ece3641a3575c9693d86d61406a79107debf32b1863b4f, log hash sha256:b1d2cbd5729c5504.
- The [recipe the stock lab wrote](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.stock-recipe.json), in the scratch copy.
- [repro.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.repro.log), [server.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.server.log), [postconditions.json](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.postconditions.json), [speed-repeat.json](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.speed-repeat.json) and the [checkout check](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.checkout-untouched.txt).
- [raw-recorder.tar.gz](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.raw-recorder.tar.gz): 44 body files, `recorder.jsonl` and `SHA256SUMS`. Archive SHA256 e4c1b4388e9feb352b804e97b84822ee2332ff24c7f543394768d8a03c4edcc0 ([.sha256](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.raw-recorder.tar.gz.sha256)).
- Task-watcher metrics, not from `repro.sh` or the kit: [memwatch-native.csv](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.memwatch-native.csv), [vmstat.txt](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.vmstat.txt) and the wrapper's [confirm.log](../../lab-runs/20261025-rtx-4080-laptop-12gb.repro-f1a5fa73.confirm.log).

Limits: three runs on one laptop card (one through the task runner, two through `repro.sh`), all through the manual device mapping described above (no NVIDIA Container Toolkit). The kit panel and full kit sweep were not run. The recipe is in the registry only; see the top of this file.
