# GTX 1650 SUPER: LFM2.5-2.6B AutoRound INT4 (GPTQ-packed) on vLLM (no GGUF)

The unchanged current-main registry lab (`lab/lab.py` at c0604cff48078bfc13678c42b501bdced0c0fb81, SHA256 b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca) ran against the owner's NVIDIA GeForce GTX 1650 SUPER (4 GB, Turing TU116, compute capability 7.5) on 2026-10-23 UTC with `--on endpoint`. All six gates passed: load, chat, reasoning, tools, context and speed. The raw run JSON and the stock-generated recipe are copied byte for byte.

The recipe uses the registry's own `vllm` template profile (vLLM 0.30.0, the upstream image pinned by digest). Its settings are `ctx=8192`, `mem=0.78`, `tools=lfm2` and `think=deepseek_r1`; `seqs` stays at its default of 1, and there are no extra flags. The weights are an AutoRound INT4 W4A16 quant of the registered `lfm2.5-2.6b`, packed in GPTQ format (`auto_round:auto_gptq`). They are not GGUF and not AWQ.

The table is the first run (22:42–22:49 UTC, steps run by hand). The [end-to-end run](#end-to-end-run-with-reprosh) below repeated it with the committed `repro.sh` and produced the recipe's newest proof.

| Measurement | Actual result |
| --- | --- |
| Context configuration | 8,192 tokens, one sequence, `--kv-cache-dtype auto` (FP16 on this card) |
| Context recall | 5,552 input tokens; correct code; 104.3 seconds |
| Streamed decode (lab speed gate, first 30 s) | 20.3 tokens/second |
| Complete uncapped streamed answer | 1,501 tokens in 74.4 seconds, `finish_reason` `stop` |
| Lab-reported prefill | 53 tokens/second |
| Independent repeat of the stock speed prompt | 20.49 tokens/second; 3,084 tokens in 153.9 seconds, `finish_reason` `stop` |
| Peak GPU memory during the run | 3,126 MiB of 4,096 MiB (167 MiB in use before launch) |
| Host MemAvailable low point / swap | 24.60 GB / 1.91 GB, unchanged from the baseline |

A 4 GB card is under the registry's 8 GB KV-floor threshold, so the 131,072-token floor does not apply. No longer context is claimed. vLLM reported 50,353 tokens of KV capacity at this memory setting, but only 8,192 was run and gated.

## Why this runs on Turing (vLLM 0.30.0 = ced6857a)

- GPU support starts at compute capability 7.5 ([gpu.cuda.inc.md#L9](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/docs/getting_started/installation/gpu.cuda.inc.md#L9)), and the image is built for 7.5 ([Dockerfile#L293](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/docker/Dockerfile#L293)). vLLM's quantization table marks GPTQ as supported on Turing ([quantization/README.md#L70](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/docs/features/quantization/README.md#L70)).
- Turing has no BF16, so vLLM uses FP16 ([platforms/cuda.py#L255-L262](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/platforms/cuda.py#L255-L262); log: "Casting torch.bfloat16 to torch.float16"). LFM2 is not on vLLM's FP16-refused list ([config/model.py#L2252](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/model.py#L2252)).
- The checkpoint's `quant_method` `auto-round` is mapped to vLLM's `inc` method ([inc/inc.py#L361-L365](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/quantization/inc/inc.py#L361-L365)), which runs the `auto_round:auto_gptq` packing as GPTQ on Marlin ([inc_wna16_linear.py#L44-L64](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py#L44-L64)). Marlin rejects only compute capability below 7.5 ([marlin_utils.py#L65](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L65)). Log: "Using MarlinLinearKernel for AutoGPTQLinearMethod".
- FlashInfer is excluded below SM 8.0 because it is broken on SM 7.5 ([flashinfer.py#L510-L516](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backends/flashinfer.py#L510-L516)), and FlashAttention needs SM 8.0, so vLLM chose `TRITON_ATTN` from its list ([platforms/cuda.py#L174-L177](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/platforms/cuda.py#L174-L177)). The CUDA graphs were captured normally; there is no `--enforce-eager`.
- The `lfm2` tool parser ([tool_parsers/__init__.py#L113](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/tool_parsers/__init__.py#L113)), the `deepseek_r1` reasoning parser ([reasoning/__init__.py#L23](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/reasoning/__init__.py#L23)) and the model class ([models/lfm2.py#L395](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/lfm2.py#L395)) are all in this release.

## Requests and stop reasons

The lab's own requests set the sampling. It sends temperature 0.6 for chat, reasoning, tools and context, and temperature 0.8 with streaming for the speed prompt. Nothing else is sent: no `max_tokens`, no thinking budget, no stop strings. Fields the request leaves out come from the model's `generation_config.json` (top_k 50, repetition_penalty 1.1), which vLLM applies by default. Those values match the base `LiquidAI/LFM2.5-2.6B` file exactly.

vLLM's access log shows only HTTP status codes. To capture the actual end of each answer, the lab talked to a transparent TCP relay in front of the server. The relay copied bytes unchanged in both directions and kept a copy of each connection. [observer-summary.json](../../lab-runs/20261023-gtx-1650-super-4gb-vllm.observer-summary.json) lists, for every request, its parameters (messages omitted), the body's SHA256, whether it carried `max_tokens` (none did) and the `finish_reason` values in the response:

- chat `stop`, reasoning `stop`, tool call `tool_calls` (the model chose to call the tool), tool reply `stop`, context `stop`;
- speed-gate stream `stop` (conn-0007); repeat stream `stop` (conn-0013).

## Machine

- WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) on a Windows host, NVIDIA driver 595.95, PCIe gen 3 x16, AMD Ryzen 5 3600, 29.43 GB RAM ([kit probe](../../lab-runs/20261023-gtx-1650-super-4gb-vllm.probe.json), kit `tools/probe.py` at ef9782ce).
- Rootless Podman 6.1.4. WSL has no NVIDIA Container Toolkit here, so the GPU reached the container through WSL's paravirtual device: `--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib`. On a Linux host with the toolkit, `--gpus all` is the equivalent; the image, argv and weights are the recipe's.
- vLLM logged 3.2 GiB of the 4.0 GiB free at startup ("Free memory on device (3.2/4.0 GiB) on startup. Desired GPU memory utilization is (0.78, 3.12 GiB)"), so this recipe sets `mem=0.78`, below the free amount. The weights took 1.72 GiB; peak activation was 0.51 GiB; KV cache got 0.78 GiB.
- Memory was recorded once a second by a separately labelled native watcher, not the kit's `memwatch.sh`: [memwatch-native.csv](../../lab-runs/20261023-gtx-1650-super-4gb-vllm.memwatch-native.csv).
- No clock, power or fan change.
- First run: CPU load was not measured during its gated period (22:42 to 22:45:53 UTC). That run saved only the load average at server start (`5.16, 3.89, 4.54` at 22:39:18) and per-second memory and GPU data. Other containers from a separate Podman store, not part of this run, were seen on the host at server start. Whether they used the CPU during those gates is unknown. The end-to-end run below closes this gap with per-second CPU attribution.

## End-to-end run with repro.sh

On 2026-10-23 from 23:35:42 to 23:43:56 UTC, the committed `repro.sh` (commit f2ec0d8f, SHA256 b3d4d127…ad30) ran once on the same card. It used `RUNTIME` set to a task-local Podman wrapper, with storage, run state and temp on the data disk, and the WSL `GPU_ARGS` shown above. It passed: `{"lab_six_gates": 1, "speed_repeat": 1, "seven_natural_ends": 1, "cleanup": 1}`, exit 0.

- **Inputs, unchanged.** The lab hash, image digest, weight files and rendered launch SHA256 7a3aea9c…89b2 are listed in [input-hashes.txt](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.input-hashes.txt).
- **Six gates.** All passed: 20.3 tok/s decode, 53 tok/s prefill, 5,552-token recall in 104.7 s. The speed answer was 1,501 tokens in 74.6 s and ended with `stop`. The new [run JSON](../../lab-runs/gtx-1650-super-4gb.lfm2.5-2.6b.vllm.8k.20261023T234208.json) has raw-file SHA256 698d816f85803c3215c06d96d6ccb6dd0ff57a49fbaeb2f3cc84fa502122ac37 and log hash sha256:b34fb7e1b374b385. The lab wrote this proof as the recipe's first entry and kept the earlier proof (sha256:8a920506e448a3f3) after it, as it does for an unchanged launch.
- **Repeat.** The stock speed prompt, rerun, gave 20.44 tok/s: 2,115 tokens in 104.8 s ([speed-repeat](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.speed-repeat.json)).
- **Seven natural ends.** [check.jsonl](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.check.jsonl) shows stop, stop, tool_calls, stop, stop, stop (stream) and stop (stream), and no limits were sent. The relay's raw copies (17 connections, 34 files) are published in [relay.tar.gz](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.relay.tar.gz), with an [archive checksum](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.relay.tar.gz.sha256) and a SHA256 for each extracted file in [relay-sha256.txt](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.relay-sha256.txt). Each raw file and the compressed archive are under 1 MB.
- **Cleanup.** The server log was saved (26,403 bytes, [server.log](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.server.log)), and only then was container 359014243bcb… (label `io.local-ai-registry.repro=gtx1650-vllm.UbBQI2QE`) removed and confirmed gone. Afterwards the GPU read 167 MiB / 0 %, ports 18100 and 18101 were closed, and no helper process was left.
- **The same 1,501 tokens.** Both runs' speed-gate stories are 1,501 tokens long. vLLM's engine config shows `seed=0` (its default; the launch does not set it), and the lab's requests set only temperature. So the same prompt reproduces the same sampled answer.

**Host CPU and resource sampling.** A separate task logger ([baseline](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.baseline-hostwatch.csv), [run](../../lab-runs/20261023T2335-gtx-1650-super-4gb-vllm-e2e.run-hostwatch.csv)) sampled host CPU busy from `/proc/stat` interval deltas, load average, available memory, swap and GPU memory/utilization about once per second.

Before the run, 72 samples spanning 75 seconds between timestamps averaged 0.8% host CPU (max 3.9%), load1 at most 0.55, and GPU 167 MiB / 0%.

| Phase (UTC) | Samples | Host CPU sample mean / max | load1 max | GPU utilization sample mean |
| --- | --- | --- | --- | --- |
| Startup, before 23:38:24 | 153 | 7.7 / 34.9% | 1.25 | 8% |
| Six gates, 23:38:24–23:42:08 | 216 | 9.2 / 13.4% | 1.27 | 91% |
| Repeat, after 23:42:08 to 23:43:53 | 101 | 9.4 / 11.7% | 1.18 | 91% |

Host CPU stayed at or below 13.4% during the gates and repeat. GPU memory peaked at 3,126 MiB. Available host memory stayed above 24.70 GB, and swap remained at 1.905 GB. The raw `own_cpu_pct` column uses a nominal one-second denominator rather than the actual elapsed interval; it is retained as logged and is excluded from quantitative attribution to this run versus other processes. The table uses the interval-normalized host CPU field.

## Exact inputs and evidence

- Image: vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 (the registry `vllm` profile's image; vLLM tag v0.30.0 = ced6857a).
- Weights: plavno/LFM2.5-2.6B-AutoRound-W4A16@2244dd328189f0b58d9b938b576937ad7d216b7f. AutoRound 0.14.2, 4-bit symmetric, group 128, `auto_round:auto_gptq` packing; `lm_head` and embeddings unquantized; licence lfm1.0 (the base's). Shard SHA256 values match the Hub LFS ids: `model-00001-of-00002.safetensors` 5792b854fb26a4b30e9dc55da1f1465db99b671ea425374511ba577fe9901d3a, `model-00002-of-00002.safetensors` 3694b8fee957f0168ddda96c5c7bed60d2d56f52eec11064b5920282baf97d5a. `chat_template.jinja` is byte-identical to LiquidAI/LFM2.5-2.6B@b212e75975bbda9676a94d3cb7b2ca25b65fed48 (SHA256 ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef). The quant's own card says 4-bit costs this 2.6B model real accuracy, unevenly across languages.
- Rendered launch SHA256 (lab `launch_hash`): 7a3aea9cbfeffc16321f292ac05632a6b8fc3bb8c78b77a166da2cd10df589b2. The container was started from the lab's own `try --dry-run` rendering.
- First run: [raw lab run JSON](../../lab-runs/gtx-1650-super-4gb.lfm2.5-2.6b.vllm.8k.20261023T224553.json); raw-file SHA256 971a4e113a1e2c7b902de8f358b9c9d15de4dcd9bbab08d4770350d30bce2a63. Recipe log hash sha256:8a920506e448a3f3.
- [Full server log](../../lab-runs/gtx-1650-super-4gb.20261023T224553.vllm.server.log) and the [speed repeat](../../lab-runs/20261023-gtx-1650-super-4gb-vllm.speed-repeat.json). The repeat called `lab.stream_rate`, imported from the unchanged `lab/lab.py`, with the speed prompt read from that file, through the same relay, after the gated run.
- Scope of acceptance: this is a registry-lab acceptance only (`lab.py try --on endpoint`, the six gates, then `make check`). The kit's speed sweep (`tools/sweep.py`, the same protocol on every card) was **not run**. The kit's one published reference quality panel is for Qwen3.8-Flash-Next only, so no quality-panel result exists or is claimed for this model.

## Reproduction

[`repro.sh`](repro.sh) runs the whole check under bounded steps and fails unless all of the following hold:

- the lab's six gates pass (`lab.py` exits 0);
- one repeat of the stock speed prompt reaches at least 15 tok/s, using the lab's own `stream_rate` and the prompt read from the unchanged `lab/lab.py`;
- exactly seven chat requests were sent, each ending on its own with one `finish_reason`: `tool_calls` for the tool gate's first request, `stop` for every other. None of them sent `max_tokens`, `max_completion_tokens`, `min_tokens`, `stop`, `stop_token_ids`, `ignore_eos` or any `*budget*` field;
- cleanup succeeded: the server log was saved (the command exited 0 and the file is non-empty), and the container was then removed and confirmed gone.

~~~bash
# once, bounded, outside the run's lifetime (about 22 GB):
timeout --kill-after=60 3600 docker pull vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
# then, from a scratch copy of this registry checkout (the lab writes lab/runs/ and the recipe into it):
MODEL_DIR=/path/to/LFM2.5-2.6B-AutoRound-W4A16-2244dd32 LAB_DIR="$PWD" \
  bash data/sources/gtx1650super-vllm-autoround/repro.sh
# WSL with rootless Podman and no NVIDIA Container Toolkit, as measured here:
#   RUNTIME=podman GPU_ARGS="--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib" ...
~~~

How the script bounds and owns its work:

- **Bounded.** The script re-executes itself under `timeout --kill-after=120 $LIFETIME` (default 5,400 s). The health wait stops at `HEALTH_SECS` (900); every `curl` has `-m 5`; the lab, the repeat and every runtime call have their own `timeout -k`. Long steps run as background children, so a TERM reaches the cleanup trap at once.
- **Cleanup budget.** The worst case after TERM is 92 s: up to 12 s for a foreground probe to return, 2 × 6 s to stop helpers, then 12 s for the label check, 22 s for the logs, 22 s for `rm` and 12 s for the existence check. That fits inside the 120 s before the outer KILL, so cleanup always reaches its verdict.
- **Helpers fenced.** The observer, the lab and the repeat each run under `setsid` as the leader of their own process group. Each is recorded by pid and start time (from `/proc/<pid>/stat`). A group is signalled only while both still match: TERM first, KILL to the group after 5 s, then the leader is reaped. A recycled pid is never signalled or waited on.
- **Owned.** Each run gets a fresh `mktemp -d` directory (mode 700). The directory's name is also the container name and the value of the label `io.local-ai-registry.repro`. Before logging or removing anything, the script checks that the cidfile holds a 64-hex id and that the container carries its own label. If either check fails, it touches nothing. It never runs `rm -f` on a shared path.
- **Logs first, removal always.** Cleanup saves the server log first. It then always attempts to remove the container, even when saving the log failed, and confirms removal only when the runtime itself reports the container does not exist; a hang or any other error counts as not removed. Cleanup succeeds only when the log was saved and removal was confirmed. `verdict.json`, `lab-try.log`, `speed-repeat.json`, `check.jsonl` and the observer's raw copies stay in the run directory.
- **Pins.** Before anything starts, the script checks the two weight shards, `chat_template.jinja` and `lab/lab.py` against their SHA256, and checks that both ports are free and the image is present.

[`observer.py`](observer.py) and [`check_run.py`](check_run.py) are task tools that sit outside the lab and do not change it. The observer is the same file (SHA256 451c18f58236d0dd47bb1bae14a7627d56d35eafb72aa04a419db87eb1e51623) that relayed the measured run. It copies bytes unchanged in both directions between the lab and the server and keeps a copy of each connection. The checker reads those copies afterwards, without changing anything that was relayed. It parses each response as real HTTP (`http.client.HTTPResponse`, which removes chunked framing) and then checks:

- the request body is valid JSON of the declared length;
- the response is JSON, or SSE for a streamed request;
- no field named `error` appears anywhere;
- each message or event has exactly one choice;
- `tool_calls` comes with a named function call whose arguments are valid JSON;
- a `stop` answer has content and no tool call;
- a stream ends with exactly one `[DONE]`.

On the measured run's raw capture it reports all seven requests passing (conn-0002 to conn-0007 and conn-0013).

What was tested: `repro.sh` ran end to end on the GPU once and passed; see [End-to-end run with repro.sh](#end-to-end-run-with-reprosh). The first run's proof came from the same steps run by hand (the task's `abbey-run.sh`).

It was also exercised on a CPU harness with a fake runtime and a fake OpenAI server, which passed 29 checks:

- a clean pass;
- a `length` ending, rejected;
- the health timeout;
- the `LIFETIME` kill, with the container still removed;
- a foreign label, where nothing was logged or removed;
- wrong weights, refused before start;
- an empty log and a failed log command, where cleanup fails but the container is still removed;
- a hanging log command under the lifetime limit, with the container removed;
- the worst case, where the log command, `rm` and the existence check all hang. Cleanup reported "not confirmed removed" and finished about 50 s after TERM, before the outer KILL;
- the pid fence: a mismatched start time is never signalled, and a matching group is stopped and reaped.

The checker passed 17 separate cases:

- accepted: a stream whose SSE JSON is split across HTTP chunks, and a real tool call;
- rejected: a 200 response carrying an `error` payload with a `finish_reason`, missing or doubled choices, `tool_calls` with no call, bad tool arguments, `length`, an empty answer, a stream without `[DONE]`, an error event, `max_tokens`, a thinking budget in `chat_template_kwargs`, and HTTP 500;
- the real seven-request capture passes, and fails once one error response is added.

`docker rm -f` removes the container and the vLLM processes inside it. The run assumes about 3.2 GiB of the card is free at startup. With a display or another CUDA process on the card, there may be less than `mem=0.78` needs; this was not tested.
