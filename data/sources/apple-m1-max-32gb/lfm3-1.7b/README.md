# LFM3-1.7B on an Apple M1 Max 32 GB — owner-machine evidence

Evidence for `registry/recipes/apple/apple-m1-max-32gb/lfm3-1.7b.mlx-vlm.256k.json` (engine
`mlx-vlm-lfm3-1.7b-8bit-256k@122c66fdb191`, weights
`LiquidAI/LFM3-1.7B-MLX-8bit@4d6d94c9c2c2f249f1f740f4f08e071b0dbfe099`). One session on one machine, 2026-11-11:
speed sweeps of Liquid AI's three MLX exports and a fidelity measurement (18:37–19:54 UTC), which chose the weights
and the settings; then the `lab/lab.py try --on endpoint` run that wrote the recipe (20:12–20:20 UTC) and one extra
needle request near the full window (20:20–20:32 UTC), with the server's memory logged every 10 s.

## Machine

MacBook Pro 16-inch 2021 (`MacBookPro18,2`), Apple M1 Max, 10-core CPU (8P + 2E), **32-core GPU**, 32 GB unified
memory, 400 GB/s. macOS 26.5 (25F5058e), Python 3.12.13, mlx-vlm 0.7.12, mlx 0.32.4, transformers 5.22.2. AC power.
`iogpu.wired_limit_mb` left at 0. No other model server was running. Details in `host.json`, the whole environment in
`pip-freeze.txt`.

This is the owner's working laptop and it was in use throughout: a browser, editors and other programs were open,
14.7 to 14.9 GB of swap was in use, and the 1-minute load average was 3 to 7 during the lab run (each measured wave
of the sweeps records its own). The card's other recipe (LFM2.5-24B-A2B) was measured on an M1 Max with the 24-core
GPU; the card record does not tell the two apart.

## Why mlx-vlm, and the two settings a long prompt needs

No mlx-lm release loads LFM3 yet (ml-explore/mlx-lm#2016), so this uses mlx-vlm's server, which loads LFM3 as a text
model. Two things had to be set before a prompt near the 262,144-token window got through:

- **`--prefill-step-size 128`.** With mlx-vlm's default of 2,048 tokens per prefill step, macOS stopped the GPU work
  of long prompts (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`): a 222K-token prompt at 124,928 tokens,
  the same prompt at 140,288 tokens with a 512-token step, and one 32K-token prompt in four at the default
  (`prefill-step-failures.txt`). With 128-token steps the 222K prompt, the lab's 184,722-token prompt (twice) and the
  255,365-token probe all got through. The smaller step costs no measurable speed: back to back, twice each, cold
  8K prefill was 1,244–1,281 tok/s at 128 and 1,211–1,403 at 2,048; at 32K, 1,045–1,171 and 986–1,112
  (`sweep/prefill-step-*.json`).
- **`MLX_VLM_TOKEN_QUEUE_TIMEOUT=7200`.** mlx-vlm's server gives up on a request whose next token takes more than
  600 s. Reading 255,379 tokens takes about 650 s here, and the first probe failed at 243,968 tokens for that reason
  alone (`token-timeout-failure.txt`). The registry's other mlx-vlm launches set the same variable.

## Results

| | lab run (recipe proof) | full-window probe |
|---|---|---|
| harness | `lab/lab.py try` gates | `probe_full.py` |
| gates | load chat reasoning tools context speed — 6/6 | needle — pass |
| context prompt tokens | 184,722 (server-reported; 70.5% of 262,144) | 255,365 (server-reported; 97.4% of 262,144) |
| prefill | 508 tok/s (363.4 s) | 346 tok/s over the whole request (737.6 s); 394 tok/s as the server times the prompt |
| decode, 1 user | 101.2 tok/s (3,037 tokens in the first 30 s, counted by the served tokenizer; the answer ran to 6,254) | 35 tok/s with 255K tokens of context (server-reported) |
| peak GPU memory (MLX, per request) | 5.38 GiB | 6.73 GiB |
| peak process footprint (`memwatch.log`) | 7.07 GB | 8.72 GB |

The lab's context gate sizes its prompt from a token estimate, which for this tokenizer came to 70.5% of the window
(the gate requires 60%). The probe repeats the needle with a prompt sized by the served tokenizer to 255,365 tokens,
so the window is shown to fit in memory and to be recalled from, not only declared. The planted code came back in
both.

## Which weights: the sweeps

Liquid AI publishes three MLX exports. Each was served as the launch serves it (128-token prefill step), except that
up to four requests could run at once, so the rows at 2 and 4 show what batching gives. Every request had a context
of its own that no request had sent before, and no output limit; answers were 300 to 500 tokens, so a decode figure
is over 3 to 7 s (the lab's figure above is over 30 s). Medians of three measured waves at one stream and of two at
two and four, after one warm-up wave each.

**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX 8-bit · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,326 tok/s | 102.1 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,217 tok/s | 147.7 tok/s (73.9/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,126 tok/s | 172.7 tok/s (43.2/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,103 tok/s | 85.7 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 986 tok/s | 118.0 tok/s (59.0/stream) (1 of 2 waves: streams ran one at a time) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 102.3 tok/s. Quality: KL to bf16 0.0037 nats (95% CI 0.0029–0.0045), top-token agreement 96.8%. Peak GPU memory in these rows 2.93 GiB.

**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX bf16 (unquantized) · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,397 tok/s | 62.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,250 tok/s | 41.5 tok/s (20.8/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,151 tok/s | 85.6 tok/s (21.4/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,184 tok/s | 56.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,076 tok/s | not run (streams ran one at a time: each answer ended before the other prompt was read) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 67.8 tok/s. Quality: the unquantized weights (the reference). Peak GPU memory in these rows 4.32 GiB.

**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX 4-bit · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,498 tok/s | 130.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,325 tok/s | 188.1 tok/s (94.0/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,200 tok/s | 207.1 tok/s (51.8/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,217 tok/s | 112.8 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,039 tok/s | 133.4 tok/s (66.7/stream) (1 of 2 waves: streams ran one at a time) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 127.4 tok/s. Quality: KL to bf16 0.2571 nats (95% CI 0.2138–0.3044), top-token agreement 75.1%. Peak GPU memory in these rows 2.12 GiB.

Fidelity is the mean token-wise KL divergence from the bf16 weights' next-token distribution, over the full
vocabulary, at 25,600 positions of one sealed panel (25 windows of 2,048 tokens: English prose from Pride and
Prejudice, this registry's documentation, Python and JavaScript; sha256 `bf5e0d11e1084863c98504822dc1530c58b4fb5f543c450f3e8cfafdf58b3237`; the panel holds
source code that is not public and is not included). 8-bit: 0.0037 nats (95% CI 0.0029–0.0045),
median 0.0022, 95th percentile 0.0119, the same first token at 96.8% of positions.
4-bit: 0.2571 nats (0.2138–0.3044), median 0.1550, 95th percentile 0.8594, 75.1%.
The control (the candidate's positions shifted out of step) gives 9.8 nats.

So the recipe uses the 8-bit weights: 1.6 times bf16's decode speed at the same prefill speed, and next to no
distance from it. The 4-bit weights are faster again and measurably further from the model.

Two things the tables show that the recipe does not use. Batching helps the quantized weights (8-bit: 102, 148 and
173 tok/s in total at one, two and four streams) but not bf16, where two streams together decode slower than one.
The recipe serves one request at a time: only one stream at the full window was tried. And at 32K with two streams,
most answers ended before the other prompt had been read, so there was little or no time in which both were decoding.

## The launcher

`registry/launches/mlx-vlm-lfm3-serve.py` (sha256 `291fa7d3743b07ee46a89345640afb3eb81f5a40caf4255890e731fb0074e935`, equal to
`launch_config_sha256` in the run record) is `mlx_vlm.server` plus three things the lab needs: a `/v1/token/encode`
route (the speed gate counts decoded tokens with the served tokenizer), a `/v1/models` that lists only the served
model by its relative path, and the model's `generation_config` sampling (temperature 0.1, top-k 50, repetition
penalty 1.1) for requests that leave it out. The launch passes `--max-tokens 262144` so no answer is capped, and
`--enable-thinking`, which the model's template otherwise leaves to each request.

## Files

| file | role | sha256 |
|---|---|---|
| `../../../../lab/runs/apple-m1-max-32gb.lfm3-1.7b.mlx-vlm-lfm3-1.7b-8bit-256k.256k.20261111T202000.json` | the run record | `2fdcf7b2e7bd569a9ccb097818172ad8545a0efbb8a4a750b63b7fc07a4fe4ac` |
| `lab-try.log` | the lab's console output | `64679dcad955614237f84fe57be33d556ce782ae9a703cf998e9dc5b87114e92` |
| `run-server.sh` | the measured launch, paths in `$HOME` form | `42bcb65e0917699a7fb03d4b239e97a9396bcd3594f2e2e118f49ab8f3542c98` |
| `run-server.log` | server log for the whole session (lab run, then the probe) | `dc7ed034d3b688beb0edbb0c6581c422ef02b6252e63ec0adea162980e878f3d` |
| `memwatch.log`, `memwatch.sh` | the server process's physical footprint every 10 s (it includes GPU memory), free memory, swap and load | `586bd44d67db71324700c29e21d61e56b1fe37df3a0d0c41f7f21bdfb569ff1f` |
| `probe_full.py`, `probe-full-255365.json` | the full-window probe and its result |  |
| `probe-start.timestamp` | when the lab run ended and the probe began |  |
| `probe-4bit-222379-step128.json` | the first prompt that got through after the step was lowered (4-bit weights) |  |
| `prefill-step-failures.txt` | the last prefill line and the error of the three prompts macOS stopped |  |
| `token-timeout-failure.txt` | the last prefill line and the error of the first full-window probe |  |
| `sweep/sweep.py`, `sweep/tables.py`, `sweep/tables.md` | the sweep harness, and the tables below as it prints them |  |
| `sweep/sweep-<weights>-c1.json`, `-cn.json` | every wave of the three tables (one stream; two and four streams) |  |
| `sweep/prefill-step-*.json` | the prefill step compared: 128 and 2,048 back to back, twice; and the earlier run in which a 32K prompt failed at 2,048 |  |
| `kld/kld.py`, `kld/kld.json` | the fidelity measurement and its result |  |
| `host.json`, `pip-freeze.txt` | machine facts (identifiers omitted) and the environment | `1ea8d287e6f4af44376971e4e113ed38c840f43d063859ecf3e88af17348c7a5` |

Logs and scripts that name the running account's home directory have it replaced with `$HOME`.
