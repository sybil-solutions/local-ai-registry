# Inkling 2 Mini on one RTX PRO 6000 Blackwell: the full 1,048,576-token window, with the DSpark drafter on sm_120

Evidence for `registry/recipes/nvidia/rtx-pro-6000-blackwell-96gb/inkling-2-mini.vllm.1024k.json` (engine
`vllm-inkling-2-mini-nvfp4-dspark-1024k-rtx-pro-6000-blackwell-96gb@c2dc759f5984`, weights
`thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`). One session on the owner's workstation,
2026-11-22, 08:47 to 09:10 UTC: the launch was started exactly as `sdk/python` prints it (`docker run --rm --gpus all
...`; the only host-side changes were a container name and publishing port 8000 on 127.0.0.1:18000), then
`lab/lab.py try --on endpoint` ran its six gates against it and wrote the recipe. Afterwards, on the same server,
supplementary repeats of the speed gate and a code prompt.

## What this launch is

| | |
|---|---|
| weights | the official checkpoint at `ed40e7da`, 20.2 GB: NVFP4 experts (8 expert tensors MXFP8), BF16 attention, embeddings and norms; `mtp.safetensors` (BF16) is the DSpark drafter. It is the quantized release, not a BF16 one: `config.json` declares modelopt NVFP4, and the top-level `"dtype": "bfloat16"` that commit added is the dtype of the unquantized parts. `weights-check.txt`: all 14 files match the Hub at that commit |
| image | `vllm/vllm-openai:inkling-2-mini@sha256:c2dc759f…`, the upstream image the model card points to until a vLLM release has the looped architecture. Unmodified |
| context | `--max-model-len 1048576`, the model's own limit (`text_config.model_max_length`), bf16 KV cache (no KV quantization) |
| drafter | DSpark, 7 drafted tokens; vLLM's `num_speculative_tokens_per_batch_size` verifies all 7 up to 32 running requests and the first 3 above |
| batch | `--max-num-seqs 128`, CUDA graphs to 1,024 tokens (`max_cudagraph_capture_size`) |
| memory | `--gpu-memory-utilization 0.90` |

### The sm_120 fix (the launch's config file)

The image's own vLLM kernels are built for sm_100/sm_103 and older, not sm_120. Without a drafter that does not
matter, but the DSpark drafter writes its KV cache with `_C_cache_ops::reshape_and_cache_flash`, which has no sm_120
kernel in this image, so the drafter fails with "no kernel image is available". The launch's `config` is a
`sitecustomize.py`, mounted at `/opt/sm120-kv-write/sitecustomize.py` with `PYTHONPATH=/opt/sm120-kv-write`. When vLLM's
kernel library loads, it registers a torch-indexing version of that one op (the same values in the same cache
slots; bf16/auto cache only). Nothing else changes: the same image digest, no rebuilt layers. Drop it once an image
ships sm_120 kernels.

## Machine

`host-and-container.json`: AMD Ryzen 9 9950X3D (16 cores, 32 threads), 94 GiB RAM, one NVIDIA RTX PRO 6000 Blackwell
Workstation Edition 96 GB on PCIe 5.0 x16, stock 600 W power limit, driver 620.70.06 (CUDA 13.5), Fedora 44, kernel
7.3.3, rootless Docker 29.9.3. The card also drove the desktop: KDE Wayland, a browser and a mail client held about 3.2
GiB of it throughout. Nothing else ran on the GPU.

## Results

| | |
|---|---|
| start to ready | 310 s, nothing cached (torch.compile, FlashInfer autotune, CUDA graph capture every start) |
| lab gates | load chat reasoning tools context speed: **6/6** (`lab-try.log`, run record `lab/runs/rtx-pro-6000-blackwell-96gb.inkling-2-mini.vllm-inkling-2-mini-nvfp4-dspark-1024k-rtx-pro-6000-blackwell-96gb.1024k.20261122T085552.json`) |
| context gate | **741,790 prompt tokens** (the gate's 85% filler, counted by the served tokenizer), cold and unique, 166.1 s for the whole request; the planted code came back (`58213`) |
| lab decode, 1 stream | **156.1 tok/s** (1,355 tokens in the 8.7 s after the first token, the whole answer; the 600-word story at temperature 0.8) |
| lab prefill | **4,466 tok/s**, measured on that 741,790-token prompt (global attention grows with length, so this is the long-context figure) |
| GPU KV cache | 1,159,258 tokens: 1.11x a 1,048,576-token request. 128 sequences is the concurrency cap for shorter conversations |
| GPU memory | 90,199 MiB at the highest (of 97,887), desktop included (`memwatch.csv`, 1 s samples) |
| host memory | MemAvailable 60,384 MiB before the start, 50,587 MiB at its lowest: 9.6 GiB. No `needs`: the weights stay on the GPU |

### What the drafter does and does not buy (same session, same machine)

The speed gate is one prose story at temperature 0.8. On prose the drafter is about even with no drafter; on code it
roughly doubles one user's speed. So the lab's decode figure undersells it, and a single run of the gate is noisy
with it: a few stories run long and repetitive, the drafter's acceptance climbs, and so does the rate.

| one user | this launch (DSpark) | the same launch without the drafter (`comparison/no-drafter-launch.json`) |
|---|---|---|
| lab speed gate | 156.1 tok/s | 160.0 tok/s (`comparison/no-drafter-lab-run.json`, 6/6) |
| speed gate x8, median (range) | 181.6 (147.0 to 284.7) (`speed-gate-x8.jsonl`) | 159.1 (156.4 to 161.3) |
| code prompt x4, median | **297.4 tok/s** (`code-prompt-x4.jsonl`) | 159.7 tok/s |
| GPU KV cache | 1.11x a 1M request (0.90) | 1.36x (0.92) |

The code prompt: "Write a complete Python module implementing a thread-safe LRU cache with per-entry TTL expiry, type
hints and docstrings, followed by pytest unit tests." Measured with the lab's own `stream_rate` (first 30 s after the
first token, served tokenizer). Its answers ran 17,000 to 30,000 tokens.

### Why 0.90

The first passing run of this launch used `--gpu-memory-utilization 0.92` (`comparison/dspark-mem0.92-lab-run.json`,
6/6, 156.8 tok/s, 4,338 tok/s prefill). Its memory trace (`comparison/dspark-mem0.92-memwatch.csv`) shows a 5-second
spike to 96,151 MiB of 97,887 early in the 741,790-token prefill, with the desktop's 3.2 GiB on the card. The
no-drafter launch peaked at 92,220 MiB at 0.92. At 0.90 the drafter launch peaked at 90,199 MiB with no spike and still
fits a whole 1M request, so this launch uses 0.90.

The next three sections were measured on later starts of this same launch (and, for the drafter-off column, of the
drafter-off launch), 09:50 to 10:30 UTC. The head-to-head further down has its own starts. `scripts/` holds the probes.

## Recall far from the question

The context gate plants its code in the last ~20 tokens before the question. Inkling's local layers see the last
1,024 tokens anyway, so that gate passes without the global layers reaching back. `deep-recall/` tests the reach:
one document of 901,431 tokens (War and Peace, then Anna Karenina, Project Gutenberg plain text,
`filler-sha256.txt`), a unique first line so nothing comes from the prefix cache, three one-sentence office notes
planted at line breaks, then "quote each of the three notes exactly". Two documents, six depths:

| depth | 5% | 25% | 50% | 65% | 85% | 95% |
|---|---|---|---|---|---|---|
| tokens between the note and the question | ~856,000 | ~676,000 | ~451,000 | ~315,000 | ~135,000 | ~45,000 |
| quoted word for word | yes | yes | yes | yes | yes | yes |

Each request took ~250 s (prefill and the answer), finished on its own (`stop`), and quoted all three notes exactly.
In the second document the notes came back in a different order than they appear; the question did not ask for order.

## Throughput with many users

`load-users-1-32-64-128.jsonl`: N closed-loop clients send IFEval and MATH-500 prompts in a fixed order, reasoning
effort medium, no limit on answer length; vLLM's own generated-token counter over 60 s once all N are running. No
errors at any level. (An earlier run with a 2,000-token answer limit, `comparison/load-capped-2000-tokens.jsonl`, was
within 5% at every level.)

| users | total tok/s | per user | first token, p50 | KV in use, peak | power |
|---|---|---|---|---|---|
| 1 | 207 | 210 | 0.05 s | 0.6% | 372 W |
| 32 | 2,403 | 76 | 0.14 s | 15% | 544 W |
| 64 | 3,919 | 62 | 0.16 s | 26% | 564 W |
| 128 | **5,174** | 41 | 0.22 s | 51% | 574 W |

## Quality with and without the drafter

`quality/`: the same 545 items (IFEval 200, MATH-500 150, MMLU-Pro 195), the same request for each (seed 4242,
reasoning effort medium, 16,000-token answer limit), 64 at a time, on this launch and on
`comparison/no-drafter-launch.json`. Graded per item; the `items` files list each item's result.

| | drafter on | drafter off | on − off (95% CI) | items only one got right (on / off) |
|---|---|---|---|---|
| IFEval (strict) | 93.5 | 91.0 | +2.5 (−1.5 to +6.5) | 11 / 6 |
| MATH-500 | 98.0 | 98.7 | −0.7 (−2.0 to +0.6) | 0 / 1 |
| MMLU-Pro | 82.6 | 82.1 | +0.5 (−2.8 to +3.8) | 6 / 5 |

No difference beyond sampling noise (McNemar exact p = 0.46 over all 545). Answers that hit the 16,000-token limit:
4 with the drafter, 3 without. The same 545 items took 261 s with the drafter and 424 s without (696,258 and 648,260
tokens).

## Head to head with #482 on this card (same weights commit, same image)

`comparison/pr482-launch-128k/`: #482's own launch (its `vllm-inkling-2-mini` profile, `ctx=131072`) started from the
SDK's command on this machine. Its rendered launch hashes to `6d72c064…`, the `launch_sha256` in #482's proof, so it
is exactly that launch. `comparison/this-launch-at-128k/`: this launch with only `--max-model-len 131072` changed.
Both passed 6/6. The context gate sent both the same 91,890-token prompt. The load rows here used the 2,000-token
answer limit on both sides.

| | #482's launch | this launch at 128K | this launch (1M) |
|---|---|---|---|
| lab decode, 1 stream (one story) | 158.6 tok/s | 143.8 | 156.1 |
| lab prefill, 91,890-token prompt | 7,922 tok/s (11.6 s) | **10,325 tok/s (8.9 s)** | (4,466 at 741,790) |
| 1 user, the mixed prompts above | 120 total, 163 while running; first token p50 1.51 s | | 216 total, 228 while running; 0.05 s |
| 4 users | 147 total: one runs, three wait (first token p50 27 s) | | |
| 128 users | one at a time (`--max-num-seqs 1`) | 5,060 | **5,153** |
| context window | 131,072 | 131,072 | **1,048,576** |
| start to ready | **80 s** | 305 s | 306 s |

On the lab's one-story decode figure #482's launch is 2 to 10% ahead on this card (158.6 against 156.1 and 143.8; with
the drafter that number swings from run to run, 147 to 285 over 8 repeats). It also starts in about a quarter of the
time. Everything else measured here favors this launch.
