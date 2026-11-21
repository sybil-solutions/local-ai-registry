# Qwen3.8-27B on Apple M1 Max 32 GB (32-core GPU): llama.cpp Metal + MTP, 128K

Evidence for `registry/recipes/apple/apple-m1-max-32gb/qwen3.8-27b.llama.cpp.128k.json` (launch
`registry/launches/llama.cpp-metal-unsloth-qwen3.8-27b-ud-q4-k-xl-mtp3-128k-apple-m1-max-32gb.json`, weights
`unsloth/Qwen3.8-27B-GGUF@4af51b3542f775b0a8e4b310bfb3f17e01c306c9`, file `Qwen3.8-27B-UD-Q4_K_XL.gguf`, sha256
`2a5f3565…369f1211`). One owner session on 2026-11-21: a hill-climb with
[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)
@ `fda968e7`, then `lab/lab.py try --on endpoint` against the exact launch on the pinned prebuilt.

This run uses the **same machine, engine build, harness and benchmark settings** as the Inkling 2 Mini recipe on
this card (#466, `data/sources/inkling2-mini-m1-max-32gb-llamacpp/`), so the two compare directly.

## Machine

MacBook Pro 16-inch 2021 (`MacBookPro18,2`), Apple M1 Max, 10-core CPU, **32-core GPU**, 32 GB, macOS 26.6.2 (25G83),
AC power, `iogpu.wired_limit_mb` left at 0 (24.96 GiB Metal working set). 352.7 GB/s read and 9.53 TFLOPS fp16
measured with MLX (`tools/hwbench.py`, `evidence/host.json`). About 14 GiB was in use by other desktop apps; the
Inkling server was stopped before this run, and no other model server ran.

## Engine

Unsloth's llama.cpp release [`b12588-mix-c0c3de8`](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)
(manifest source commit `883286076`): the hill-climb on a build from its source tarball, the lab gates on its macOS
arm64 prebuilt (sha256 `229def18…fc496ceb8`). Mainline llama.cpp also runs this model; the Unsloth build was kept so
that only the model differs from the Inkling 2 Mini recipe. MTP uses the head built into the GGUF (`blk.64`,
`nextn_predict_layers = 1`); no separate drafter file.

## Results

| | lab run (recipe proof) | hill-climb best (`bench.py`) |
|---|---|---|
| harness | `lab/lab.py try` gates | skill `scripts/bench.py`, temp 1.0, thinking on |
| gates | **6/6** (load, chat, reasoning, tools, context, speed) | — |
| decode, 1 user | **17.6 tok/s** (529 tokens in the first 30.1 s, served tokenizer, MTP 3) | 14.84–15.03 tok/s median of 6 × 512 tok (MTP 3; no-MTP controls 10.82–11.92) |
| prefill | 74 tok/s over the whole 106,294-token context request (1,431.6 s) | 124.1 tok/s @ 2.8K; 116.6 @ 21.6K (no MTP); 118.7–122.3 @ 2.8K with MTP |
| peak system memory | 28.49 GiB (server RSS 22.54 GiB, swap 1.39–1.52 GiB, flat) | 29.46–29.95 GiB |

Ceilings (`THEORY.md`, the skill's `ceiling.py`): decode 21 tok/s at 4K without speculation (16.5 GB read per token;
the baseline reaches 57% of it), ~39 tok/s with 2.2 accepted tokens per step; prefill 87 tok/s at the skill's 50%-MFU
convention, which the measured 124 tok/s exceeds (71% of fp16 peak), so that ceiling is conservative here.

Gate details: chat `red, yellow, blue` (finish `stop`); reasoning returned separately, answer `391`;
`get_weather(city="Paris")` and the reply quotes the tool's random code `b81525a4`; context recalled `58213` from a
**106,294-token** prompt (81% of the window). The speed gate's answer to "Write a detailed 600-word story" ran to its
natural end at **15,182 tokens** (778.7 s, mostly thinking at the gate's temperature 0.8); MTP acceptance over that
request was 82% (10,797 / 13,164). The lab gate decodes faster than `bench.py` (17.6 vs ~15 tok/s) because its
story prompt drafts more predictably than the bench's mixed prompt panel (82% vs 39–42% acceptance).

## What fits in 32 GB

Metal buffers at `-ub 256`, q8_0 KV, 128K: weights 994.65 + 14,819.41 MiB, KV 4,352 MiB (only the 16 full-attention
layers of 64 keep KV), recurrent state 149.62 MiB, compute 638.54 MiB = 20,954 MiB of 25,558 MiB. f16 KV (8,704 MiB)
would total ~25,307 MiB, at the edge where the Inkling run already failed (25,342 MiB), so q8_0 KV is kept as the
same labelled trade.

## Hill-climb

Medians of 6, every candidate between controls (full log with every sample: `PLAN.md`; raw: `ledger.jsonl`, `logs/`):

| lever | result | verdict |
|---|---|---|
| MTP 1 / 2 / **3** drafts (round 1, controls 11.92 / 11.14) | 12.23 / 14.29 / **15.03** tok/s | **3 accepted** |
| MTP 4 / 5 / **3 repeat** (round 2, controls 10.84 / 10.82) | 14.38 / 13.54 / **14.84** | 3 stays best |
| `-ub 512` / `-ub 1024` (no MTP) | prefill 120.8 / 112.9 vs controls 121.7 / 119.7 | rejected |

Acceptance falls from 63% (1 draft) to 39–42% (3) and 28% (5). With 3 drafts a 4-token verify step costs ~1.6× a
1-token step, because this dense model reads its weights once for all drafted tokens; on the MoE Inkling 2 Mini a
2-token verify cost 1.98× and DSpark lost at every length. Controls drifted down ~9% over 70 minutes (no thermal
warning; swap grew 0.8 → 1.66 GiB with other apps open; likely memory pressure, not verified); interleaving keeps
each comparison within one drift state.

## Files

| file | role |
|---|---|
| `../../lab-runs/apple-m1-max-32gb.qwen3.8-27b.llama.cpp-metal-unsloth-qwen3.8-27b-ud-q4-k-xl-mtp3-128k-apple-m1-max-32gb.128k.20261121T144727.json` | the run record (moved unchanged from `lab/runs/`) |
| `evidence/lab-try.log` | the lab's console output |
| `evidence/run-server.log` | server log for the lab run |
| `evidence/run-server.sh` | the measured launch, paths in `$HOME` form |
| `evidence/memwatch.jsonl`, `evidence/memwatch.sh` | 1 Hz system memory, swap and server RSS during the lab run |
| `evidence/server-start.timestamp` | UTC time the lab-run server started (memwatch began just before) |
| `evidence/host.json` | machine facts (same machine as the Inkling 2 Mini evidence) |
| `PLAN.md`, `THEORY.md`, `ceiling-qwen38-27b-m1max.json` | hill-climb log, ceilings and their inputs |
| `ledger.jsonl`, `configs/`, `logs/` | every `bench.py` run, its config and output |
| `tools/` | `hwbench.py`, `gguf_meta.py`, `memprobe.sh` (copied from the Inkling evidence), the serve/stop and sweep scripts |

Logs and scripts that name the running account's home directory have it replaced with `$HOME`.
