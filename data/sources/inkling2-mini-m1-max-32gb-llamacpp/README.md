# Inkling 2 Mini on Apple M1 Max 32 GB (32-core GPU): llama.cpp Metal, 128K

Evidence for `registry/recipes/apple/apple-m1-max-32gb/inkling-2-mini.llama.cpp.128k.json` (launch
`registry/launches/llama.cpp-metal-unsloth-inkling-2-mini-ud-q4-k-xl-128k-apple-m1-max-32gb.json`, weights
`unsloth/Inkling-2-Mini-GGUF@ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`, file `Inkling-2-Mini-UD-Q4_K_XL.gguf`).
One owner session on 2026-11-21: a hill-climb with
[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)
@ `fda968e7`, then `lab/lab.py try --on endpoint` against the exact launch on the pinned prebuilt.

## Machine

MacBook Pro 16-inch 2021 (`MacBookPro18,2`), Apple M1 Max, 10-core CPU (8P + 2E), **32-core GPU**, 32 GB unified
memory, macOS 26.6.2 (25G83), AC power. `iogpu.wired_limit_mb` left at 0 (system default: 26.80 GB / 24.96 GiB
Metal working set). Measured with MLX: 352.7 GB/s read, 367.6 GB/s copy, 9.53 TFLOPS fp16 GEMM
(`tools/hwbench.py`). About 14 GiB was in use by other desktop apps the whole time; no other model server ran.
`evidence/host.json` has the details (identifiers omitted). This is a second machine for the existing
`apple-m1-max-32gb` card, whose first recipe came from a 24-core-GPU M1 Max.

## Engine

Unsloth's llama.cpp release [`b12588-mix-c0c3de8`](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)
(upstream b12588 plus merged PRs; manifest source commit `883286076`). Mainline ggml-org llama.cpp reads the
`inkling2_loop` layout since ggml-org/llama.cpp#32694, but gives wrong output for these quants until
ggml-org/llama.cpp#32704 merges (open on 2026-11-21; see the weights repo's `Mainline_Support/README.md`).

- The hill-climb ran on a build from the release's source tarball
  (`llama.cpp-source-b12588-mix-c0c3de8.tar.gz`, sha256 `92ee8439…573607`, Metal, Release, static).
- The lab gates ran on the release's macOS arm64 prebuilt named in the launch
  (`llama-b12588-mix-c0c3de8-bin-macos-arm64.tar.gz`, sha256 `229def18…fc496ceb8`; `llama-server --version`:
  `build 12588, commit 883286076`). Same commit as the source build; the two decode within 0.5 tok/s
  (20.5 tok/s in the hill-climb controls vs 20.1 in the speed gate, different harnesses).

## Results

| | lab run (recipe proof) | hill-climb best (`bench.py`) |
|---|---|---|
| harness | `lab/lab.py try` gates | skill `scripts/bench.py`, temp 1.0, thinking on |
| gates | load chat reasoning tools context speed — 6/6 | — |
| decode, 1 user | **20.1 tok/s** (603 tokens in the first 30 s, served tokenizer; 3,665-token answer) | 20.50 tok/s median of 6 × 512 tok (controls 20.39–20.77) |
| prefill | **75 tok/s**: 91,890 server-counted prompt tokens in 1,230 s (whole request); code `58213` recalled | 202.6 tok/s @ 3.3K; 146.5 @ 25.5K |
| peak system memory | 29.99 GiB (memwatch, incl. ~14 GiB other apps); server RSS 19.80 GiB; swap flat at 799 MiB | 28.1–29.7 GiB |

Ceilings (`THEORY.md`, the skill's `ceiling.py` with the GGUF's tensor sizes and the looping counted): decode
83 tok/s at 4K, 45 tok/s at 108K; prefill ~420 tok/s practical. Decode sits at 25% of the bandwidth ceiling: it is
bound by kernel efficiency over 48 looped layer passes, not memory bandwidth.

## What fits in 32 GB, and what did not

The Metal working set (25.0 GiB) is the binding constraint at 128K. Measured Metal buffers:

| launch | weights | KV | compute | total | result |
|---|---:|---:|---:|---:|---|
| default (-ub 512, f16 KV) | 16,332 MiB | 4,336 | 6,118 | 26,786 | OOM at warmup |
| -ub 256, f16 KV | 16,332 | 4,296 | 4,714 | 25,342 | OOM at warmup |
| **-ub 256, q8_0 KV (this recipe)** | 16,332 | 2,282 | 4,712 | 23,326 | **runs** |
| + DSpark drafter, -ub 256 (f16 draft KV) | 16,704 | 2,806 | 5,319 | 24,829 | OOM at first request |
| + DSpark drafter, -ub 128, q8_0 draft KV | 16,704 | 2,559 | 3,836 | 23,099 | runs, but slower (below) |

q8_0 KV is a labelled quality trade (as in #460); f16 KV at 128K would need a raised `iogpu.wired_limit_mb` (sudo),
which this run did not change. The compute buffer has a ~3.3 GiB fixed part (relative-bias attention over the full
window) plus ~5.5 MiB per ubatch token.

## DSpark speculative decoding loses on this machine

Interleaved with -ub 128 controls (20.77, 20.39 tok/s), medians of 6:

| draft tokens | c1 decode | acceptance |
|---:|---:|---:|
| 1 | 16.56 | 58.0% |
| 2 | 17.04 | 43.7% |
| 3 | 15.04 | 32.7% |

At one draft token, 1.58 tokens per step at 16.56 tok/s means a 2-token verify costs 1.98× a 1-token step: verify
is linear in tokens. Below 32 rows, Metal's `mul_mat_id` uses the per-token mat-vec kernel, and each verified token
picks its own 8 of 128 experts in all 48 looped passes, so nothing is shared between them. The launch has no drafter.

The same holds at the [Unsloth guide](https://unsloth.ai/docs/models/inkling-2-mini)'s own launch (32K context,
default -ub 512, f16 KV, DSpark Q8_0 with 3 drafts, text only): 15.37 tok/s with the drafter vs 20.95 and 20.87
without it, interleaved (`logs/guide32k-*.bench.log`).

## Other levers (all inside noise)

`-lm none` 20.60 tok/s (startup 7 s), `-lm mmap+mlock` 20.52 (9 s), vs controls 20.49 / 20.50 / 20.52 (2 s);
`-fa off` cannot run with a quantized V cache. Full log with every number and reason: `PLAN.md`; raw runs:
`ledger.jsonl`, `logs/`.

## Files

| file | role |
|---|---|
| `../../lab-runs/apple-m1-max-32gb.inkling-2-mini.llama.cpp-metal-unsloth-inkling-2-mini-ud-q4-k-xl-128k-apple-m1-max-32gb.128k.20261121T115012.json` (sha256 `285cd2f0…721d5d76`; proof log `sha256:47cfeb5223b864c6`) | the run record (moved unchanged from `lab/runs/`) |
| `evidence/lab-try.log` | the lab's console output |
| `evidence/run-server.log` | server log for the lab run |
| `evidence/run-server.sh` | the measured launch, paths in `$HOME` form |
| `evidence/memwatch.jsonl`, `evidence/memwatch.sh` | 1 Hz system memory, swap and server RSS during the lab run |
| `evidence/host.json` | machine facts |
| `evidence/server-start.timestamp` | UTC time the lab-run server started (memwatch began just before) |
| `PLAN.md`, `THEORY.md`, `ceiling-inkling2mini-m1max.json` | hill-climb log, ceilings and their inputs |
| `ledger.jsonl`, `configs/`, `logs/` | every `bench.py` run, its config and output; OOM server logs (debug lines removed). `dspark3-a` is the OOM run (null metrics); `faoff` and `nommap` failed at server start, so they have configs but no ledger rows (`PLAN.md` E3, E4) |
| `tools/` | `hwbench.py` (MLX bandwidth/GEMM), `gguf_meta.py` (header sizes), `memprobe.sh` (macOS memory probe for `bench.py`), the serve/stop and sweep scripts |

Logs and scripts that name the running account's home directory have it replaced with `$HOME`.
