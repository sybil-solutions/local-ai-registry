# Inkling-2-Mini on M1 Max 32 GB: hill-climb log

Skill: sybil-solutions/inference-optimisation-skill @ fda968e7 (read in full before use; ran `bench.py` and
`ceiling.py` only, with `python3 -I`, from a separate copy; `mem.sh` and `queue.sh` are Linux/Docker-only and were
replaced by `tools/memprobe.sh` and sequential runs).

## Targets and constraints

- Metrics, in order: c1 decode tok/s (thinking on, the model's default), prefill tok/s at 4K/32K/~111K, TTFT, startup.
- Context that must stay stable: 131,072 tokens (the registry's KV floor for cards over 8 GB), one sequence.
- Memory cap: the Metal recommended working set, 26.80 GB (24.96 GiB); no `iogpu.wired_limit_mb` change, no sudo.
- Quality: lossless only (same weights; speculative decoding is lossless in distribution).
- Fixed environment: AC power, no power-profile or clock changes, no other model server, one GPU job at a time.

## Inventory

- MacBook Pro 16-inch 2021 (MacBookPro18,2), M1 Max, 10-core CPU (8P+2E), 32-core GPU, 32 GB, macOS 26.6.2 (25G83).
- Engine: unslothai/llama.cpp prebuilt release b12588-mix-c0c3de8, built from its release source tarball
  (sha256 92ee8439f9ae51a40f15c168ac64e1d657bf8b08d301c96f8fb2d58625573607; manifest source commit 88328607,
  upstream b12588 + merged PRs incl. ggml-org#32218 DSpark, unslothai#348 Metal NVFP4 kernels, #385/#386 Inkling 2
  attention). Mainline ggml-org llama.cpp cannot run these quants correctly until ggml-org#32704 merges (open on
  2026-11-21). Metal, Release, static.
- Weights: unsloth/Inkling-2-Mini-GGUF@ae9327c5fbc7333116ba32eee78bb5d2bbefae4f, Inkling-2-Mini-UD-Q4_K_XL.gguf
  (17.63 GB, sha256 e62d463c…) and dspark-Inkling-2-Mini-Q8_0.gguf (0.40 GB drafter).

## Experiments

All runs: `harness/bench.py` (skill), temperature 1.0, thinking on (model default), c1 decode = median of 6 × 512-token
answers from the fixed prompt panel, tokens from server usage; prefill = prompt_tokens ÷ TTFT on a unique cold prompt.
Memory = system wired+active+compressed (`tools/memprobe.sh`), idle ≈ 14 GiB from other apps.

### E0 — default launch (-ub 512, f16 KV, 128K) — FAILED (OOM)
Metal buffers: weights 15,863 + 470 MiB, KV 4,096 + 240 MiB, compute 6,118 MiB = 26,786 MiB > 25,558 MiB working set.
`kIOGPUCommandBufferCallbackErrorOutOfMemory` at warmup. The compute buffer has a ~3.3 GiB fixed part plus ~5.5 MiB
per ubatch token (relative-bias attention over the full window).

### E0b — -ub 256, f16 KV — FAILED (OOM)
Compute 4,714 MiB; total 25,342 MiB, still OOM at warmup (other processes hold GPU memory). f16 KV at 128K needs
`iogpu.wired_limit_mb` raised (sudo); out of scope (no system setting changes).

### E1 — baseline: -ub 256, q8_0 KV (labelled quality trade, as in PR #460) — base-a
c1 **20.45** tok/s [20.46 20.49 20.44 20.45 20.45 20.43], prefill 194.7 (3,302 tok) / 146.5 (25,554 tok), peak 29.65 GiB.
25% of the 83 tok/s bandwidth ceiling → kernel/launch-bound, not bandwidth-bound.

### E2 — DSpark drafter (dspark-Inkling-2-Mini-Q8_0, draft KV q8_0) — REJECTED at every length
Needs -ub 128 to fit (draft adds weights 372 MiB + KV 277 MiB + compute 561 MiB; at -ub 256 → OOM). Controls -ub 128:
base128-a 20.77 [20.44 … 21.33], base128-b 20.39 [20.37 … 20.43].

| draft n | c1 median | samples | acceptance (whole run) |
|---|---:|---|---:|
| 1 | 16.56 | 18.32 16.51 15.59 16.61 18.55 15.57 | 0.580 (1132/1952) |
| 2 | 17.04 | 19.29 16.50 14.68 17.57 18.54 15.14 | 0.437 (1437/3289) |
| 3 | 15.04 | 20.21 15.34 14.50 14.74 18.00 13.44 | 0.327 (1524/4666) |

Why: at n=1, 1.58 tokens/step at 16.56 tok/s means a 2-token verify step costs 1.98× a 1-token step. Verify
cost is linear in tokens: below 32 rows Metal's `mul_mat_id` uses the per-token mat-vec kernel
(`ggml_metal_op_mul_mat_id_use_mm`: ne21 ≥ 32), and each verified token routes to its own 8 of 128 experts in all 48
looped layer passes, so nothing is shared. Matches the THEORY.md bandwidth model (union of experts grows with k).
Don't retry without a batched small-row MoE kernel for Metal.

-ub 128 vs 256 (no draft): decode same within noise (20.39–20.77 vs 20.45); prefill at 3.3K 184–186 vs 194.7 (−5%).
Keep -ub 256 when no draft is loaded.

### E3 — -fa off — NOT RUNNABLE
`quantized V cache requires flash_attn to be enabled`; f16 KV does not fit (E0b). Rejected by constraint.

### E4 — --no-mmap — renamed in this build; tested as `-lm none` (E5)

### E5 — `-lm none` (weights copied, not mmapped) — REJECTED (inside noise, slower start)
Controls (-ub 256, q8_0 KV): base256-b 20.49 / prefill 202.6, base256-c 20.50 / 202.6.
lmnone: c1 20.60, prefill 202.0, healthy after 7 s (controls 2 s, warm page cache).

### E6 — `-lm mmap+mlock` — REJECTED (inside noise, slower start)
c1 20.52, prefill 202.6, healthy after 9 s.

## STATUS — plateau (E2, E5, E6 inside noise or worse; E3 not runnable)

| metric | baseline (base-a) | best (= baseline launch) | ceiling | how |
|---|---:|---:|---:|---|
| c1 decode | 20.45 | 20.49–20.50 (base256-b/c) | 83 (4K) | bench.py median of 6 × 512 tok, temp 1.0, thinking on |
| prefill ~3.3K | 194.7 | 202.6 | 423 | cold unique prompt, prompt_tokens ÷ TTFT |
| prefill ~25.5K | 146.5 | — | 376 | same |
| KV capacity | 131,072 tok × 1 | same | — | q8_0, 2,176 MiB global + 106 MiB SWA |
| peak system memory | 29.65 GiB | 28.1–29.4 GiB | 32 GiB | wired+active+compressed incl. ~14 GiB other apps |

What limits further gains: decode runs at 25% of the bandwidth ceiling, so it is bound by kernel efficiency and launch
overhead (48 layer passes × MoE mat-vec + relative-bias attention per token), not memory bandwidth. Config levers
left are none that fit memory; next would be kernel work (batched few-row MoE kernel for Metal, which would also make
DSpark pay), out of scope. Memory is the binding constraint for everything else: f16 KV, -ub 512 and DSpark at -ub 256
all run out of the 25.0 GiB Metal working set.

Remaining levers, ranked: (1) raise `iogpu.wired_limit_mb` (sudo; would allow f16 KV or -ub 512 for prefill),
(2) Metal few-row `mul_mat_id` kernel (would let DSpark win), (3) MLX build (PR #461 measured 40 tok/s on an M4 Max
with a fused decode path; untested here).

### E7 — the Unsloth guide's launch (unsloth.ai/docs/models/inkling-2-mini), 32K — DSpark REJECTED again
Guide command, text only: `-c 32768`, default -ub 512, f16 KV, `-md dspark-Inkling-2-Mini-Q8_0.gguf --spec-type
draft-dspark --spec-draft-n-max 3 -ngld 99`, recommended sampling. Same source build (commit 883286076).
guide32k-nodraft-a: c1 20.95, prefill 207.0 (3.3K). guide32k-dspark3: c1 15.37, prefill 189.3. guide32k-nodraft-b: c1 20.87, prefill 208.3.
At the guide's own settings the drafter costs 27% decode on this M1 Max, as at 128K (E2). The guide's quant choice
(UD-Q4_K_XL, 17.63 GB) is the one used here; UD-Q8_K_XL (22.87 GB) leaves no room for KV in the 25.0 GiB budget.
