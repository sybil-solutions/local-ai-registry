# Qwen3.8-27B on M1 Max 32 GB: hill-climb log

Skill: sybil-solutions/inference-optimisation-skill @ fda968e7 (read in full during the Inkling 2 Mini run; only
`bench.py` and `ceiling.py` used, with `python3 -I`, from a separate copy). Same machine, engine, harness and benchmark
settings as `data/sources/inkling2-mini-m1-max-32gb-llamacpp/`, so the two runs compare directly.

## Targets and constraints

- Metrics, in order: c1 decode tok/s (thinking on), prefill tok/s at 4K/32K, TTFT, startup.
- Context that must stay stable: 131,072 tokens (registry KV floor), one sequence.
- Memory cap: the Metal recommended working set, 26.80 GB (24.96 GiB); no `iogpu.wired_limit_mb` change, no sudo.
- Quality: q8_0 KV as a labelled trade (same as the Inkling recipe); speculative decoding is lossless in distribution.
- Fixed environment: AC power, no power-profile or clock changes, no other model server (Inkling server stopped first).

## Inventory

- MacBook Pro 16-inch 2021 (MacBookPro18,2), M1 Max, 32-core GPU, 32 GB, macOS 26.6.2 (25G83).
- Engine: unslothai/llama.cpp b12588-mix-c0c3de8 (source build for the hill-climb, macOS prebuilt for the gates;
  both commit 883286076) — the same build as the Inkling run, so engine differences do not enter the comparison.
- Weights: unsloth/Qwen3.8-27B-GGUF@4af51b3542f775b0a8e4b310bfb3f17e01c306c9, Qwen3.8-27B-UD-Q4_K_XL.gguf (17.66 GB,
  Dynamic V3.1), built-in MTP layer. Sampling: Qwen thinking mode (temp 1.0, top-p 0.95, top-k 20, min-p 0).

All runs: `harness/bench.py`, temperature 1.0, thinking on (template default), c1 decode = median of 6 × 512-token
answers from the fixed prompt panel; prefill = prompt_tokens ÷ TTFT on a unique cold prompt.

## Experiments

### E1 — baseline: 128K, -ub 256, q8_0 KV (the Inkling recipe's launch shape) — base-a
Metal buffers: weights 994.65 + 14,819.41 MiB, KV 4,352 MiB (q8_0, 16 attention layers), recurrent state 149.62 MiB,
compute 638.54 MiB = 20,954 MiB of 25,558 MiB. (Inkling at the same settings: 23,326 MiB; its compute buffer is 4.7 GiB
because of its full-window relative-bias attention.) Sanity: 17×23 → `391`, thinking returned separately.
c1 **11.89** tok/s [11.89 11.90 11.89 11.90 11.90 11.89], prefill 124.1 (2,839 tok) / 116.6 (21,573 tok),
TTFT short 417 ms, peak system memory 29.46 GiB (incl. ~14 GiB other apps).
Decode = 57% of the 21 tok/s bandwidth ceiling (Inkling: 25% of 83). Prefill = 71% of fp16 peak, above the skill's
50%-MFU "practical" ceiling, which is therefore conservative for this dense model. Prefill falls only 6% from 2.8K to
21.6K (Inkling: −25% from 3.3K to 25.5K): 48 of 64 layers are linear-time Gated DeltaNet.

### E2 — built-in MTP head (`--spec-type draft-mtp`), draft length sweep — ACCEPTED at 3 drafts
No extra file and no memory problem (peak system memory 29.6–29.95 GiB vs 28.6–29.5 without). Round 1 (controls
ctl-a 11.92, ctl-b 11.14) and round 2 (controls ctl-c 10.84, ctl-d 10.82), medians of 6:

| draft n | c1 median | samples | prefill 2.8K | acceptance (whole run) |
|---|---:|---|---:|---:|
| 1 | 12.23 | 12.11 12.18 12.16 12.36 12.48 12.28 | 122.9 | 0.633 (1195/1888) |
| 2 | 14.29 | 14.52 13.73 14.07 14.06 15.30 14.57 | 124.5 | 0.501 (1543/3082) |
| **3** | **15.03** | 15.03 15.17 13.48 14.37 15.03 15.48 | 122.3 | 0.389 (1660/4272) |
| 4 (round 2) | 14.38 | 14.09 13.48 14.25 15.03 16.03 14.51 | 119.4 | 0.340 (1731/5092) |
| 5 (round 2) | 13.54 | 13.36 15.82 13.03 13.47 14.59 13.61 | 117.2 | 0.275 (1783/6478) |
| **3, repeat (round 2)** | **14.84** | 18.28 15.27 12.70 12.52 15.52 14.41 | 118.7 | 0.418 (1714/4104) |

3 drafts beat both neighbouring controls by 26–37% in both rounds and beat 2, 4 and 5 drafts; prefill stays between
the controls (no regression). With 3 drafts, ~2.2 tokens per step at 15 tok/s means a 4-token verify costs ~1.6× a
1-token step: unlike the MoE Inkling (where a 2-token verify cost 1.98×), every drafted token of this dense model
shares one read of the weights. Per-sample spread grows with drafting (content-dependent acceptance), so medians matter.

Drift: controls fell 11.92 → 11.14 → 10.84 → 10.82 over ~70 minutes of back-to-back runs, with no thermal warning in
`pmset -g therm`; swap grew from 0.8 to 1.66 GiB while ~14 GiB of other desktop apps stayed open. Memory pressure is a
likely cause; not verified. Interleaved controls keep every comparison within one drift state.

### E3 — larger ubatch for prefill — REJECTED
Unlike Inkling, the compute buffer is small (639 MiB at -ub 256), so larger ubatches fit. Between ctl-b (121.7) and
ctl-c (119.7): -ub 512 prefill 120.8, decode 10.86; -ub 1024 prefill 112.9, decode 10.88. No gain; keep -ub 256.

### E4 — f16 KV — not run
KV 8,704 MiB f16 + weights 15,814 + state 150 + compute 639 = 25,307 MiB, at the edge of the 25,558 MiB budget that
already failed for Inkling at 25,342 MiB. Kept q8_0 (same labelled trade as the Inkling recipe).

## STATUS — plateau (E3 rejected, MTP length sweep closed at its peak)

| metric | baseline (base-a) | best (MTP 3) | ceiling | how |
|---|---:|---:|---:|---|
| c1 decode | 11.89 | 14.84–15.03 (controls 10.8–11.9) | 21 no spec / 39 with 2.2 accepted | bench.py median of 6 × 512 tok, temp 1.0, thinking on |
| prefill ~2.8K | 124.1 | 118.7–122.3 (controls 119.7–127.4) | 87 at 50% MFU (174 at 100%) | cold unique prompt |
| prefill ~21.6K | 116.6 | — | 79 at 50% MFU | same |
| KV capacity | 131,072 tok × 1 | same | — | q8_0, 4,352 MiB |
| peak system memory | 29.46 GiB | 29.58–29.95 GiB | 32 GiB | incl. ~14 GiB other apps |

What limits further gains: without speculation decode reaches 57% of the bandwidth ceiling (Inkling: 25%), so this
dense model is much closer to the memory-bandwidth limit; prefill runs at ~71% of fp16 peak. Remaining levers: a
raised `iogpu.wired_limit_mb` (sudo) for f16 KV, and MLX builds (registry has mlx-vlm recipes for this model on other
Macs), both untested here.
