# PLAN.md — Inkling-2-Mini Hill-Climb on RTX 5060 Ti 16 GB (Windows 11 + WSL2 host)

## Target & Constraints
- **Model:** `thinkingmachines/Inkling-2-Mini` (26.4B looped MoE, 3.4B active, 48 logical visits, DSpark MTP)
- **Hardware:** NVIDIA GeForce RTX 5060 Ti 16 GB (`sm_120a`, 36 SMs, 32 MiB L2, 376.4 GB/s D2D BW, 51.1 TFLOPS FP16 TC) on Windows 11 + WSL2 host (Windows + WSL2 Ubuntu 26.04, CUDA 13.3, Zig 0.17.1)
- **Hard constraints:**
  - **VRAM cap:** ≤ 15.1 GiB (zero WDDM host-RAM spill; Windows desktop uses ~540–760 MiB)
  - **Context:** 48K–64K tokens stable, verified by 3-needle retrieval gates
  - **Concurrency:** `c=1` primary
  - **Quality:** NVFP4 + Q4 KV, verified outputs (no loops, exact 3/3 needle recall)

## Experiment Log

### 1. Baseline — `llama.cpp` (`drofn-16GB-96K` GGUF, `-c 98304 -ctk q8_0 -ctv q8_0 -ub 32 -b 2048 -np 1`)
- `bench.py` (`configs/llamacpp-baseline.json`, 6-prompt panel, `temp=0.6`):
  - `c1_decode`: **45.61 tok/s** median (`[45.70, 45.49, 45.67, 45.59, 45.62, 45.55]`) — 34.3% of `mtp=0` ceiling
  - `ttft_short_ms`: **254.6 ms**
  - `prefill_4096` (3,302 tok): **411.0 tok/s**
  - `prefill_16384` (12,900 tok): **362.5 tok/s**
  - `peak_vram`: **15,934 / 16,311 MiB** (97.7%)
  - `flags`: `[]`

### 2. v0 — `squid5060ti` (Zig 0.17.1 + Blackwell `sm_120a`, 16 GB Zero-Spill Architecture)
- **Changes vs `squid5090` (which would OOM/spill at 15.59+ GiB):**
  1. Default `--embed-host` (`embed.weight` pinned in zero-copy host RAM: **-0.86 GiB VRAM**)
  2. `markov_w1` pinned in zero-copy host RAM (**-98 MiB VRAM**)
  3. Load-time staging (`d_stage`, `d_fold_bf16`) in zero-copy host RAM (**-879 MiB peak load VRAM spike eliminated**)
  4. `prefill_batch_max = 512` instead of `2048` (**-515 MiB runtime VRAM**)
  5. Default `--kv q4` (`9,792 B/tok` across 9 global caches)
- **Results (`ctx=65536`, `--mtp 7 --spec-threshold 0.80`, full 200K draft vocab):**
  - Weights VRAM: **13.59 GiB** (1 arena, 995 tensors) | Runtime VRAM (64K): **1.04 GiB** (1 arena, 401 slices) | Total `nvidia-smi`: **15,735–15,743 MiB** (`308–316 MiB` free)
  - CLI deterministic `mtp=0`: **110.36–111.00 tok/s** (**83.0% of 133 tok/s ceiling**, 2.42× llama.cpp)
  - CLI deterministic `mtp=7`: **144.01 tok/s** (`acc=57.7%`, `dl=2.32`)
  - HTTP `bench.py` controls (`v0-ctl1..ctl4`, `temp=0.6`): **128.38 / 131.65 / 127.80 / 131.97 tok/s** (median of controls **130.0 tok/s**, up to **174.3 tok/s** on code), `ttft_short=83.6 ms`, `prefill_4096=1,755 tok/s` (**4.27×**), `prefill_16384=2,061 tok/s` (**5.69×**).
- **Verdict:** **ACCEPTED** (massive 2.85× decode, 4.3×–5.7× prefill win, zero spill).

### 3. Lever 1 — Pruned 32K Draft Vocabulary (`draft_vocab_32k.bin`)
- **1a (`lever1-dv32k`, pure BPE merge order):** Missed `149859+` space-prefixed (`Ġ...`) block (`5/17` probe coverage) → `94.28 tok/s` vs `128.38 / 131.65` controls. **REJECTED.**
- **1b (`lever1b-dv32k-bal`, 12.7K non-space + 20K space-prefixed):** `84.4%` token coverage (missed `68000+` newlines/code-punct and `196973+` `Ġ+punct`) → `118.00 tok/s` vs `131.65 / 127.80` controls. **REJECTED.**
- **1c (`lever1c-dv32k-cat`, category-aware across all 4 blocks):** `97.7%` token coverage (250/256) → `125.06 tok/s` (beat full vocab on Python LRU cache `175.57` vs `139–158`, but lost on prose prompts) vs `127.80 / 131.97` controls. **REJECTED.**
- **Why:** DSpark drafting runs overlapped on `aux_stream`, so losing even 2.3% draft coverage on prose breaks speculative chains and costs more than the draft LM head savings. Full 200K draft vocab also saves 16 MiB VRAM.

### 4. Lever 2 — Adaptive Speculation Threshold (`--spec-threshold`) & Fixed `--mtp 3`
- `--spec-threshold 0.70`: `125.15 tok/s` (regressed — too many `B=8` verifies on 376 GB/s bus) — **REJECTED.**
- `--spec-threshold 0.80` (control): `128.99 tok/s` (`ttft=84.0 ms`).
- **`--spec-threshold 0.90`:** **`132.61 tok/s`** (+3.6 tok/s over control, `ttft=78.0 ms`) — **ACCEPTED** and baked as new default in `src/main.zig`.
- `--mtp 3` (fixed 3 drafts): `120.25 tok/s` — **REJECTED** (adaptive `{2,4,8}` wins by +12.4 tok/s).

### 5. Lever 3a — 36-SM `LM_HEAD_BLOCKS = 144` (vs 5090's `680`)
- Deterministic `mtp=0` (`ctl → lm144 → ctl`): `111.00 → 110.43 → 110.19 tok/s` (noise).
- HTTP `bench.py` (`ctl1 → lm144 → ctl2`): `123.67 → 126.58 → 132.01 tok/s` (did not beat both controls).
- **Verdict:** **REJECTED** (reverted to `680`). `lm_head` is pure DRAM-bandwidth bound at both grid sizes.

### 6. Lever 3b — 36-SM `KV_SPLIT_K = 16` (vs 5090's `64`)
- Interleaved `longctx` sweep (`v0 → kv16 → v0`):
  - `pos=1024`: `9.16 → 9.13 → 9.16 ms` (`B=1`), `18.22 → 18.12 → 18.22 ms` (`B=4`)
  - `pos=16384`: `10.04 → 10.12 → 10.06 ms` (`B=1`), `18.62 → 18.24 → 18.65 ms` (`B=4`, +2.1%)
  - `pos=32768`: `10.86 → 11.19 → 10.90 ms` (`B=1`, -2.8%), `20.23 → 20.69 → 20.25 ms` (`B=4`, -2.2%)
- **Verdict:** **REJECTED** (reverted to `64`; `16` splits is too coarse at 32K+).

### 7. Stability Gates & WDDM Headroom Audit
- **25,652-token 3-Needle Gate (`ctx=65536`):** `prefill=2,003.1 tok/s`, `decode=165.47 tok/s`, **3/3 needles PASS**.
- **43,867-token 3-Needle Gate (`ctx=65536`):** `prefill=1,879.1 tok/s`, `decode=119.3 tok/s`, **3/3 needles PASS** (`0ES5K3SU`, `OH16QMDL`, `VBWB086A`).
- **WDDM Headroom Finding:** At `ctx=65536`, `nvidia-smi` hit `15,959 MiB` (`92 MiB` free) when Windows desktop apps grew from `540 → 760 MiB`, causing transient WDDM paging on 3 prompts (`101.7 tok/s`). Reducing default serve context to **`--ctx 49152`** (48K, still covering the 44K needle gate) frees `+153 MiB` of guaranteed WDDM headroom (`14.48 GiB` engine total).

## STATUS / Plateau
- **Plateau reached:** 3 consecutive kernel/vocab levers (`draft_vocab_32k`, `LM_HEAD_BLOCKS=144`, `KV_SPLIT_K=16`) landed inside noise or regressed.
- **Bottleneck:** Single-token decode is already at **83.0% of the measured GDDR7 memory-bandwidth ceiling** (`111.0 / 133 tok/s` at `mtp=0`); prefill is at **61.4% of practical FP16 TC ceiling** (`2,061 / 3,359 tok/s` at 16K).
- **Remaining out-of-scope / higher-cost levers:**
  1. Native Blackwell FP8 `cublasLtMatmul` for prefill dense GEMMs (labelled lossy trade; could push 16K prefill toward ~2,600–3,000 tok/s).
  2. Quantizing the 40 local SWA KV caches from BF16 (`170 MiB`) to Q4/FP8 (`-125 MiB` VRAM, enabling 64K ctx with >400 MiB WDDM headroom).
