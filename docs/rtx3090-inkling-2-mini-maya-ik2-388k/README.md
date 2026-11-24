# Inkling-2-Mini at 388K context on one RTX 3090, every expert in VRAM (maya-ik2)

Engine: [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) at `cc41ab1` + `maya-ik2-session2-k13-full.patch` (host-RAM slots, mid-prompt snapshots, fit-reserve knob, plus K1b–K13 CUDA kernel & graph optimizations).
Build everything with `build.sh` (an empty directory; it ends in the launch's working directory). Results discussed in https://github.com/alesha-pro/maya-inkling-mini/issues/17.
Weights: `unsloth/Inkling-2-Mini-GGUF@ae9327c5` UD-Q4_K_XL, with the experts of blocks 20-23 requantized Q8_0 -> NVFP4 (`requant.py`).
Card: RTX 3090 24 GB (desktop on it, ~0.36 GB), PCIe 3.0 x16, Ryzen 5 5600, CUDA 12.x, driver 615.78.

## What changed vs the 262K/248K config
1. **NVFP4 tail.** UD-Q4_K_XL keeps the experts of layers 0-19 in NVFP4 (the release bits) and layers 20-23 in Q8_0
   (4.03 GiB). Requantizing those 8 tensors to NVFP4 (per-16 UE4M3 scales + a per-expert `.scale`, which maya-ik2 requires)
   saves 1.9 GiB = +123K tokens of FP8 KV (16 KiB/token). Quality, 3 prompts x 253 forced tokens, engine logits:
   KL vs the original 0.0010 / 0.0011 / 0.0021, top-1 98.4-99.6 %. For scale: FP8 KV vs F16 KV (already in the
   stock config) is KL ~0.01.
2. **Fit reserve.** The start-up estimate keeps 1.5 GiB + 0.5 GiB free; the real need is ~1.05 GiB of work buffers (chunk 2048)
   + 256 MiB room. `IK2_FIT_RESERVE_MIB=1280 IK2_FIT_MARGIN_MIB=0` (patch) lets the start plan use it. The expert-cache
   check after the allocation is unchanged, so a context that does not fit still falls back (and says so).
3. **Drafter context 16K, `--parallel 1`.** The drafter costs 0.94 GiB at `IK2_DRAFT_CTX=16384` (1.12 at 65536).
   With one slot the host-RAM conversation slots (`IK2_SLOTS`, patch) work again: a side request no longer evicts a long chat.
4. **CUDA kernel & graph optimizations (K1b–K13).** Lossless (`f16` KLD `0.0007–0.0039`, `code` top-1 `1.0000`, 0 extra VRAM):
   - Vectorized `uint4` FP16 mat-vecs & sinks (`mv_f16`, `mv_f16_rows`, `prep_qkv_r`), 1-pass register-cached `attn_res_1pass_576` & `rms_1pass_576`, vectorized BF16 `loop_gate`, vectorized/fewer-sync routing (`route_token`, `route_rows`, `group_route`).
   - Double-buffered `q8_mma_tile` + fused 1-launch `wqkvrg` (`Q8Alt`, −774 µs/window), 1-tile `n<=4` fast path in `experts_block_gu`, multi-stream CUDA Graph fork/join DAG (`g_cs`).
   - FP8 `V`-load prefetch + `STACK:0` padded `sm_q[24][36]` (`NT=3`) + `ns_global=82` in `attn_decode_mma` / `attn_block_mma` (**845 GB/s** on `T=1` @393K = 100 % of copy BW ceiling).
   - Hybrid GQA online-softmax `draft_attn` (Tensor Core FP16 MMA for the global drafter layer + pure-FP32 online-softmax for the local F32 layers; eliminates 3x KV over-read and all global logits DRAM traffic).

## Results (llama-swap -> server.py, temperature 0, thinking off)

| | stock 248K (`--parallel 2`, drafter 64K) | 388K s1 (`cc41ab1` + s1) | **this 388K (K13)** | 448K no drafter (s1) |
|---|---|---|---|---|
| decode prose / code, short (med 3) | 122 / 263 | 124.8 / 260.1 | **136.7 / 291.1** (+9.6 / +11.9 %) | 134 / 134 |
| plain forced decode (CLI, 2K ctx) | - | 130.7 (7.65 ms) | **140.6 (7.11 ms)** (+7.6 %) | - |
| 64K deep: prefill / decode | 2016 / 98 | 2022 / 105 | - | 2036 / 111 |
| 188K deep | 1552 / 78 | 1557 / 83 | - | 1562 / 84 |
| 244K deep | 1344 / 74 | 1407 / 79 | - | 1410 / 76 |
| 378K deep: prefill (`pp`) | - | 1146.5 | **1157.3** (+0.9 %) | 1149 |
| 378K deep: decode (6q med / mean, drafter) | - | 103.0 / 98.9 | **113.0 / 111.4** (+9.7 / +12.6 %, peak 149.4) | 62 (plain) |
| global FP8 attn `@393K` (`T=1` / `T=6..8`) | - | 728 / 599 GB/s | **845 / 667 GB/s** (+16.1 / +11.2 %) | - |
| needles (3, at 5/50/95 %) | - | 3/3 @395K | **3/3 @395K** (`tg 104–119`) | 3/3 @455K |
| VRAM peak | 23.9 GiB | 23.9 GiB | **23.9 GiB** (23,912 MiB) | 23.9 GiB |

## Measured limits
- Global FP8 attention decode (`attn_decode_mma_fp8`, 8 visits at 393K): **845 GB/s** (7.62 ms) = 100 % of the card's
  842 GB/s copy bandwidth; verify windows `T=4..5`: 789–797 GB/s; `T=6..8`: 662–674 GB/s (GA102 71 TFLOPS FP32-acc MMA limit).
- Plain decode = ~7.11 ms fixed + KV / 845 GB/s (fixed byte floor ~4.6 ms; `mv_q8p` + `experts_mma` are 4.89 ms = 69 % of the token).
- Prefill at 378K (1157 tok/s) is near the compute ceiling: ~30 GFLOP/token, mostly attention.
- 0.5M with every expert resident does not fit: 480K was 0.45 GiB short of the expert cache.

## Tried and rejected
- unsloth UD-IQ4_XS: maya-ik2 refuses it (`blk.20.ffn_gate_up_exps: a row is not whole Q5_K blocks`).
- project-maya v1.1.75 at 512K (offload path, experts in RAM over PCIe 3): 41 tok/s short. At 248K: same speed as this fork.
  Its GGUF drafter is estimated at 1.4 GiB and left out at 388K.
- maya-inkling-mini `IK2_KV_MANAGED=1 IK2_TIER=1` at 524,288: works (needles 3/3 at 457K), but 29 tok/s deep.
- `IK2_DRAFT_CTX=65536` deep: 80 vs 101 tok/s (pre-K11 drafter attention). 8192 / width 5: noise.
- `IK2_DRAFT_SHORTLIST=1`: 3.46 vs 5.0 tokens a window, 197 vs 245 tok/s.
- Kernel sweep rejects: `IK2_EXPERT_MMA_SPLIT` != `(8,4)`, `mv_q8p` `uint2` loads, dynamic-array multi-token `loop_gate` (local spills), 18-block fused `moe_combine_conv_block` (less parallelism than `moe_combine_f32`).

## Files
- `build.sh`: clone + patch + build (engine sm_86, vision on CPU, a shared ggml-base for the requant), pinned downloads,
  tokenizer, the requant with its sha256 check, the server config.
- `requant.py`: copies a GGUF, requantizing matched tensors through ggml's quantizer (NVFP4 with a per-expert `.scale`).
- `maya-ik2-session2-k13-full.patch`: complete patch on `cc41ab1` (slots + mid-snaps + fit-reserve + K1b–K13 kernels).
- `maya-ik2-k13-kernels-only.patch`: Session 2 kernel/graph diff only (3 files on top of the session-1 patch).
- `maya-ik2-slots-midsnap-fitreserve.patch`: original Session 1 patch on `cc41ab1`.
- `ik2-388k.json`: the server config the launch names (copied to `docs/inkling2-mini/` by build.sh).
