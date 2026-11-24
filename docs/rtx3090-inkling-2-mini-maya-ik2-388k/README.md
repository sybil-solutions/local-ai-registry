# Inkling-2-Mini at 388K context on one RTX 3090, every expert in VRAM (maya-ik2)

Engine: [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) at `cc41ab1` + `maya-ik2-slots-midsnap-fitreserve.patch`.
Build everything with `build.sh` (an empty directory; it ends in the launch's working directory). Results discussed in https://github.com/alesha-pro/maya-inkling-mini/issues/17.
Weights: `unsloth/Inkling-2-Mini-GGUF@ae9327c5` UD-Q4_K_XL, with the experts of blocks 20-23 requantized Q8_0 -> NVFP4 (`requant.py`).
Card: RTX 3090 24 GB (desktop on it, ~0.36 GB), PCIe 3.0 x16, Ryzen 5 5600, CUDA 12.x, driver 615.78.

## What changed vs the 262K/248K config
1. **NVFP4 tail.** UD-Q4_K_XL keeps the experts of layers 0-19 in NVFP4 (the release bits) and layers 20-23 in Q8_0
   (4.03 GiB). Requantizing those 8 tensors to NVFP4 (per-16 UE4M3 scales + a per-expert `.scale`, which maya-ik2 requires)
   saves 1.9 GiB = +123K tokens of FP8 KV (16 KiB/token). Quality, 3 prompts x 253 forced tokens, engine logits:
   KL vs the original 0.0010 / 0.0011 / 0.0021, top-1 98.4-99.6 %. For scale: FP8 KV vs F16 KV (already in the
   stock config) is KL ~0.01. Plain decode gets a bit faster (fewer bytes): 130 vs ~125 tok/s.
2. **Fit reserve.** The start-up estimate keeps 1.5 GiB + 0.5 GiB free; the real need is ~1.05 GiB of work buffers (chunk 2048)
   + 256 MiB room. `IK2_FIT_RESERVE_MIB=1280 IK2_FIT_MARGIN_MIB=0` (patch) lets the start plan use it. The expert-cache
   check after the allocation is unchanged, so a context that does not fit still falls back (and says so).
3. **Drafter context 16K, `--parallel 1`.** The drafter costs 0.94 GiB at `IK2_DRAFT_CTX=16384` (1.12 at 65536).
   With one slot the host-RAM conversation slots (`IK2_SLOTS`, patch) work again: a side request no longer evicts a long chat.

## Results (llama-swap -> server.py, temperature 0, thinking off, medians of 3)

| | stock 248K (`--parallel 2`, drafter 64K) | this 388K (397,312) | 448K no drafter (458,752) |
|---|---|---|---|
| decode prose / code, short | 122 / 263 | 125 / 261 | 134 / 134 |
| 64K deep: prefill / decode | 2016 / 98 | 2022 / 105 | 2036 / 111 |
| 188K deep | 1552 / 78 | 1557 / 83 | 1562 / 84 |
| 244K deep | 1344 / 74 | 1407 / 79 | 1410 / 76 |
| 378K deep | - | 1147 / 64 | 1149 / 62 |
| code at 378K deep (drafter) | - | 89-101 | - |
| needles (3, at 5/50/95 %) | - | 3/3 @395K | 3/3 @455K |
| VRAM peak | 23.9 GiB | 23.9 GiB | 23.9 GiB |

## Measured limits
- Attention decode over the FP8 cache (`attn_decode_mma_fp8`, one global visit at 393K): 740 GB/s = 88 % of the card's
  842 GB/s copy bandwidth; verify windows of 2..5 tokens cost the same as one (T>=6: -20 %).
- Plain decode = ~7.6 ms fixed + KV / 740 GB/s. The fixed part's byte floor is ~4.6 ms.
- Prefill at 378K (1147 tok/s) is near the compute ceiling: ~30 GFLOP/token, mostly attention.
- 0.5M with every expert resident does not fit: 480K was 0.45 GiB short of the expert cache.

## Tried and rejected
- unsloth UD-IQ4_XS: maya-ik2 refuses it (`blk.20.ffn_gate_up_exps: a row is not whole Q5_K blocks`).
- project-maya v1.1.75 at 512K (offload path, experts in RAM over PCIe 3): 41 tok/s short. At 248K: same speed as this fork.
  Its GGUF drafter is estimated at 1.4 GiB and left out at 388K.
- maya-inkling-mini `IK2_KV_MANAGED=1 IK2_TIER=1` at 524,288: works (needles 3/3 at 457K), but 29 tok/s deep.
- `IK2_DRAFT_CTX=65536` deep: 80 vs 101 tok/s (the drafter's FP16 attention over 64K). 8192 / width 5: noise.
- `IK2_DRAFT_SHORTLIST=1`: 3.46 vs 5.0 tokens a window, 197 vs 245 tok/s.

## Files
- `build.sh`: clone + patch + build (engine sm_86, vision on CPU, a shared ggml-base for the requant), pinned downloads,
  tokenizer, the requant with its sha256 check, the server config.
- `requant.py`: copies a GGUF, requantizing matched tensors through ggml's quantizer (NVFP4 with a per-expert `.scale`).
- `maya-ik2-slots-midsnap-fitreserve.patch`: on `cc41ab1`.
- `ik2-388k.json`: the server config the launch names (copied to `docs/inkling2-mini/` by build.sh).
