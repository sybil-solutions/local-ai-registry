# THEORY — Inkling-2-Mini (MLX 4-bit) on Apple M4 Max 64 GB, mlx-lm

## Hardware (measured with tools/hwbench.py, MLX 0.32.4, two runs)
- Read bandwidth 467 GB/s (copy 430 GB/s; spec 546). bf16 GEMM 14.8 TFLOPS at 4096^3 and 8192^3.
- MLX 4-bit affine GEMV reaches ~230 GB/s effective on an 8192x32768 matrix: about half of peak bandwidth.

## Model (mlx-community/Inkling-2-Mini-4bit @ 9856dc53, sizes from safetensors headers)
- 24 layers, hidden 2304, 128 experts top-8 (intermediate 1152), vocab 201,024.
- **Looped:** layers 6..17 run 3 times per token -> 48 layer visits/token (6 + 12x3 + 6).
- Full-attention layers 5, 11, 17, 23 -> 8 full-attention visits; the other 40 visits are 1024-token sliding window.
- Experts NVFP4 in layers 0-19 (573 MB/layer), MXFP8 in 20-23 (~1.05 GB/layer). Attention/gates 4-bit affine.
- Totals: experts 15.67 GB, attention+gates 0.48 GB, lm_head 0.26 GB, embed 0.26 GB, vision 0.64 GB.

## Decode bytes per token (c1, short context)
| Component | GB/token |
|---|---:|
| Routed experts: 44 NVFP4 visits x 35.8 MB + 4 MXFP8 visits x 65.7 MB | 1.84 |
| Attention + message gates + norms: 48 visits x 20 MB | 0.96 |
| LM head (4-bit) | 0.26 |
| **Total** | **3.06** |

-> **c1 decode ceiling ~146 tok/s at 4K** (467 GB/s), 113 at 32K, 63 at 128K, 40 at 256K (KV read of the 8 full-attn visits).
Assumes no expert reuse between loop passes (conservative: reuse would raise the ceiling).

## Prefill (compute bound)
Active linear params/token ~4.81 B (experts 3.06 B + attention 1.29 B + lm_head 0.46 B), 2 FLOP each, plus attention on 8 full visits.
Practical (50% of 14.8 TFLOPS): **753 tok/s at 4K, 659 at 32K, 461 at 128K, 329 at 256K.**

## KV
8 full-attention visits x 2 x 8 KV heads x 128 x 2 B = **32 KiB/token** -> 4.3 GB per 128K, 8.6 GB per 256K stream.
Sliding visits add a fixed ~170 MB (40 x 1024 x 4 KiB). Weights 17.4 GB resident.
Under a 56 GiB cap: (56 - 17.4 - ~3 runtime) GiB / 32 KiB -> ~1.1M tokens of full-attention KV -> e.g. 4 x 256K or 1 x 1M.

## First observation
mlx_lm.generate single run: 36.7 tok/s decode = 25% of ceiling. 27 ms/token vs 6.5 ms of bytes -> dominated by
per-op dispatch: each visit runs dozens of small elementwise kernels (4 MessageGates per looped visit = 144/token).
