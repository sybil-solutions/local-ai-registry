# Inkling-2-Mini IQ4_NL on Apple M1 Max 32 GB: ceilings

Hardware (measured, MLX 0.32.4): 359 GB/s copy bandwidth; 7.10 / 7.13 TFLOPS fp16 GEMM at 4096^3 / 8192^3.

Model (GGUF ggml-org/Inkling-2-Mini-GGUF@db61c5b4, 17.62 GB): arch inkling2_loop, 24 blocks, layers 6..17 looped x3
-> 48 layer evaluations per token. 128 experts, top-8, expert FFN 1152, hidden 2304, 24 q / 8 kv heads x 128.
SWA 1024 on 5 of 6 layers; full attention on layers 5, 11, 17, 23 -> 8 full-attention evaluations with the loop.

Bytes per token (decode, c1): experts 15.80 GB x 2 (loop) x 8/128 = 1.975 GB; non-expert per-layer 0.834 GB x 2 = 1.668 GB;
LM head (Q8_0, 201K vocab) 0.492 GB. Total 4.135 GB -> 87 tok/s at short context (359 / 4.135).
KV: 8 full-attn evals x 2 x 8 x 128 x 1.0625 B (q8_0) = 17.4 KB/token (34.8 KB f16) plus a fixed 1024-token SWA ring.

| context | c1 decode ceiling | c2 aggregate | prefill ceiling (50% MFU, 4.76 B active) |
|---|---:|---:|---:|
| short | 87 | 240 | 373 |
| 32K | 76 | 201 | 319 |
| 64K | 68 | 174 | 279 |
| 128K | 56 | 136 | 222 |
| 256K | 41 | 95 | 158 |

Memory: default Metal budget on 32 GB ~25 GiB (recommendedMaxWorkingSetSize); weights 16.4 GiB -> ~7 GiB for KV and buffers.
