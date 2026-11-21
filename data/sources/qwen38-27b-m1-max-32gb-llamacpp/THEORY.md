# Ceilings: Qwen3.8-27B UD-Q4_K_XL on Apple M1 Max 32 GB (32-core GPU)

Same machine and measurements as the Inkling 2 Mini run: read 352.7 GB/s, fp16 GEMM 9.53 TFLOPS (MLX), Metal
working set 25.0 GiB (26.80 GB), max buffer 20.10 GB.

Model (GGUF header, `tools/gguf_meta.py`): dense, 64 decoder layers, hidden 5120; every 4th layer is full attention
(16 layers, 4 KV heads × 256), the other 48 are Gated DeltaNet with a fixed-size recurrent state; one built-in MTP
layer (`blk.64`, `nextn_predict_layers = 1`). Tensor bytes: 15.89 GB non-embedding (incl. MTP layer), output head
1.043 GB, token_embd 0.715 GB (one row read per token).

Bytes per decoded token (no speculation): ~15.5 + 1.04 = **16.5 GB**, plus KV 64 KiB/token of context (f16; 34 KiB q8_0).
That is 4× Inkling 2 Mini's 4.14 GB, so decode is ~4× lower at the bandwidth limit.

`python3 harness/ceiling.py ceiling-qwen38-27b-m1max.json` (KV counted at f16; q8_0 halves the KV term):

| context | decode c1, no spec | MTP 1 draft @1.7 | prefill practical (50% of 9.53 TFLOPS, 27B active) |
|---|---:|---:|---:|
| 4K | 21 tok/s | 33 | 87 tok/s |
| 32K | 19 | 30 | 79 |
| 108K | 15 | 24 | 63 |
| 128K | 14 | 23 | 60 |

Unlike the MoE Inkling, a dense model's verify step reads the same weights for every drafted token, so MTP can win
on bandwidth here if the Metal small-batch kernels share the weight read.

KV at 128K: 8.0 GiB f16 (does not fit: 16.45 GiB weights + 8 GiB > 25 GiB with compute), 4.25 GiB q8_0.
