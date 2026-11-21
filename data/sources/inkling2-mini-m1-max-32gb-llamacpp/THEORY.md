# Ceilings: Inkling-2-Mini UD-Q4_K_XL on Apple M1 Max 32 GB (32-core GPU)

Measured hardware (`tools/hwbench.py`, MLX 0.32.4, 2 GiB buffer, median of 20, download running in background):
read 352.7 GB/s, copy 367.6 GB/s, fp16 GEMM 8192³ 9.53 TFLOPS. Metal recommended working set 26.80 GB, max buffer 20.10 GB.

Model (GGUF header, `tools/gguf_meta.py`): 24 stored layers, 48 applications per token (layers 6–17 loop 3×),
128 experts top-8, hidden 2304, 8 KV heads × 128, 8 global-attention applications, 40 sliding-window (1024) applications.
Tensor bytes: experts 15.798 GB, dense 0.836 GB, output head 0.492 GB, token_embd 0.492 GB (one row read per token).

Bytes per decoded token (no speculation): dense 1.672 + experts 31.596 × 8/128 = 1.975 + head 0.492 = **4.14 GB**,
plus KV 32 KiB/token of context (f16).

`python3 harness/ceiling.py ceiling-inkling2mini-m1max.json`:

| context | decode ceiling c1 (no spec) | prefill practical (50% of 9.53 TFLOPS) |
|---|---:|---:|
| 4K | 83 tok/s | 423 tok/s |
| 32K | 68 | 376 |
| 108K (lab context gate, 85% of 128K) | 45 | 288 |
| 128K | 42 | 272 |

Speculative rows (DSpark, acceptance assumed, not measured): 1 draft @1.6 → 87 tok/s at 4K; 3 drafts @2.4 → 79.
Because each looped application re-routes 8 of 128 experts, verifying k+1 tokens reads a union of experts that grows
quickly, so on a pure bandwidth model speculation adds little at short context and more at long context (KV is
read once per verify step). If measured decode sits well below the bandwidth ceiling (kernel/launch bound, typical
on Metal at batch 1), speculation can win more than this model predicts.

KV at 128K: 4.29 GB f16 (2.28 GB q8_0). Weights 17.63 GB + KV 4.29 GB = 21.9 GB + compute buffers, under the
26.80 GB working set.
