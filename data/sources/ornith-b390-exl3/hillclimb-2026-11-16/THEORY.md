# Ornith 1.5 35B-A3B on Arc B390: theoretical ceilings

## Hardware (measured in the serving container, 2026-11-15)

| | Spec | Measured |
|---|---:|---:|
| LPDDR5X-9600, 128-bit, shared CPU/GPU | 153.6 GB/s | **~115 GB/s** device copy, 106 GB/s read-reduce |
| FP16 XMX GEMM | — | **35.8 TFLOPS** (4096³), 20 TFLOPS at 8192³ (clock/thermal drop) |

"Practical" below = 115 GB/s, 50% of 36 TFLOPS. "Spec" = 153.6 GB/s, 100% of 36 TFLOPS.

## Bytes read per decoded token (from checkpoint tensor sizes)

| Component | GB/token |
|---|---:|
| Routed experts (16.26 GB × 8/256) | 0.508 |
| Dense: GDN linear attn 0.515 + full attn 0.137 + shared expert 0.064 + router 0.042 | 0.758 |
| LM head (6-bit, 248,320 vocab) | 0.382 |
| **Target forward, short context** | **1.648** |
| FP8 KV per context position (10 full-attn layers × 2 KV heads × 256 × K/V) | 10,240 B → 2.68 GB at 256K |
| GDN recurrent state (30 × 32 × 128 × 128 × fp16) | ~0.03 read + 0.03 write |
| MTP draft step: MTP layer ~0.03 + **LM head 0.382** | 0.41 (0.06 with a 32K-vocab draft head) |

Verifying k tokens reads dense+head once, experts for the union of k tokens' picks:
union fraction = 1 − (1 − 8/256)^k → 4 tokens 11.9%, 8 tokens 22.4% of 16.26 GB.

## c1 decode ceilings (short context)

| Mode | GB per emitted token | Practical | Spec |
|---|---:|---:|---:|
| No speculation | 1.68 | 68 tok/s | 91 |
| MTP 1 draft, 1.85 tok/step (measured acceptance), full draft head | 1.38 | 83 | 111 |
| MTP 1 draft, pruned 32K draft head | 1.19 | 97 | 129 |
| MTP 2 drafts (~2.5 tok/step), pruned head | 1.11 | **~103** | 138 |
| MTP 3 drafts (~2.95 tok/step), pruned head | 1.14 | 101 | 135 |

**100 tok/s c1 is at the practical bandwidth ceiling**; it requires speculative decoding,
a cheap draft head, and kernels near 100% of measured bandwidth. Baseline 41 tok/s ≈ 57 GB/s effective (≈50%).
At 256K context each step also reads 2.68 GB of KV → c1 ceiling ~ 115/(2.2+2.68)/1.85⁻¹ ≈ 44 tok/s practical (1 draft).

## c4 aggregate decode ceilings (short context)

| Mode | Practical | Spec |
|---|---:|---:|
| No speculation (3.05 GB / 4 tok) | 151 tok/s | 201 |
| MTP 1 draft each, pruned head (4.79 GB / 7.4 tok) | **178** | 237 |

## Prefill ceilings (compute bound)

FLOPs/token ≈ 2 × 2.43 B active linear params = 4.87 GF, + full attention 163,840·p FLOPs at position p, + GDN ~0.28 GF.

| Prompt | GF/token avg | Practical (50% MFU) | Spec (100%) | Baseline | Headroom vs practical |
|---|---:|---:|---:|---:|---:|
| 4K | 5.5 | ~3,300 tok/s | 6,500 | 1,510 | 2.2× |
| 32K | 7.9 | ~2,300 | 4,600 | 1,081 | 2.1× |
| 128K | 15.9 | ~1,130 | 2,260 | 521 | 2.2× |
| 256K | 26.7 | ~670 | 1,350 | 297 | 2.3× |

A 3× prefill gain needs ~70% MFU (or INT8/FP8 XMX for attention/experts, which doubles the peak).
MoE at 1,024-token chunks gives each expert only ~32 rows: low GEMM efficiency is the main prefill gap.

## TTFT / startup

- Short-prompt TTFT floor: one forward (~15 ms of weight reads) + tokenize/schedule → **~30–50 ms**.
- Startup floor: 19.7 GB weights from page cache/NVMe (~3–6 s) + allocation → **~10 s** if kernel JIT and graph capture are cached.

## KV allocation

FP8 KV at 262,144 tokens = 2.75 GiB (+0.25 GiB MTP). Under the 46 GiB cap the pool can hold
4 × 262K = 1.05 M tokens (~11 GiB) so c4 can run each stream at full context.
