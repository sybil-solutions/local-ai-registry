# Inkling-2-Mini on Arc B390: theoretical ceilings

Model: thinkingmachines/Inkling-2-Mini (26.4B total, 128 experts top-8, 24 layers; layers 6–17 form a loop
core run 3× → 48 layer applications per token). Hybrid attention: 20 SWA layers (window 1024), 4 global
(5, 11, 17, 23). Global-attention applications per token: 5, 11×3, 17×3, 23 = **8**. GQA 24 q / 8 kv heads, d=128.
Vocab 201,024. Max context 1M. Drafter: DSpark 3-layer block drafter (block 8, 0.40 GB Q8_0).

## Hardware (measured on this laptop, Ornith hill-climb 2026-11-15)

| | Spec | Measured |
|---|---:|---:|
| LPDDR5X-9600, 128-bit, shared CPU/GPU | 153.6 GB/s | **~115 GB/s** device copy |
| FP16 XMX GEMM | — | **~36 TFLOPS** (4096³), ~20 TFLOPS sustained |

"Practical" = 115 GB/s and 50% of 36 TFLOPS (18 TFLOPS).

## Weights (unsloth UD-Q4_K_XL GGUF = the checkpoint's native formats: NVFP4 experts L0–19, Q8_0 elsewhere)

| Tensor group | Stored | Read per token (c1, no spec) |
|---|---:|---:|
| Routed experts L0–19, NVFP4 (0.573 GB/layer) | 11.47 GB | 44 applications × 8/128 × 0.573 = **1.576 GB** |
| Routed experts L20–23, Q8_0 (1.083 GB/layer) | 4.33 GB | 4 × 8/128 × 1.083 = **0.271 GB** |
| Attention q/gate/o/k/v/rel, Q8_0 (28.5 MB/layer) | 0.68 GB | 48 applications × 28.5 MB = **1.37 GB** |
| Router (F32, 1.18 MB/layer) + loop gates (F32, ~9 MB/core layer) + small | ~0.2 GB | ~0.06 + 0.11×(passes) ≈ **0.15 GB** |
| LM head (Q8_0, 201K × 2304) | 0.49 GB | **0.49 GB** |
| **Target forward, short context** | 17.6 GB | **≈ 3.86 GB/token** |

The loop makes the dense attention weights expensive: they are re-read 48× per token (1.37 GB), more than
the whole 4-layer MXFP8 tail. Requantizing attention to ~4.5 bpw would save ~0.64 GB/token (−17%), LM head
Q6_K ~0.11 GB — at a quality cost that must be measured (KLD vs BF16).

## KV cache

- Global layers: 8 applications × K+V × 8 heads × 128 = 16,384 elements/token → **32 KiB/token f16, 17 KiB q8_0**.
  256K: 8.0 GiB f16 / 4.25 GiB q8_0. 1M: 32 GiB f16 / 17 GiB q8_0.
- SWA layers: 40 applications × 1024 window × 2048 elements ≈ 168 MB/sequence (f16), independent of context.
- Decode at position p reads p × 17 KiB (q8_0): 256K → **4.45 GB/token**, more than the weights.

## c1 decode ceilings (short context)

| Mode | GB per emitted token | Practical (115 GB/s) | Spec (153.6) |
|---|---:|---:|---:|
| No speculation, native quant | 3.86 | **30 tok/s** | 40 |
| No speculation, attention 4.5 bpw + LM head Q6_K | 3.11 | 37 | 49 |
| DSpark, verify 4 rows (n-max 3), accept ~2.5 tok/step* | (1.92 dense + 1.85×3.6 union + 0.9 draft)/2.5 ≈ 3.8 | 30 | 40 |
| DSpark, perfect routing overlap, accept 3 | (1.92 + 1.85×2 + 0.9)/3 ≈ 2.2 | 52 | 70 |

\* Verifying k tokens reads dense+head once but the union of expert picks: 1 − (1 − 8/128)^k →
k=2: 12.1%, k=4: 22.8%, k=8: 40.3% of the expert bytes (×1.9, ×3.6, ×6.4 vs one token) if routing were
independent. Consecutive tokens' routing is correlated in practice, so the real union is smaller — measure it.
With only 128 experts and top-8, **speculation buys much less here than on 256-expert models**; it pays only
if acceptance is high and routing overlap is strong. Must be measured, not assumed.

At 256K context (q8_0 KV): (3.86 + 4.45) GB → **~14 tok/s practical** c1 without speculation.

## c4 aggregate decode (short context, no spec)

4 tokens/step: dense 1.92 GB once + experts union for 4 tokens 1.85 × 3.6 = 6.7 GB → 8.6 GB / 4 tok
→ **~53 tok/s aggregate practical** (71 spec).

## Prefill ceilings (compute bound)

Linear FLOPs/token = 2 × (experts 48 × 8 × 3 × 2304 × 1152 = 3.06 G + attention 48 × 25.9 M = 1.24 G
+ LM head 0.46 G) ≈ **9.5 GF**; SWA scores 40 × 1024 × 24 × 128 × 4 ≈ 0.5 GF; global scores 98,304·p FLOPs at position p.

| Prompt | GF/token avg | Practical (18 TFLOPS) | Spec (36) |
|---|---:|---:|---:|
| 4K | 10.2 | ~1,760 tok/s | 3,500 |
| 32K | 11.6 | ~1,550 | 3,100 |
| 128K | 16.4 | ~1,100 | 2,200 |
| 256K | 22.9 | ~790 | 1,570 |

## TTFT / startup

- Short-prompt TTFT floor: one forward (~34 ms of weight reads) + tokenize → ~50 ms.
- Startup floor: 17.6 GB from page cache (~3–5 s) + shader/kernel JIT (cacheable) → ~10 s.

## RAM

Weights 17.6 GB + mmproj 0.64 GB + KV: 4 × 256K q8_0 = 17 GiB (or 1 × 1M) → ~37 GiB + compute buffers.
Fits the 46 GiB cap; f16 KV at 4 × 256K (32 GiB) does not.

## SGLang-XPU stack (trellis-serve inkling2) — what is actually read per token, and where we are

Formats served: routed experts native (NVFP4 L0–19, MXFP8 L20–23); dense attention/controllers INT8 g32 (fp16 scales,
1.0625 B/param) for decode, bf16 copies for prefill GEMMs; LM head INT8 g32 only; router fp32.

| Per decoded token (c1, short context) | GB |
|---|---:|
| Experts (44 NVFP4 + 4 MXFP8 visits, 8 of 128) | 1.85 |
| Dense attention (48 visits × 26.9 M params × 1.0625 B) | 1.37 |
| Loop controllers (24 visits × 4 modules) + router (fp32) | 0.12 |
| LM head (INT8 g32) | 0.49 |
| **Total** | **3.83** → 30 tok/s at 115 GB/s |

Measured (2026-11-21, decode graph, offline interleaved A/B): 50.4 ms/step = **19.8 tok/s c1 = 66% of the practical
ceiling**. Of the 50 ms, ~37 ms are the GEMVs running at 100–110 GB/s (at bandwidth); the rest is ~150 small kernels/visit
(depth mixes, convs, norms, attention at short context). FP4 dense (opt-in) cuts dense bytes 1.37→0.73 GB: 45.0 ms
(22.2 tok/s) at KLD 0.011 vs bf16 (Q8: 0.002).

Prefill (4K chunk, cold): measured ~460 tok/s at 4K, ~490 at 32K, 353 at 128K, 244 at 256K vs practical ceilings
1,760 / 1,550 / 1,100 / 790 → 26–31%. Gaps: MoE prefill is dequant + per-expert oneDNN GEMMs (~10 TFLOPS on 256-row
batches, plus 1 s/chunk of bf16 dequant traffic); long-context attention (ESIMD DPAS) falls from ~15 to ~5.5 TFLOPS as the
prefix grows (K/V streamed once per 4-query-tile work-group).

Speculative decoding (DSpark block drafter, upstream accept length 3.93 on GB300) is the largest remaining decode lever:
verifying 4–8 tokens reads dense+head once and the union of experts (~2.5–3.6× one token's experts) — estimated ~2× c1.
It needs SGLang-main's DSpark worker + Inkling2 drafter (SWA pools, KV write locations, vocab gather) ported to the
0.5.20-XPU stack.
