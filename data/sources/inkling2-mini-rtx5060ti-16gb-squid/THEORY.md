# THEORY.md — Hardware Ceilings for Inkling-2-Mini on RTX 5060 Ti 16 GB

## Measured Hardware (Windows 11 + WSL2 host, WSL2 Ubuntu 26.04, CUDA 13.3, Driver 621.01)

| Property | Measured Value | Spec | Notes |
|---|---:|---:|---|
| GPU | NVIDIA GeForce RTX 5060 Ti | Blackwell `sm_120a` | 36 SMs, 32.0 MiB L2 |
| VRAM | 15.90 GiB total (`~15.1–15.4 GiB` free) | 16 GiB GDDR7 | Windows WDDM reserves ~540 MiB |
| D2D Copy Bandwidth (`cudaMemcpy`, 512 MiB) | **376.4 GB/s** median (`378.2` peak) | 448 GB/s | 84.0% of spec |
| D2D Copy Kernel BW (512 MiB, `float4`) | **372.7 GB/s** median (`375.4` peak) | 448 GB/s | |
| FP16 Tensor Core GEMM `4096³` | **48.3 TFLOPS** median | — | FP32 accumulation |
| FP16 Tensor Core GEMM `8192³` | **51.1 TFLOPS** median (`51.2` peak) | — | Zero thermal throttling |

## Model Geometry (`thinkingmachines/Inkling-2-Mini`)

- **Architecture:** 24 physical layers executed as **48 logical visits** (layers `0..5` once, layers `6..17` looped 3×, layers `18..23` once).
- **Attention:** 24 Q heads, 8 KV heads (`head_dim = 128`, GQA 3:1). Only **4 physical layers** (`5, 11, 17, 23`) = **8 logical visits** have global attention; the other 40 visits use 1,024-token sliding-window local attention (SWA).
- **MoE:** 128 routed experts, `top_k = 8`, `hidden = 2304`, `intermediate = 1152` (~3.4B active params per visit-equivalent, 26.4B total).
- **DSpark MTP:** 3-layer drafter (2 local + 1 global) + rank-256 Markov RNN, up to 7 drafts per step.

## Per-Token Traffic (NVFP4 Mode, `mtp=0`, Short Context)

- Routed experts (48 visits × `8/128` × `w13 + w2` NVFP4): **1.68 GB**
- Dense attention + loop gates + norms (48 visits, NVFP4): **0.86 GB**
- LM head (`200,058 × 2304`, NVFP4): **0.26 GB**
- **Total per emitted token (`mtp=0`):** **2.80 GB**

## Theoretical Ceilings (`scripts/ceiling.py`)

### Single-User Decode (`c=1`, at 376.4 GB/s measured BW)

| Mode | 4K Ctx | 16K Ctx | 32K Ctx | 64K Ctx | Notes |
|---|---:|---:|---:|---:|---|
| **`mtp=0` (no speculation)** | **133 tok/s** | 127 tok/s | 121 tok/s | 109 tok/s | `376.4 / 2.80` |
| **DSpark (chat, `k=3, a=2.5`)** | **125–155 tok/s** | 123 tok/s | 120 tok/s | 116 tok/s | Independent-union lower bound; routing overlap raises ceiling |
| **DSpark (code, `k=4, a=3.8`)** | **160–200 tok/s** | 158 tok/s | 155 tok/s | 150 tok/s | With routing correlation + 32 MiB L2 |
| **Prompt-Lookup (`k=7, a=5.4`)** | **161–240 tok/s** | 160 tok/s | 158 tok/s | 154 tok/s | Zero draft cost (`~5 ns` CPU hash) |

### Prefill (Practical Ceiling at 50% MFU of 51.1 TFLOPS)

| Context | Practical Ceiling | `llama.cpp` Baseline | Headroom |
|---|---:|---:|---:|
| **4K** | **3,649 tok/s** | 411.0 tok/s | **8.9×** |
| **16K** | **3,359 tok/s** | 362.5 tok/s | **9.3×** |
| **32K** | **3,038 tok/s** | — | — |
| **64K** | **2,550 tok/s** | — | — |

### 16 GiB VRAM Cap & Zero-Spill Budget

- Available VRAM under Windows WDDM: **~15.1–15.4 GiB** (hard cap target: **≤ 15.1 GiB** to guarantee zero host-RAM spill).
- Q4 Global KV (`8` target + `1` DSpark = `9` global caches): **`9,792 B/tok`** = **306 MiB @ 32K**, **612 MiB @ 64K**.
- Target zero-spill layout (`squid5060ti`):
  - NVFP4 weights (`embed` + `markov_w1` zero-copy host-pinned, 32K pruned draft vocab): **~14.05 GiB**
  - Runtime arena (`B=512` prefill scratchpad + decode + local SWA KV): **~0.38 GiB**
  - Global Q4 KV (64K context): **~0.60 GiB**
  - **Total: ~15.03 GiB** (fits inside 15.4 GiB free VRAM with **0 GiB WDDM spill**!).
