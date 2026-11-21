# Inkling-2-Mini on Arc B390 — hill-climb, 2026-11-21

Day-one port and tuning of `thinkingmachines/Inkling-2-Mini` to SGLang 0.5.20 on Intel XPU. The full timestamped log is
[PLAN.md](PLAN.md); the ceilings are in [THEORY.md](THEORY.md); the harness is `hc_sgl.py` (`soak4_sgl.py` for the
4-user soak; `stackmem.sh` measures xe GTT + cgroup anon). Kernel and model tests are in `tests/` (they run inside the
SGLang XPU image with trellis-serve mounted at `/opt/trellis-serve`).

## Path

| Step | c1 decode | prefill 4K / 32K | note |
|---|---:|---:|---|
| PyTorch reference forward (oracle) | — | — | ppl 6.77; independent of SGLang |
| First SGLang-XPU serve, eager | 4.5 | — | Triton MoE 11 GB/s → SYCL GEMV |
| XPU decode graphs (static decode path) | 18.0 | 202 / 101 | 128K and 256K not practical yet |
| ESIMD/DPAS prefill attention + grouped MoE prefill | 17.6 | 426 / 482 | 128K gate 3/3 (353 tok/s) |
| Fused decode pre-processing, 32 SWA splits | 18.9 | 396–460 / 443–487 | 256K gate 3/3 (244 tok/s) |

## Kernels (microbenchmarks, B390, cache-cold)

| Kernel | Before | After |
|---|---:|---:|
| NVFP4 expert GEMV (Triton → SYCL v1 → v2: 4 rows in flight) | 11 GB/s | 109 GB/s |
| INT8 g32 dense GEMV | — | 104–111 GB/s |
| Decode attention, paged FP8 (Triton → SYCL flash-decoding) | 2.5 GB/s | ~100 GB/s |
| Prefill attention 4K+4K global (Triton → ESIMD DPAS) | 290 ms | 36 ms |
| Prefill attention 4K SWA | 54 ms | 8 ms |
| Triton GEMM on Xe3 (reference point) | 7.4 TFLOPS | oneDNN 44.9 TFLOPS |

## Decisions

- **Accepted:** native NVFP4/MXFP8 experts (no requantization); INT8 g32 dense for decode (KLD 0.002 vs bf16 over
  1,024 tokens); fused router+GEMV, depth-mix (+norm), RMS, conv, attention pre-processing; pair-mode MoE decode plan;
  in-kernel relative bias; 32 SWA decode splits (−7% step with fused pre-processing, interleaved A/B); fused FP8→FP16
  K/V gather for prefill (lower peak RAM).
- **Kept optional:** FP4 dense decode weights (−11% step, KLD 0.011).
- **Rejected / not useful:** Triton for compute-bound work on Xe3; torch SDPA (no LSE output, ~1 TFLOPS); persistent
  SYCL program cache (segfaults the kernels); GEMV tile variants beyond the chosen ones (±3% in context).
- **Parked:** DSpark speculative decoding (needs SGLang-main DSpark worker + Inkling2 drafter); DPAS grouped NVFP4 GEMM
  for prefill; long-context prefill attention efficiency (5.5 TFLOPS at 128K+).

Method notes: decode A/B numbers are offline XPU-graph replays of the SGLang model class, one graph per variant in one
process, alternating, 6 × 20 replays (spread ±0.3 ms). Server numbers (`hc_sgl.py`) are medians of 6 streamed 512-token
answers at temperature 1.0 with `reasoning_effort: none`, cold prefill with three hidden retrieval markers.
