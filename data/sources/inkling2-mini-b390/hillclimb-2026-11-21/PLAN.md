# Inkling-2-Mini on Arc B390: hill-climb plan (living file)

Harness: `python3 hc.py run configs/X.json [--long 131072|262000]`; `python3 hc.py show`; `bash q.sh a b c` (flock-serialized).
Ledger: `ledger.jsonl`. Ceilings: `THEORY.md`. Best: `configs/final-best.json`, `best-config.json`.

## Direction change (user, 00:45): SGLang/vLLM support, 256K KV — llama.cpp is only a reference
Own SGLang-XPU model (trellis-serve branch inkling2-mini-b390, xpu/exl3xpu/inkling2/): native NVFP4/MXFP8 experts via SYCL,
Q8 dense GEMV, Triton banded-bias sink attention, global KV in SGLang pool (8 visits, fp8), SWA/conv position rings.

## (superseded) Why llama.cpp, not SGLang-XPU + trellis EXL3 (the Ornith stack)
- Inkling 2 is a looped MoE with custom pieces (attention residuals over earlier layers, relative-bias attention,
  short convs on q/k/v, per-pass loop gates, per-pass KV, DSpark block drafter). SGLang-XPU has no model for it and
  there is no EXL3 quant of Mini; porting both + quantizing is weeks of work.
- Mainline llama.cpp gained `inkling2_loop` on 2026-11-20 (#32694; #32704 loop-scale fix still open). Unsloth's
  prebuilt b12588-mix-c0c3de8 carries the loop fixes + DSpark drafter (#32218). It has Vulkan and SYCL backends.
- Weights: unsloth UD-Q4_K_XL = native checkpoint formats (NVFP4 experts L0–19, Q8_0 everything else), 17.6 GB.

## Rules
- **Vision enabled** (user, 00:40): every config loads mmproj-F16.gguf; gates include an image check.
- Stack RAM ≤ 46 GiB (watchdog: xe GTT + cgroup anon). Context target 256K per user; 4 users if it fits.
- AC + `powerprofilesctl performance`; ≥6 c1 samples, median; interleave controls; accept only if it beats both
  neighbouring controls and regresses nothing else >3%.
- Never stop the server mid-prefill: harness waits for /slots idle, then `docker stop -t 60`.
- Never `pkill -f` with a pattern that can match the calling shell (did it once at 00:29: killed my own shell).

## Queue
1. [ ] Backend: Vulkan (Unsloth prebuilt, Mesa 26.2.5 ANV) vs SYCL (oneAPI 2026.0, FP16, oneMKL) — llama-bench + server.
2. [ ] DSpark drafter: n-max 1..7, p-min, draft KV type; accept length at T=1.0 and greedy.
3. [ ] FA on/off, KV q8_0 vs f16 (decode at long ctx reads 17–32 KiB/token of KV), ubatch/batch for prefill.
4. [ ] Parallel 4 × 256K (kv-unified), c4 aggregate.
5. [ ] Startup: Mesa shader cache / SYCL cache persistence, mmap vs no-mmap, --no-warmup.
6. [ ] Requant: attention Q8_0 → Q6_K/Q5_K (read 48×/token), LM head Q6_K; measure KLD vs Q8_0 reference.
7. [ ] Kernel phase: profile (GGML_VK_PERF_LOGGER / SYCL), MoE mat-vec NVFP4 bandwidth, loop-gate F32 matvecs.

## Log (newest first)
- 03:45 **gate256 PASSED (sgl-v3 = trellis 5076a00)**: 256,263 tok, retrieval 3/3, prefill 244 tok/s (TTFT 17.5 min),
  decode@256K 8.6 tok/s, RAM peak 44.67 GiB (thin margin: K/V fp8→fp16 prefill scratch ~1.6 GB/visit at 256K).
  c1 17.8, prefill 4K 460 / 32K 487. Soak deferred to the final build (SKIP_SOAK).
- 03:30 Registry: trellis-serve branch inkling2-mini-b390 pushed (5076a00); registry worktree branch inkling2-mini-b390
  commit 956ae18f (Dockerfile, entrypoint, launch json); image inkling2-b390-recipe:registry builds clean (BUILD_RC=0).
  DSpark: upstream drafter needs SGLang-main internals (SWA pools, KVWriteLoc, vocab gather, fused kernels) → parked.
- 03:20 **Vision works** (vendored upstream Inkling2 vision tower + image processor; 0.5.20 multimodal registration):
  red/42 test image → "red, 42"; results card → "RTX PRO 6000; 190". (Bug on the way: input_embeds not passed to the body.)
- 03:11 **sgl-g2 (ESIMD prefill attention + grouped MoE prefill)**: c1 17.6, c4 34.9, TTFT 1.15 s, prefill 4K 426 / 32K 482 tok/s
  (was 202 / 101), **128K gate PASSED** (128,237 tok, retrieval 3/3, prefill 353 tok/s, TTFT 363 s, decode@128K 11.6),
  RAM peak 43.3 GiB.
- 03:00 Prefill: ESIMD/DPAS attention (fa_esimd.h generalized: D=128, positions, window, bias band, sinks) → global 4K+4K 36→ms
  (Triton was 290), SWA 8 ms (54). Triton XPU tops at ~7 TFLOPS GEMM vs oneDNN 45 → no Triton for compute-bound work.
  MoE prefill regrouped (one gather, per-expert GEMMs into shared buffers, fused SiTU, one scatter) + vectorized dequant.
  Prefill 4K chunk 19.8 s → 8.05 s (~510 tok/s). Remaining: expert GEMMs 3.3 s (small-M oneDNN ~10 TFLOPS), dequant 1.1 s,
  casts/elementwise ~3 s. Next prefill lever: DPAS grouped NVFP4 GEMM (dequant in registers).
- 02:30 First SGLang harness run (sgl-graph): startup 43 s, TTFT 1.8 s, **c1 18.0**, c4 35.4, prefill 4K 202 / 32K 101 tok/s,
  RAM peak 44.8 GiB (close to cap). NB: I ran a GPU probe during its startup window (rule violation) — c1/c4/prefill phases
  were after it and look clean; rerun as sgl-g2.
- 02:20 Decode fusion round: fp32-activation GEMV + SiTU epilogue, fused mix(+norm)/rms/router(+GEMV), in-kernel rel bias,
  single-launch conv, pair-mode MoE plan, SYCL flash-decoding (2.5→~100 GB/s fp8 paged). Graph step 98.7 → 57.9 ms.
  Finding: launch count (10.5K→~5K/step) mattered less than kernel efficiency; GEMVs now at ~100-110 GB/s (bandwidth).
- 01:37 **SGLang-XPU serves Inkling-2-Mini end to end** (chat → "Paris."). KV pool 1,052,672 fp8 tokens (4×256K), weights 20.2 GB,
  startup 39 s. Eager decode 4.5 tok/s c1 (no graphs; ~3000 launches/token + host syncs) → next: graph-capturable decode.
  Bug found: SYCL_CACHE_PERSISTENT=1 segfaults _inkling2.so kernels (repro offline, fresh dir too) → off; NEO cache kept.
- 01:27 Offline driver (tests/t_inkling2_sgl_offline.py) SGLang model vs ref: prefill KL mean 1.8e-3, decode 5e-3, argmax 98.6%.
- 01:16 Reference forward validated: ppl 6.77 (300 tok), coherent greedy; chunked/token-by-token KL at bf16 noise floor.
- 01:06 Kernels: Triton attention (8/8 cases ~5e-3), SYCL NVFP4/MXFP8 MoE GEMV 61–91 GB/s (Triton was 11), prefill dequant+XMX 4.4 TFLOPS,
  Q8 dense GEMV 87–110 GB/s.
- 00:15 Start. Model card + config read; THEORY.md written (c1 ceiling ~30 tok/s no-spec at native quant, ~52 with ideal DSpark).
  Downloads: UD-Q4_K_XL 17.6 GB, dspark Q8_0, mmproj F16. Built inkling2-b390:vk (Arch + prebuilt) and inkling2-b390:sycl (on ornith-hc:dev oneAPI 2026.0).
