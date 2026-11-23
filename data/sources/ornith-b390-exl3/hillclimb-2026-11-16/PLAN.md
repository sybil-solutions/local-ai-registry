# Hill-climb plan (living file — the loop updates it every iteration)

Harness: `python3 hc.py run configs/X.json [--long 131072|262000]`; `python3 hc.py show`.
Ledger: `ledger.jsonl`. Best-so-far: `best.json` (config name + metrics). Ceilings: `THEORY.md`.

## Rules
- Stack RAM footprint ≤ 46 GiB (harness watchdog). Keep 262,144 context.
- A config is "accepted" only if: ok, c1 retrieval/text sane, and a 128K long run passes retrieval
  (`--long 131072`). Every ~3 accepted improvements, run the full `--long 262000` gate.
- Kernel changes: run `tests/test_moe_k4.py` / bit-exact tests in-container before benchmarking.
- Drain requests before stopping (xe driver fault seen on forced stop mid-prefill).
- Score: primary c1_decode; also track c4_decode_sum, prefill_4k/32k, ttft_short_ms, startup_ready_s, kv_pool_tokens.
  Accept a change if it improves its target metric and regresses no other by >3%.

## Queue (config-only first)
1. [ ] baseline (max-running 4)
2. [ ] EXL3_DRAFT_VOCAB=dv256 × spec steps 1/2/3   (theory: 2 steps + pruned head best)
3. [ ] dv128 / dv512 at best step count
4. [ ] EXL3_SGL_GDN_FOLD=1 + --enable-linear-replayssm-spec (B70 winner)
5. [ ] --linear-attn-prefill-backend intel_xpu
6. [ ] chunk 2048/4096 with --max-prefill-tokens matched
7. [ ] build _C_dnnl.so (oneDNN) → EXL3_INT8_PREFILL=1, EXL3_ONEDNN_ATTN=1
8. [ ] EXL3_SGL_SAMPLE_DEVICE gpu vs cpu (TTFT + decode)
9. [ ] KV: --max-total-tokens 1048576 (4×256K) within RAM cap; mem-fraction-static
10. [ ] startup: persistent SYCL/NEO caches (in harness), AOT device build, --skip-server-warmup, fewer graph bs
11. [ ] cuda-graph-bs-decode incl. 3, mamba cache size, --max-mamba-cache-size

## Kernel phase (after config plateau)
- Profile decode step (EXL3_STEP_TIMING, bench/trace_kernels.py): which ops are < 100 GB/s
- K4 MoE GEMV at M=2..8 (verify) bandwidth; fuse gate/up+act; MR variants
- lm_head 6-bit GEMV; GDN decode kernel; fp8 attention decode at long ctx
- Prefill: MoE grouped GEMM with larger M per expert, INT8 XMX, attention kernel at head_dim 256

## Log (newest first)
- 08:18 baseline-c4: startup 112.7s, TTFT 133ms, c1 42.6 (accept 1.78), prefill 4K 1125 / 32K 1023, KV 264,192, RAM: MemAvailable delta said 14 GiB but xe fdinfo GTT shows 29.4 GiB real; watchdog now uses GTT+anon (stackmem.sh).
  c4 INVALID: max_running capped to 1 by mamba cache (8 slots, 5/req). Fixed: --max-mamba-cache-size 40 everywhere; c4 metric now wall-clock aggregate. Rerun as baseline-mamba40.
- 08:21 baseline-mamba40: startup 41.2s (persistent SYCL/NEO caches: −63%), TTFT 148ms, c1 43.6 (accept 1.79), **c4 84.7 tok/s real**, prefill 4K 1341. Concurrency now 4.
- 08:25 Note: another session runs `local-ai-18701` (llama.cpp Qwen3-0.6B, GPU) — not ours; possible benchmark noise. Always use q.sh (flock) to queue runs.
- Built ornith-hc:dnnl (onednn 2025.3, _C_dnnl.so) for INT8 prefill / oneDNN attention.
- 08:38 sweep: **dv256-s1 c1 49.5 (+14%), c4 94.8 → BEST**. dv256-s2 46.1, s3 39.3: accept_len flat ~1.9 for 2/3 drafts (2nd draft pos accepted ~12%) → suspect MTP chain bug or T=0.6 sampled-verify effect; probing greedy.
  p-lapb c1 47.2/prefill4K 1462 (+3% prefill, within noise), d-fold 43.4 (no gain at s1). p-int8 / p-int8-dnnattn: container exited at startup (logs lost; harness now saves logs/<name>.log). s-gpusample: 500 (XPU topk stall known) → dropped.
  Noise looks ~±5% on c1; treat <5% as no change.
- 08:55 Greedy probes: dv256-s1 greedy c1 54.5 (accept 1.7); s3 greedy accept still 1.9 (SGLang accept rate 0.25–0.33) → **multi-step MTP draft is broken, not a sampling artefact**. dv256-s3-fold 36.1 (no help). Probing --disable-cuda-graph.
- 08:58 Step profile (prof-s1): step 36.3 ms = verify graph 28.5 (2.16 GB → 76 GB/s, 66% of 115) + draft-extend 1.4 + **host gap 6.4 ms**. Greedy vs sampled implies CPU sampler ≈ 3.3 ms/step.
  → Wrote exact two-stage device sampler EXL3_SGL_SAMPLE_DEVICE=xpu2 (block-max top-64 then top-k; no 248K topk, no host copy). Testing s-xpu2 vs s-xpu.
- p-int8 failure: onednn 2025.3 wheel needs libsycl.so.8; rebuilt ornith-hc:dnnl with onednn 2026.0.2 (libsycl.so.9 OK). Retry queued.
- 09:05 p-int8 (onednn 2026.0.2): prefill 4K 1609 / 32K 1120; **p-int8-dnnattn: 4K 1613 (+14%), 32K 1331 (+28%)**, decode unchanged → adopt. Combined c1-int8 (+chunk 2048/4096, +lapb) queued.
  s3-nograph-greedy: accept rate still 0.31 → not a graph bug. Ref: Ornith-1.0 derivative reports 73% accept at MTP K=3 → our multi-step draft is buggy. Probing: VERIFY_AS_DECODE=0, attention-mode decode, draft KV bf16, draft attn triton.
  s-xpu / s-xpu2 sampler: scheduler dies (500); standalone test segfaults rc=139 → isolating which XPU op crashes.
- 09:20 c1-int8 (dv256-s1 + int8 + onednn attn): c1 48.1, prefill 1632/1366. ch2048: 1716/1519. **ch4096: c1 49.3, c4 97.3, prefill 4K 1675 / 32K 1588 (+53% vs baseline) → BEST**. lapb +0. Queued ch8192, ch4096+lapb, then 128K gate.
- 09:25 **gate128-ch4096 PASSED**: 129,243-token prompt, retrieval 3/3, cold prefill 951 tok/s (was 521, +82%), TTFT 136 s (was 252 s), decode @128K 30.6 tok/s.
  ch8192: 4K 1776 / 32K 1589 (≈ ch4096, +1 GiB) → keep 4096. ch4096+lapb no gain.
  Draft probes (vad0, attn-mode decode, draft KV bf16, draft attn triton): all accept 1.9 → bug elsewhere; read-only code investigation delegated.
  Op isolation: only torch.topk over 248K segfaults (multinomial OK). xpu2 sampler switched to Gumbel-max; testing topk sizes + best-xpu2.
- 09:35 **best-xpu2b (3-stage device sampler, Gumbel-max, all topk inputs ≤2K): c1 54.4 (+10%), c4 102.2, prefill 1692/1571 → BEST.** (2-stage version segfaulted in torch.topk over 8192 in-server.)
  Startup breakdown: load_weight 16.4s, scheduler_e2e 21.2s, graphs ~1.6s. Next: prefill graphs for TTFT (120 ms), weight-load speed.
- 09:45 Kernel profile (graphs off, per step 38.6 ms device): dense K4 GEMV 8.9 ms (~85 GB/s), MoE gate-up+down 9.5 (~105 GB/s), LM head 4.0, **Hadamard in/out 4.5 ms (334 tiny launches)**, **GDN update 2.6 ms (~24 GB/s)**, rest ~4. Potential −8 ms/step → ~73 tok/s before multi-step fix.
  Fused GEMV (in-kernel had_in + split-K tail) exists but needs fp16 activations (we run bf16) → candidate: bf16 support. First: sweep EXL3_TARGET_THREADS (tuned for B70's 32 Xe cores; B390 has 12).
  t-pgraph-* failed: mkcfg multi-value bug (fixed), rerun queued.
- 09:50 Kernel: added bf16 support to the fused small-M linear (FusedArgs.bf16 runtime flag; fused_had_in/had_out load/store bf16), EXL3_FUSED=1 env enables it (ops.py). Dev image ornith-hc:dev (rebuilds _moe/_C/_C_dnnl from ~/Projects/trellis-serve/xpu). Correctness+timing test t_fused.py queued, then server A/B.
- 10:05 t-pgraph-*: XPUAttentionBackend asserts graphs only for decode/verify/draft-extend → prefill graphs not possible; TTFT needs another route.
  Draft-bug investigation (read-only): tree/verify kernels, positions, hidden chain, RoPE, weight map all verified correct. s2 also accept 1.92. Top suspect: **M=1 kernel bug** (draft-decode steps are the only 1-row forwards at bs=1). Added EXL3_SGL_MTP_M1=probe|pad to plugin. Running m1-pad-s3 (fix candidate) + m1-probe-s3 (drift log). Other checks if not: greedy s3 vs no-spec output equality; --disable-overlap-schedule.
- 10:10 **k-tt256 (EXL3_TARGET_THREADS=256): c1 60.2, prefill 4K 1950 / 32K 1717** (384: 53.0, 512: 53.7, 768: 50.6, default 1024: 54.4). Verifying 128/192/256-repeat.
  Fused test: fused vs unfused EQUAL at M=2 but differ (rel ~1) at M=1,4,8 → one path is wrong at M≠2. Running t_m1.py against an M=2-batched reference (bf16+fp16, 3 layers).
- 10:20 t_m1.py: **unfused production linears correct at all M** (rel ~1e-3 vs M=2 ref); **fused path broken for K=4** (rel ~1, K=6 OK) → EXL3_FUSED stays off (would need K4 port). m1-pad-s3: accept still 1.9; probe drift ≤4e-3 everywhere → M=1 kernels ruled out.
  Next: greedy equivalence nospec vs s1 vs s3 (eqtest.py) to test target verify rows ≥1; --disable-overlap-schedule probe.
- 10:35 Thread sweep repeat: tt256r 53.9, tt192 52.7, tt128 50.3 → the 60.2 was noise; keep default. **Run-to-run noise ~±10% on c1 → harness should use more decode reps.** s3-nooverlap: accept 1.9 (overlap ruled out).
  eqtest greedy: nospec vs s1 and vs s3 diverge early (28–340 chars) — inconclusive (bf16 vec vs DPAS numerics differ); need logit-level compare of verify rows.
  Poster: poster/ornith-b390-2026-11-15.png (best-xpu2b stats).
- 17:05 MTP shape probe: draft-decode gets 1 hidden row, positions/seq_lens consistent across draft-extend(6)/decode(2)/verify → bookkeeping correct; multi-step bug unresolved (numeric in draft decode, or Ornith MTP weak beyond depth 1). Time-boxed; moving to kernels.
  EXL3_FUSED is now a per-M bitmask (1=all, else bit M). Fused exact at M=2 (= dense verify rows with 1 draft). A/B dev-base vs dev-fused2 (EXL3_FUSED=4), 6 decode samples each.
- 17:05 **Laptop unplugged (battery, balanced profile)** → dev-base 43.2 (vs 54.4) invalid; dev-fused2 500 (check logs on AC). Harness now waits for AC power (wait_ac) and marks runs invalid if on battery at end. Queue paused; resumes on AC.
- 17:35 On AC: ctl-best 53.8 (reproduces best). GDN BV/warps sweep: no gain (bv8 54.6 ≈ noise). dev-base 51.1 (~noise). dev-fused2 crashes (device assert: garbage outputs on some shapes) → fused path shelved.
  Insight: EXL3_MOE_STAGE_MIN_M=0 makes every non-graph forward (prefill, TTFT) stream experts host-USM→device via copy engine (dGPU offload design) — pointless on iGPU. Testing p-nostage (STAGE_MIN_M=1e6).
- 17:50 p-nostage WORSE (4K 1638 vs 1870, 32K 1490 vs 1693) → staging into device buffers helps ⇒ device allocations faster than host USM on B390. Experts currently all host USM for decode. Testing m-res128 (5120 device slots, 128/layer resident, +8.1 GB).
- 18:05 m-res128 (half experts in device slots): c1 50.0, prefill 1553, +6.6 GiB → worse. m-arena-dev(±stage): KV pool shrank to 13.5K ctx (arena counts in SGLang mem budget) and c1 48–50 → rejected. Device memory is NOT faster for decode reads.
  Suspect drift: post-control runs all 48–52. Re-running ctl-best to measure drift (thermal/charging).
- 18:20 **Drift confirmed**: ctl-best2 49.5 vs ctl-best 53.8 (same config, 1 h apart). Recent rejections partly drift. Set power profile → performance (revert: powerprofilesctl set balanced). New method: interleave controls (ctl/cand/ctl). Re-testing g-bv8 and tt256 interleaved.
- 18:45 Interleaved (perf profile): ctl 50.6/51.6/52.1; g-bv8 51.1 (no gain); **tt256 53.0 (+2.2% vs neighbours, prefill +1.7%) → adopt (best-tt256)**.
  eqtest re-read: s1 and s3 diverge from nospec at the SAME char per prompt → verify numerics, not multi-step bug. Conclusion: Ornith-1.5 MTP head weak beyond depth 1 (73% ref was Ornith-1.0). Keep 1 draft.
  Next: prefill kernel profile (16K prompt, chunk 4096).
- 19:00 Prefill profile (4096-token chunk = 1.73 s device): MoE DPAS 594 ms (34%, ~14 TFLOPS = 39% of FP16 peak; MB=32 compile-time, acc fills GRF), GDN 305 (18%), oneDNN INT8 GEMM 153 (9%), Hadamard in/out 155 (9%), SDPA 125 (7%), misc ~300. Kernel backlog: MoE DPAS weight reuse across M-blocks, Hadamard fusion.
  Now: KV pool 4×256K (kv1m, --max-total-tokens 1048576) + full 256K gate.
- 19:20 **kv1m: KV pool 1,048,576 tokens (4×256K), max_running 4, c1 55.3, c4 103.3, prefill 1755/1569, RAM 38.9 GiB → BEST** (4× KV, no speed cost). gate256 interrupted by T3 restart; relaunching detached (setsid).
- 19:45 **gate256-kv1m PASSED**: 258,417 tokens, retrieval 3/3, cold prefill 598 tok/s (2.0× vs 297), TTFT 7m12s (vs 14m40s), decode@256K 20.9 tok/s, RAM 37.7 GiB.
  Next lever (long-ctx decode): verify-as-decode reads the whole KV once PER verify row (2× at 1 draft). Plan: fold the 2 verify tokens into the GQA head dim (one decode call, KV read once, cache_seqlens=L-1) + 1-key LSE merge for token 1. Expected @256K ~21 → 30+ tok/s.
- 20:00 Verify-fold attention: REJECTED + removed. Wrong outputs and slower (3.0 vs 1.7 ms @131K). Key fact: per-row route moves 2×134 MB in 1.7 ms (≈157 GB/s > DRAM) → GPU cache already dedups the 2nd row; KV not read twice. Long-ctx attention ≈ 3.4 ms/layer @256K (~79 GB/s effective).
- 20:20 c4 profile (8 verify rows, no graph 69.7 ms/step): MoE 29.9 (at BW), dense DPAS M=8 11.7 ms (~65 GB/s vs 85 on vec M=2), GDN 7.7, Hadamard 6.7, LM head 4.0. Sweeping EXL3_TARGET_THREADS_MB16 (B70 default 1408) interleaved with controls.
  Startup: pack_expert is cheap for K=4; remaining load time diffuse (imports ~18 s, load 14 s, KV 3 s, graphs+warmup 5 s). Parked.
- 20:55 MB16 sweep (c4; controls 99.6/100.7/100.2): 384 → 105.0 (+4.8%), 704 → 102.4, 1024 → 100.5. Lower better; sweeping 128/192/256 + 384 repeat.
- 21:30 MB16 sweep #2 (controls 101.4/100.8): 256 → 104.4, 192 → 103.8, 128 → 101.7, 384 repeat → 100.2. Real effect ≈ +2–3% at 192–256 → adopt MB16=256 (best-mb256). Config space exhausted (all within ±3% noise).
  c1 gap analysis vs ideal (~90 tok/s for 1 draft + pruned head): dense GEMV+Hadamard 13.4 vs 6.6 ms (biggest), small kernels ~3 ms, GDN 1.8 ms. → Debug fused K4 GEMV (in-kernel Hadamards) properly.
- 21:50 **Fused K4 GEMV FIXED**: root cause = cross-Xe-core ordering on Xe3 (split-K partials held in writer L1). Fix: fused partial block_store L1-uncached + fence(clean) before the arrival atomic. t_m1: fused == unfused at M=1,3,4,8 (K4 and K6, bf16/fp16). Server A/B dctl/dfused-all interleaved.
- 22:20 Fused-all in server: correct output, but c1 50.5/49.6 vs ctl 51.4/52.6/51.3 (−3%), c4 95.5/92.8 vs ~102 (−8%) → slower (serialized split-K tail). Trying M=1,2 only (EXL3_FUSED=6) and M=2 only (=4).
- 22:50 Fused M=1,2 only: c1 50.2 vs ctl 50.5/52.6; M=2 only 49.2 → fused decode REJECTED (graph-mode Hadamard kernels not the bottleneck). Correctness fix kept in source (fused off by default).
  Plateau: c1 ~52, c4 ~102–105, prefill 4K ~1.75K / 32K ~1.57K / 256K 598, startup ~44 s, TTFT ~120–150 ms, KV 1M tokens.
  Next cheap lever: draft vocab size (dv512 98.7% coverage, dv128 87%).
- 23:15 Draft vocab: dv512 accept 1.768 (vs ctl 1.741/1.754), c1 52.0 ≈ ctl 52.1/53.0; dv128 1.682 / 50.6 → keep dv256. Config plateau confirmed.
  Next: dense K4 GEMV microbench (all layer shapes, M=1/2, GB/s) to iterate kernel params offline.
- 23:40 GEMV microbench: big dense matrices fine (cache-inflated), small 0.5 MB ones (shared expert, k/v) only 21–45 GB/s — fixed ~12–25 µs/call (HadIn+GEMV+HadOut). Lever: shared-expert fusion (SGLang supports: expert #256 with sigmoid-gate weight) — we run --disable-shared-experts-fusion. Testing sefuse.
- 00:20 Shared-expert fusion: SGLang only enables it on aiter/CUDA → implemented in plugin (EXL3_SHARED_FUSE=1): store E=257, post-load pack of shared gate_up/down into slot 256, skip block's shared MLP, MoE apply appends id 256 w/ sigmoid(shared_gate(x)). Kernel topk is generic (size(1)). A/B + greedy eq test queued.
  NOTE: EXL3_MOE_STAGE_MIN_M=0 means staging OFF (0 → never). p-nostage was identical to control → its "slower" was drift. Plugin default 512 (staging on for ≥512-row chunks) untested → queue.
- 00:50 sefuse2 (shared expert fused as expert #256, 41 blocks): correct (accept 1.753, sane text), c1 53.4 vs ctl 53.8/52.9, c4 95.6 vs 103.8/98.7 → REJECTED (opt-in code kept). Third kernel-count reduction with no gain → under XPU graphs time is in the main kernels, not launches.
  Testing real MoE staging for prefill: STAGE_MIN_M=512 / 2048 vs off.
- 01:15 MoE staging prefill: stage512 4K 1323/32K 1396, stage2048 1182/1398 vs ctl 1536–1722/1525–1544 → REJECTED (keep off).
- 01:30 GDN verify kernel read: per (head, BV tile) serial over tokens; ~88 µs/call regardless of BV/warps → latency-bound reductions, not BW. Needs hand-written SYCL kernel → kernel backlog #1 for decode.
- 01:50 Deferred MoE prefill NT/MB rework (~7% est., Hadamard-cooperation assumes NT=4). Baked current best into image ornith-b390-exl3:best (plugin + sampler + dv256) with ~/Projects/ornith-b390-exl3/best/run-best.sh; validating as final-best (+128K gate).
- 02:10 **final-best (baked image ornith-b390-exl3:best) VALIDATED**: c1 53.8, c4 102.2, prefill 1778/1571, TTFT 125 ms, startup 42.9 s, 128K gate 3/3 @ 944 tok/s / decode 30.6, RAM 38 GiB; smoke text/vision/tools pass. README updated; run-best.sh is the recommended launcher. Server left running on :8000.
  Kernel backlog (decode): GDN verify update SYCL rewrite (~88 µs/layer-call, latency-bound). (prefill): MoE DPAS NT2/MB64 decode reuse (~7%).
- 02:40 GDN microbench (interleaved, min of 8): orig T=2 38.6 µs vs memcpy of same bytes 35.5 µs → GDN already at BW (earlier 88–96 µs were clock-ramp/profiler artefacts). Static-T variant bit-exact, −9% (0.1 ms/step) → not worth it. Next: GPU clock during decode.
- 03:20 **MoE prefill DPAS MB=64/NT=2 (K=4 only)**: bit-exact vs MB=32 on random experts (M=4096, top-8/256); 16.34 → 14.47 ms/layer (−11.4%, 12.6 → 14.2 TFLOPS). MB48 15.38. Server A/B fb-mb64 vs final-best interleaved.
- 03:55 **fb-mb64 ADOPTED**: prefill 32K 1665/1661 vs ctl 1517/1573 (+7%), 4K 1897/1876 vs 1521/1801; decode unchanged. Tagged ornith-b390-exl3:best = MB64 image (previous kept as :best-prev). Running 256K gate on it.
- 04:20 **gate256-mb64 PASSED**: 258,417 tok, 3/3, prefill 628 tok/s (was 598; 2.1× baseline 297), TTFT 6m51s, decode@256K 21.7, RAM 38.1 GiB.
  Long-ctx prefill is attention-compute bound. Testing chunk 8192 vs 4096 at 128K (lc-*).
- 04:50 lc-ch8192 @128K: 991 vs ctl 979/982 (+1%, noise) → keep 4096. Testing EXL3_SGL_FP8_Q=1 and EXL3_ONEDNN_MIN_SEQ=1024 (oneDNN SDPA from shorter prefixes).
- 05:20 at-fp8q: server crash (FP8 q unsupported in extend) → rejected. at-dnnmin1k ≈ ctl → rejected. Config space exhausted again.
  Soak: 4 concurrent ~250K prompts (fills the 1,048,576-token KV pool), retrieval each, RAM watchdog.
- 06:00 **Soak 4×~248K concurrent PASSED**: 989,879 prompt tokens in the 1M pool, 4/4 streams retrieval 3/3, RAM 36.8 GiB, agg prefill 635 tok/s. BUT streams that finished prefill decoded at 0.2 tok/s while others prefilled (decode starved behind 4096-chunk prefills). Lever: --enable-mixed-chunk / smaller chunks under concurrency.
- 06:40 Interference (3 decoders + 64K prefill): base decode 8.4 tok/s/stream, 64K TTFT 49.8 s; mixed-chunk 6.1 / 76.4 s; ch1024 7.6 / 57.3 s; mixed+ch2048 5.8 / 83.5 s → base best; mixed/smaller chunks rejected. Trying --num-continuous-decode-steps 4/8.
- 07:10 --num-continuous-decode-steps: 4 → 9.2 tok/s / 42.5 s (noise-level), 8 → identical to base (flag likely ignored with spec decoding). KNOWN LIMITATION: a long cold prefill starves concurrent decoders (~1 decode step per 2.4 s chunk); needs scheduler time-slicing work.

## STATUS @ 07:15 (plateau)
Best = image ornith-b390-exl3:best (MB64 MoE prefill), launcher best/run-best.sh, config configs/final-best.json.
c1 ~53 tok/s (median), c4 ~103, prefill 4K ~1.85K / 32K ~1.66K / 128K ~980 / 256K 628 tok/s, TTFT ~125–140 ms,
startup ~43 s, KV 1,048,576 tokens (4×256K proven by soak: 989,879 tok, 4/4 retrieval, 36.8 GiB).
Remaining big levers (all kernel/system work): (1) faster EXL3 dense+MoE decode kernels (graph-mode verify is ~28 ms vs ~19 ms BW ideal), (2) attention-compute at long ctx (oneDNN SDPA ~13 TFLOPS eff), (3) scheduler fairness under long prefill, (4) better drafter for Ornith-1.5 (MTP weak beyond depth 1).
