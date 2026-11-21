# PLAN: Inkling-2-Mini on M1 Max 32 GB (llama.cpp b12600 Metal)

Targets: c1 decode (primary), prefill, context as large as fits stably under the default Metal budget (no sysctl).
Constraints: text-only, IQ4_NL weights (pinned), lossless levers; q8_0 KV allowed if labelled. AC power, idle host.
Engine: official llama-b12600-bin-macos-arm64.tar.gz sha256 9ced6f8c... (commit 29b52d398, same as the local build).
Harness: skill bench.py unmodified; launch.sh/stop.sh/mem.sh wrappers (macOS footprint + mapped weights); queue.sh lock.

## Log
- R1 (11:05) ctl1/kvf16/ctl2/ub1024/ctl3 at 64K q8_0: c1 20.2/20.97/20.61/-/20.3. kvf16 +3% (beats both ctls, +1.6 GiB). ub1024 REJECTED: Metal kIOGPUCommandBufferCallbackErrorOutOfMemory at load.
- Profile: llama-bench tg128 21.1 tok/s at empty ctx -> not KV- or server-bound. Decode graph = 11,826 nodes, 2 splits. 47 ms/token vs 11.5 ms of weight reads: dispatch-bound.
- Metal env screen (tg64, interleaved): CONCURRENCY_DISABLE -8%, FUSION_DISABLE -4%, GRAPH_OPTIMIZE_DISABLE -7%, SHARED_BUFFERS_ENABLE 0, NO_RESIDENCY 0 -> defaults kept.
- CPU screen: -t 4 -2%, -t 2 -1%, --prio 2 needs root (skip). --poll 0 +4.2% (21.52/21.62 vs 20.66-20.84 ctls), --poll 100 +4%. Take poll 0 (no spin).
- R2 (11:35) 64K q8_0: ctl4 20.15 / poll0 20.46 / ctl5 20.11 / ub512 20.05 / ctl6 20.11. poll0 ACCEPTED (+1.6% server, +4% llama-bench, no CPU spin).
  ub512 REJECTED (prefill 166 vs 164-165, TTFT +7%).
- Verify cost (llama-bench, poll 0): batch 1/2/4/8/16 = 46.5/63.4/97.1/150.5/270 ms -> drafts must be accepted often to pay.
- ngram-simple (n=12,m=8): never drafted on the rewrite probe (0 drafts). ngram-mod (defaults n_match 24, n 48..64):
  greedy 1500-token code rewrite, thinking off: off 20.16/20.24, mod 45.02/45.06 tok/s (2.23x), 1047/1063 accepted; output
  byte-identical off vs on (cmp). Novel prose: no drafts, 20.7 vs 20.1. -> candidate, check panel neutrality at 128K.
- R3 (12:15) 128K ub256 (q8_0 and f16): ALL runs void: Metal kIOGPUCommandBufferCallbackErrorOutOfMemory on first request.
  Cause: MTL0 compute buffer 6552 MiB at 128K/ub256 (3276 MiB at ub128); scales with ctx x ubatch (~ n_kv x n_ub x 24 heads x f32 x2:
  relative-position attention bias is materialised, not fused into FA). Weights 16.4 + KV 2.2 + compute 6.4 > 25 GiB Metal budget.
  Same cause as the ub1024@64K failure. -> 128K needs ub <= 128.
- R4 (12:40) 128K ub128 q8_0 poll0: ctl 20.48/20.47/20.45 c1, prefill4K 149.9-151.6, 16K 130.6-130.7; mod 20.44 (-0.2%, noise), prefill same
  -> ngram-mod ACCEPTED (panel-neutral, 2.23x on rewrites, byte-identical greedy). ub64 REJECTED (prefill -16%).
  f16 KV at 128K not tried: 16.4 weights + 4.35 KV + 3.2 compute = 24 GiB, no headroom under the 25 GiB budget.
- PLATEAU: last three levers (ub512, ub64, ngram-mod on panel) inside noise for c1/panel. Decode is dispatch-bound
  (12.4K graph nodes/token, 47 ms vs 11.5 ms of weight reads, 24% of the 87 tok/s bandwidth ceiling).
  Remaining levers (out of scope, engine work): fuse Inkling's relative-position bias into FA (frees ~3 GiB -> ub256/f16 KV/256K),
  cut node count of the loop controllers / attn-res (decode), MoE batch-verify cost (spec on prose).
- Needles (best config, bench.py --long): 48,493 tokens 3/3 (prefill 84.3 tok/s, 575 s); 93,824 tokens 3/3 (prefill 56.7 tok/s, 1,656 s). Peak 20.2 GiB.
- DSpark draft (unsloth/Inkling-2-Mini-GGUF@ae9327c5 dspark-Inkling-2-Mini-Q8_0.gguf, LFS sha256 verified) REJECTED: mainline
  b12600 refuses it (markov_w1.weight expected 256 x 200058, got 256 x 201024: the file is built for Unsloth's fork).
- Qualification on the clean b12600 tarball, exact launch: lab.py try --on endpoint 6/6, decode 20.1 tok/s, context 91,890 tokens
  (1,646 s), peak 22.36 GiB. Full-window probe 125,950 tokens (96%): code recalled; 91,758 tokens came from the prefix cache.
  Clean build vs hill-climb best: same binary and flags (only alias/port differ); 20.1 (lab, thinking on, first 30 s) vs 20.44 (bench median).

## STATUS (plateau)

| metric | baseline (64K ub256, user launch) | best (128K ub128 poll0 ngram-mod) | ceiling |
|---|---:|---:|---:|
| c1 decode, fresh text (bench median of 6) | 20.2-20.6 | 20.44-20.5 | 87 (short ctx) |
| c1 decode, code rewrite (greedy, 1500 tok) | 20.2 | 45.0 | - |
| prefill 4K / 16K (cold) | 164 / 139 | 151 / 131 | 365 / ~340 |
| prefill to 94K (average) | - | 56.7 | ~240 |
| context window | 65,536 | 131,072 | Metal budget |

Limits, with evidence: decode is dispatch-bound (12.4K graph nodes per token, 47 ms/token vs 11.5 ms of weight reads);
long prefill is bound by the materialised relative-position attention bias (compute buffer ~ ctx x ubatch, forcing ub128 at 128K).
Remaining levers, ranked: (1) fuse the bias into the FA kernel in llama.cpp (frees ~3 GiB: ub256+, f16 KV or 256K; long prefill);
(2) reduce node count of loop controllers / attn-res (decode); (3) a mainline-compatible DSpark draft (decode on prose).
