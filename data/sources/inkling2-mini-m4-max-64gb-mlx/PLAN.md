# PLAN — Inkling-2-Mini on mlx-lm, M4 Max 64 GB

Targets (priority): c1 decode tok/s, prefill 4K/32K/long, c4 aggregate, KV capacity (registry floor 131,072), TTFT.
Constraints: memory cap 56 GiB footprint (64 GB machine); lossless only (no requant), context must pass needle gates.
Environment: AC power required (harness refuses battery), powermode 0 (default), one GPU job at a time.

## Pins
- mlx-lm: jundot/mlx-lm @ 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (ml-explore/mlx-lm PR #2086, Inkling 2 Mini)
- mlx 0.32.4; weights mlx-community/Inkling-2-Mini-4bit @ 9856dc53edead71405f8348930937c3638ff21a2
- Launcher: inkling2_mlx_serve.py (adds Inkling response format: reasoning split + tool calls, /v1/token/encode)

## Harness notes
- skill-scripts/bench.py unchanged; tools/queue_mac.sh replaces queue.sh (no flock on macOS);
  tools/mem_mac.sh = `footprint` phys footprint (includes IOAccelerator/Metal residency).
- mlx_lm.server loads the model lazily on the first request: bench `startup_s` (~2 s) is NOT model load.

## Log
### 2026-11-20 17:23 baseline `base` (mlx_lm.server defaults via launcher, no patches)
c1 31.17 tok/s (30.1–32.1, n=6, temp 1.0, thinking on, 512 tok) · c4 85.6 agg (22.0/stream) · prefill 749 @3.3K, 732 @25.5K tok
· TTFT short 238 ms · peak footprint 25.7 GiB. Prefill ≈ practical ceiling (~100% of 50%-MFU estimate); c1 = 21% of 146 ceiling.
### 17:27–17:40 profiling + fusion patches (in-process tools/stepbench.py, 1K ctx, greedy, interleaved)
- Step split: graph build 5.5 ms (CPU, overlapped by async eval) vs GPU eval 20–24 ms -> GPU-side bound.
- Component profile (net of ~90 us eval sync): MoE NVFP4 ~163 us x44 = 7.2 ms (ideal 3.4); attention+gates+norms ~240 us/visit
  x48 = ~11 ms (ideal ~2); lm_head 0.5 ms (at bandwidth). Small 4-bit GEMVs are latency bound (q_proj 3.5 MB in 32 us = 110 GB/s).
- v1 = mx.compile fusion of gates/mask/moe/conv/attn elementwise chains: 28.6 -> 25.7 ms/tok (+11%), BIT-EXACT (0.0 logit diff,
  144/144 greedy tokens). Individually only `mask` (-1.5 ms) is clearly outside noise; others <=0.5 ms each.
- `gates3` (batch q/k/v message gates, concat forget+message weights): +6.5% alone. Not bit-exact: KL 0.00015, max 1.5.
- `proj` (fuse q/gate/r/sink GEMVs): REJECTED — no speed gain alone (27.9 vs 27.5 ms) and perturbs logits (KL 0.0014).
- Noise floor for "equivalent" numerics: prefill chunk 512 vs 256 on the unpatched model = KL 0.0044, max 2.66, mean 0.64.
  gates3 is 30x below that -> accepted as numerically equivalent (labelled: not bit-exact).
- v2 = v1 + gates3: 23.8 ms/tok (~42 tok/s) vs base ~28.5 (+20%). all (v2+proj) 23.6 -> same within noise.
### 17:33 server A/B ctl-1 / fast-v2 / ctl-2 (bench.py, n=6)
c1: 32.56 / **38.96** / 33.73 (+15.5–19.7%, beats both). c4: 59.4(disturbed) / 85.2 / 84.7. Prefill 3.3K: 454(disturbed) / 640 / 717;
25K: 509(disturbed) / 622 / 696 -> **prefill regressed ~11% vs ctl-2: REJECTED as-is.** ctl-1's c4/prefill phases were disturbed.
Cause hypothesis: compiled paths in prefill (gates3 shape-specialised compile recompiles for every new chunk length).
Fix: `decode_only` dispatcher — fused paths only when seq len == 1; prefill runs mlx-lm reference code.
In-process after fix: prefill logits bit-exact (0.0); decode 23.9 vs 28.3 ms, greedy identical; interleaved 16K prefill
v2d 573–683 vs base 570–661 tok/s (no systematic gap; whole machine drifting ~800 -> ~620; Activity Monitor at 96% CPU).
### 17:53 server A/B ctl-3 / fast-v2d / ctl-4 (decode-only fused paths)
c1: 33.76 / **40.35** / 35.20 (+14.6–19.5%, beats both; spreads 33.6–35.4 vs 40.1–40.7). c4: 91.0 / **94.3** / 89.5.
Prefill 3.3K: 839 / 816 / 779; 25K: 792 / 775 / 751 -> between controls, no >3% regression vs their mean. **ACCEPTED -> best-config.json.**
### 18:00 gate: needle ~50% (best config, --max-tokens 262144)
102,426-token prompt: **retrieval 3/3**, cold prefill 535 tok/s (≈ practical ceiling ~500 at 100K), decode at that context 27.7,
c1 40.9 (n=2), peak footprint 28.6 GiB.
### 18:04 gate: needle ~92% of 256K (best config)
242,039-token prompt: **retrieval 3/3**, cold prefill 330 tok/s (= practical ceiling 329 @256K), TTFT 732 s, decode at that
context 20.8 tok/s (ceiling ~40), peak footprint **40.4 GiB** (weights 17.4 + KV ~7.9 + prefill transients/prompt cache) ->
recipe = 1 x 256K (2 x 256K would not fit with headroom on 64 GB).
### 18:20 profiling after v2d + kernel-gate attempt
- Ablation (profiling only): MoE ≈ 4.9 ms/tok (experts ≈ 3.5 ≈ bandwidth ideal 3.9; router ≈ 1.4), message gates ≈ 8.3 ms.
- Dependent-chain latency: ~7 us per kernel link (even a trivial add). Reference gate 36.5 us/link, compiled 35.2.
- `kgates` (custom Metal: norm+down kernel, up+tanh+combine kernel): 31.5 us/link alone but in-model 24.3–24.5 vs v2 23.2–23.3 ms
  (stack/slice copies around it): **REJECTED.**
- Conclusion: ~23 ms/token ≈ 3,300 dependent kernel links. Shaving kernels has diminishing returns; next lever class =
  more tokens per chain (speculation / batching).
### 18:25 speculation feasibility
Forward cost at 1K ctx (best paths): L=1 23.2 ms, L=2 37.6, L=4 46.3, L=8 61.6 (L>1 runs the unfused reference path).
k=1 drafts would need >60% acceptance just to break even. No draft available to mlx-lm: Inkling's DSpark head (mtp.safetensors
in thinkingmachines/Inkling-2-Mini) is unsupported by mlx-lm and absent from the MLX conversion; no n-gram mode in mlx_lm.server.
Gate re-check on best config: chat/reasoning/tools round trip all pass.

## STATUS / plateau (2026-11-20 18:25)
| Metric | Baseline | Best | Ceiling |
|---|---:|---:|---:|
| c1 decode (bench.py, n=6, temp 1.0, thinking on) | 31.2 (controls 32.6–35.2) | **40.4** | ~146 @4K |
| c4 aggregate | 85.6 (controls 89.5–91.0) | **94.3** | — |
| Prefill ~3.3K / ~25K | 749 / 732 | 816 / 775 (= controls) | 753 / 659 practical |
| Prefill @102K / @242K | — | 535 / 330 | ~500 / 329 |
| Needle 102K / 242K | — | 3/3 / 3/3 | — |
| Peak footprint (242K prompt) | — | 40.4 GiB | cap 56 |
Limits: ~7 us per dependent Metal kernel x ~3,300 links/token (48 looped layer visits). Experts already ≈ bandwidth.
Remaining levers, ranked (payoff x likelihood / cost):
1. Port DSpark draft head to mlx-lm + fused small-L verify path (L=2..8): potentially 1.5–2x c1; multi-day.
2. Fewer kernels in attention/residual chain (merge depth-attention + norms per visit): maybe +5–10%; days.
3. Upstream the fused decode paths to ml-explore/mlx-lm#2086 so the recipe can drop the runtime patch.
### 18:33 clean build + registry lab (lab.py try --on endpoint) on the clean build
- Clean venv from the rendered launch's pip list only (mlx-lm from ml-explore/mlx-lm@0c4c9919 upstream URL): builds;
  pip freeze identical to hill-climb venv except the mlx-lm source URL. Launcher written from rendered config (sha256 e04fd3dd…).
- Published launcher = hill-climb launcher minus rejected code (proj, kgates); recipe flags add --decode-concurrency 1
  --prompt-concurrency 1 --prompt-cache-size 1 (seqs=1).
- **All six gates pass**: chat, reasoning (391), tools (code quoted), context 58213 @184,730 tokens, speed **40.2 tok/s**
  (tokenizer-counted, 30 s window), prefill 411. Peak footprint during the run 34.0 GiB (110 samples).
### 18:48 clean build vs hill-climb best (bench.py, n=6)
hc-best-1 / **clean-recipe** / hc-best-2: c1 39.98 / **39.53** (37.2–41.1) / 40.80; prefill 3.3K 825 / 667 / 730, 25K 754 / 683 / 705;
c4 95.3 / 39.5 (expected: recipe serves one request at a time) / 89.2. Prefill looked 7–15% low -> investigated before publishing:
### 19:00 clean build, recipe flags vs no flags (interleaved, --skip-cN --reps 3)
cr-1 / cn-1 / cr-2 / cn-2: c1 40.68 / 41.22 / 40.93 / 40.86; prefill 3.3K 828 / 796 / 765 / 771; 13K 789 / 753 / 741 / 742;
25K 775 / 747 / 731 / 730. Flags make no difference, and the clean build's prefill (765–828) matches the hill-climb range
(730–825): the 667/683 run was single-request noise (each prefill cell is one cold request, ±10%).
**Clean build = hill-climb best within noise. Recipe qualified.**
