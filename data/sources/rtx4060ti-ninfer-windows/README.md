# RTX 4060 Ti 16 GB — Qwen3.8-27B tuning matrix (Windows, ninfer + llama.cpp)

Source machine: owner-run desktop — Windows 10 Pro 19044, i9-9900K, 64 GB DDR4,
RTX 4060 Ti 16 GB (sm_89, driver 617.14), NVMe Samsung 970. Desktop apps migrated to the
Intel UHD 630 iGPU during measurement (~350 MiB desktop VRAM floor). Weights:
`roofkid/Qwen3.8-27B-GSQ3-NInfer@2359d374` (SHA-256 verified). Engine: roofkid/ninfer-4080
rtx4080-port @ `2469b575`, local MSVC/CUDA 13.4 build (8-file Windows patch, see the kit PR).
All numbers are medians of 3 measured repetitions on a warmed GPU; method notes in the
recipe-kit `REPORTS.md` (branch `ninfer-rtx4080`).

This directory documents the variant search behind the recipe
`recipes/nvidia/rtx-4060-ti-16gb/qwen3.8-27b.ninfer.128k.json` — what was tried, what won,
and where the remaining headroom is. **The card's SOTA is not yet reached**: see "Open levers".

## Decode tok/s by spec profile × workload

| profile | chat prose | structured JSON (e2e) | tiled corpus 8K/32K/64K | depth 98K (novel) |
|---|---|---|---|---|
| **DFlash2 K=7** (recipe profile) | **61.1** | **97.5** | 80.1 / 125.7 / 117.4 | 52.0 |
| MTP3 + n-gram chain (repo default) | 50.0 | 80.0 | **174.7 / 183.4 / 164.7** | 12.7 @32K (collapses) |
| MTP3, n-gram off | — | — | 67.9 @8K | — |
| llama.cpp b11243 + OrcaRouter v2.0, embedded MTP | 37.9 | — | — | 34.6 @32K |
| llama.cpp b11696 + v2.1, spec off | 23.4 | — | — | 21.0 @32K |

Prefill (tok/s): ninfer 1,141/1,075/994/913 at 8K/32K/64K/98K — leads all llama.cpp configs
(886/843/720/557 best). Workload rule learned: DFlash2 wins everything except
saturated-acceptance repetitive content, where MTP3+n-gram is ~1.5× faster; n-gram must stay
OFF with DFlash2 (it competes with the drafter: 236 vs 271 on identical content).

## KV dtype × context (16 GB fit)

| config | max ctx | notes |
|---|---|---|
| rk4v4-e8 | ~129,900 | planner budget 3,402,194,432 B; 131,072 refused by 17.8 MB deterministically; benchmarked profile |
| **rk2v4-e8** | **131,072** ✓ | recipe profile; halves KV bytes, passes the registry 128k floor; six gates + 111K needle recall verified on it |
| host-kv-mib 4→16 GiB | — | no speed change at ≤32K (61.4 vs 61.1): the pool is not the chat bottleneck |

## Context fill on 16 GB

131K fits (llama.cpp 132K config: 24.1 tok/s @depth; 196K-cap baseline: 12.9 @131K).
262K spills on every engine via silent CUDA sysmem fallback (prefill collapses to 19-29 tok/s)
— for strict-VRAM behavior disable the fallback in the NVIDIA driver.

## Open levers (SOTA not yet reached)

- Tiled-corpus scaling vs RTX 4080 is 0.49-0.50× on every point; the 4080 reaches 375 tok/s
  the same way — the 4060 Ti ceiling with saturated acceptance is ~183 today.
- Untried here: DFlash2 K=9/K=12 at saturated acceptance (on the 4080, K=9 did not beat K=7
  but that card is less bandwidth-bound); `--kv-dtype rk2v4-e8` *speed* effect at depth
  (only its fit was needed so far); batch/concurrency > 1 (profile pins max-concurrency 1);
  CUDA-graph-disabled control; newer ninfer commits.
- Quality gate: no Qwen3.8-27B reference panel exists yet; rk2v4-e8 has passed the registry's
  six gates incl. 111K needle recall, but a KL/top-1 panel is still needed for the kit side.

## Files

- `../../lab-runs/rtx-4060-ti-16gb.qwen3.8-27b.ninfer-gsq3-dflash2.128k.20261017T132536.json` — the six-gate evidence run behind the recipe
- `../../lab-runs/rtx-4060-ti-16gb.20261017.ninfer-tiled-corpus-bench.txt` — publisher-methodology bench matrix
- `../../lab-runs/rtx-4060-ti-16gb.20261017.context-fill.txt` — 131K fill / depth-curve raw results
- `../../lab-runs/rtx-4060-ti-16gb.20261017.probe.json` — machine probe (recipe-kit tools/probe.py, windows branch)
