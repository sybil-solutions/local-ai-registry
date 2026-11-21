# Apple M1 Max 32 GB: Inkling 2 Mini, owner-machine evidence

Evidence for `registry/recipes/apple/apple-m1-max-32gb/inkling-2-mini.llama.cpp.128k.json` (engine
`llama.cpp-metal-inkling-2-mini-iq4-nl-ngram-128k-m1-max-32gb@6fc2e136d4eb`, weights
`ggml-org/Inkling-2-Mini-GGUF@db61c5b492a6a034d986328d28274af0a439aa6c`, `Inkling-2-Mini-IQ4_NL.gguf`). One session on
one machine (2026-11-21): a hill-climb with the
[inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (`hill-climb/`), then the
`lab/lab.py try --on endpoint` run that wrote the recipe, then one needle request near the full window, with a 1 Hz
memory log running throughout the last two.

## Machine and engine

MacBook Pro 14-inch 2021 (`MacBookPro18,4`), Apple M1 Max, 24-core GPU, 32 GB unified memory. macOS 27.2.1 (26B131),
AC power. `iogpu.wired_limit_mb` left at 0 (llama.cpp reports a 26.80 GB recommended working set). Measured 359 GB/s
copy bandwidth and 7.1 TFLOPS fp16 GEMM (`hill-climb/hwbench.py`). Details in `host.json`.

Engine: the official `llama-b12600-bin-macos-arm64.tar.gz` (ggml-org release b12600, commit `29b52d398`, the first
release with the Inkling 2 architecture), unpacked from the verified tarball into a clean directory. The weights file
matches the HF LFS sha256 of revision `db61c5b4`. `run-server.sh` was generated from the launch JSON's `command`.

## Results

| | lab run (recipe proof) | full-window probe |
|---|---|---|
| harness | `lab/lab.py try` gates | `probe_full.py` |
| gates | load chat reasoning tools context speed: 6/6 | needle: pass |
| context prompt tokens | 91,890 (70% of 131,072) | 125,950 (96%), 91,758 of them from the prefix cache |
| prefill | 56 tok/s end to end (1,646 s) | n/a (mostly cache hits) |
| decode, 1 user | 20.1 tok/s (603 tokens in the first 30 s, served tokenizer) | n/a |
| peak memory (memwatch: footprint + mapped weights) | 22.36 GiB | 20.70 GiB |

The probe reused the lab prompt's filler, so most of its prompt came from the prefix cache. It shows that 126K tokens
of KV fit and recall works, not how long a cold 126K prefill takes. Cold long-context prefill from the hill-climb: 48,493
tokens in 575 s and 93,824 tokens in 1,656 s, both with 3/3 needles.

## What the hill-climb found (`hill-climb/PLAN.md`, `THEORY.md`, `ledger.jsonl`)

- **Decode is dispatch-bound, not bandwidth-bound.** The looped architecture runs 48 layer visits per token, and the
  decode graph has about 12,400 nodes. That takes 47 ms per token, while reading the active weights would take 11.5 ms.
  Decode sits at about 20.5 tok/s on fresh text, 24% of the 87 tok/s bandwidth ceiling.
- **`-ub 128` is what fits 128K.** The prompt compute buffer grows with context x ubatch: 6.4 GiB at 128K/ub256 and
  3.2 GiB at ub128. ub256 at 128K and ub1024 at 64K both fail with Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
  ub128 costs about 8% prefill at 4K compared with 64K/ub256. #460 found the same limit on CUDA.
- **`--spec-type ngram-mod`** drafts from earlier context. On a 1,500-token greedy code-rewrite, decode goes from 20.2
  to 45.0 tok/s (1,047 of 1,063 drafts accepted, 2/2 repeats) and the output is byte-identical with it off. On the
  fresh-text panel it never drafts and costs nothing (20.44 vs 20.45-20.48 controls).
- **`--poll 0`**: +4% decode in llama-bench (3/3 against interleaved controls), +1.6% on the server, and no CPU spin.
- **Rejected:** Metal env switches (concurrency, fusion and graph-optimise off are 4-8% slower; the rest is noise), fewer
  threads, ub512 (no prefill gain), ub64 (-16% prefill), f16 KV at 128K (no headroom), and Unsloth's DSpark draft. That
  draft does not load on mainline b12600: its `markov_w1` is 256 x 201,024, and the loader expects the unpadded 200,058 vocab.

Hill-climb numbers come from the skill's `bench.py` (thinking on at the default effort, temperature 1.0, median of 6
512-token answers, a unique cold prompt per prefill length). Gate numbers come from `lab.py`. The two are not directly
comparable.

## Files

| file | role | sha256 |
|---|---|---|
| `../../../lab-runs/apple-m1-max-32gb.inkling-2-mini.llama.cpp-metal-inkling-2-mini-iq4-nl-ngram-128k-m1-max-32gb.128k.20261121T125856.json` | the run record (moved unchanged from `lab/runs/`) | `5cbe9a0e49204dea4543cd6128ac6b8a56786924b015c564ab05a82bc742a4eb` |
| `lab-try.log` | the lab's console output | `926f867f9d0156f14c892474b3de6fb8948a0ce6201714b8001a194b716a5508` |
| `run-server.log` | server log for the lab run and the probe | `3df6e30dd5716891c1ce1b39848b74674e2a1c340532b89297dea415eca25941` |
| `run-server.sh` | the measured launch | `d394a254d97ad82b9847da32af568e809ec50831319e300c9debbd5a52a5d8a1` |
| `memwatch.jsonl` | 1 Hz memory log (GiB) | `6ae3cbee368cfa7a4dedb83e6c164dd955577e90ddead89c2edf95e4dbb31166` |
| `memwatch.sh` | its sampler (`footprint`: phys_footprint + mapped weights) | `6aff9ad6ad1037a98c32d4fda8bf5e35dc14c7247d0e52f3773f4adfd710fca8` |
| `memwatch-reset.timestamp` | boundary: samples after it belong to the probe | `ca12046342265f36b107034bf2452deaffe1d370ee71c5ddbf74c50776e1a04b` |
| `probe_full.py` | full-window needle probe | `a9a5c576d983c53f99433cdaed1f39f9e41705bf5d3926599ec34dc1ec6b3cc6` |
| `probe-full.json` | its result | `ee677d622243776b7c86094598323713d7d588a99113f75437c8238d7556961b` |
| `host.json` | machine facts (identifiers omitted) | `691bd69c93987971046fabc15ae7833e77c08aa43da7919c9284f62bccd0194d` |
| `hill-climb/` | experiment log, ceilings, ledger, configs, harness wrappers, screens | |

Logs that name the running account's home directory have it replaced with `$HOME`.
