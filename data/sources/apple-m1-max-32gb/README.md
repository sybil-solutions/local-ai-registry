# Apple M1 Max 32 GB — owner-machine evidence

Evidence for `registry/recipes/apple/apple-m1-max-32gb/lfm2.5-24b-a2b.mlx-lm.128k.json` (engine
`mlx-lm-lfm2.5-24b-a2b-6bit-128k@01b0cef40877`, weights
`LiquidAI/LFM2.5-24B-A2B-MLX-6bit@1deec53eb49571e21cdcbf82fd5839248f07ed88`) on the new card
`registry/cards/apple/apple-m1-max-32gb.json`. One session on one machine (2026-11-03 21:24–21:36 UTC): the
`lab/lab.py try --on endpoint` run that wrote the recipe, then one extra needle request near the full window, with a
1 Hz MLX memory log running throughout.

## Machine

MacBook Pro 14-inch 2021 (`MacBookPro18,4`), Apple M1 Max, 10-core CPU (8P + 2E), 24-core GPU, 32 GB unified
memory, 400 GB/s. macOS 27.0.1 (26A434), Python 3.12.11, mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.1. AC power.
`iogpu.wired_limit_mb` left at 0 (system default; MLX reports a 26.80 GB / 24.96 GiB recommended working set).
No other model server was running. Details in `host.json`, the whole environment in `pip-freeze.txt`.

`host.json` and `memwatch.jsonl` stand in for the recipe kit's `probe.py` and `memwatch.sh`, which are Linux-only
(kit `ef9782c`: `probe.py` stops at `open("/proc/cpuinfo")` on macOS, and `memwatch.sh` reads `/proc/meminfo`). The
kit's panel and sweep need an SGLang `/generate` endpoint, which mlx-lm does not have, so this run has no kit
submission file; the lab gates and the probe below are the measurement.

## Why MLX

The bounty formats (EXL3, NVFP4, TR3, AWQ) have no Metal kernels, so this uses the model's native Apple format:
Liquid AI's own MLX 6-bit export (19.5 GB). 8-bit (~25 GB) would not leave room for the KV cache under the default
GPU budget; 6-bit is the largest that does.

## Results

| | lab run (recipe proof) | full-window probe |
|---|---|---|
| harness | `lab/lab.py try` gates | `probe_full.py` |
| gates | load chat reasoning tools context speed — 6/6 | needle — pass |
| context prompt tokens | 91,882 (server-reported) | 127,942 (server-reported; 97.6% of 131,072) |
| prefill | 412 tok/s (223.0 s) | 360 tok/s (355.7 s) |
| decode, 1 user | 62.8 tok/s (1,884 tokens in the first 30 s, counted by the served tokenizer) | — |
| peak GPU memory (memwatch) | 22.05 GiB | 23.40 GiB |

The lab's context gate sizes its prompt from a token estimate and filled 70% of the window (the gate requires 60%);
the probe repeats the same needle prompt sized with the served tokenizer to 127,942 tokens, so the 128K window is
shown to fit in memory and recall, not only declared. (#43 measured a `metal::malloc` OOM at 128K for Qwen3.5-9B on
mlx-lm on this same machine, a 20.10 GB single-buffer limit. LFM2.5-24B-A2B keeps KV in only 10 of its 40 layers,
so its 128K cache stays far below that.) The peak counter was reset at 2026-11-03T21:29:45Z
(`memwatch-reset.timestamp`), right after the lab run and before the probe, so every `peak_gb` after it belongs to
the probe alone.

## The launcher

`registry/launches/mlx-lm-lfm2.5-serve.py` (sha256
`7003b19c082de027838540eb23ef5b8f3a15374bfa4361d806b3e8319b95024f`, equal to `launch_config_sha256` in the run
record) is `mlx_lm.server` plus three things the lab needs: a `/v1/token/encode` route (the speed gate counts decoded
tokens with the served tokenizer), a `/v1/models` that lists only the served model by its relative path, and the
model's `generation_config` sampling for requests that leave it out. The launch passes `--max-tokens 131072` so
mlx-lm's 512-token default never caps an answer, and serves one request at a time (`seqs` 1).

## Files

| file | role | sha256 |
|---|---|---|
| `../../lab-runs/apple-m1-max-32gb.lfm2.5-24b-a2b.mlx-lm-lfm2.5-24b-a2b-6bit-128k.128k.20261103T212925.json` | the run record (moved unchanged from `lab/runs/`) | `a0443efb131cb2f07cd41aa7b2763b1836b9f320d94ab9eafd380d48722c09f1` |
| `lab-try.log` | the lab's console output | `2da8f8c228a399dc18f02fbfd69f69f5a1937e8278d2d04ff623123234459a6d` |
| `run-server.log` | server log for the whole session | `f53fff3e48fe602d09506f5738a318a48bce7adf2e90eabdf157fae8deb1327c` |
| `memwatch.jsonl` | 1 Hz MLX memory log, whole session (GiB) | `3a65a35e8cd47af2a89830c5017943ccbf57ee0acf6266e4061e947513ce91d1` |
| `memwatch-reset.timestamp` | peak-reset boundary | `350830c15d12b0841ede864476d60e4ead64cf65e054e3f6fcb59e0c6cfe9e7e` |
| `probe_full.py` | full-window needle probe | `71432ec3abf53f5fd0cf5b609ad89980024f08b3a0f5c2b184278df8634414e2` |
| `probe-full-127942.json` | its result | `77390831307b124346060c7198e4179f39591363abc97a7365aae32668d2053f` |
| `host.json` | machine facts (identifiers omitted) | `6fc18917eb630b7c79be856242ea3390788ac9123ab2cde002a6a613acc1ac67` |
| `pip-freeze.txt` | the environment | `953d0fb14d1718830d5d1d393ea4ed8b39e31b77754755814521d89a3950251f` |
| `run-server.sh` | the measured launch, paths in `$HOME` form | `8d2cb7d1cdb103bf718fe15082a5a5f843cf2f7948933de6fcb6f741cf072025` |
| `memserve_lab.py` | launcher as `__main__` plus the memory log | `41404ea7a83779eb814a9bf8294d306c4f9fa775ebee6172f1fb800e51c42d3c` |

Logs and scripts that name the running account's home directory have it replaced with `$HOME`.
