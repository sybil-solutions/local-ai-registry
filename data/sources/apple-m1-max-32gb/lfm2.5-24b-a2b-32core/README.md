# LFM2.5-24B-A2B on Apple M1 Max 32 GB, 32-core GPU — second proof for the existing recipe

Evidence for the new first proof of `registry/recipes/apple/apple-m1-max-32gb/lfm2.5-24b-a2b.mlx-lm.128k.json`. The
launch, launcher, weights and Python environment are unchanged from the 24-core proof in `../README.md`
(`launch_sha256` `d8378a7b…6c94b` in both proofs; `evidence/pip-freeze.txt` is identical to `../pip-freeze.txt`;
launcher sha256 `7003b19c…5024f`; all four safetensors and `tokenizer.json` matched the HF LFS sha256 of
`1deec53e`). Only the machine differs: the same chip with 32 GPU cores instead of 24. One owner session on
2026-11-21: a hill-climb with
[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)
@ `fda968e7` (its `bench.py` and `ceiling.py` only), then `lab/lab.py try --on endpoint` against the pinned launch.

## Machine

MacBook Pro 16-inch 2021 (`MacBookPro18,2`), Apple M1 Max, 10-core CPU, **32-core GPU**, 32 GB, macOS 26.6.2 (25G83),
AC power, `iogpu.wired_limit_mb` left at 0 (24.96 GiB Metal working set). 352.7 GB/s read and 9.53 TFLOPS fp16
measured with MLX (`evidence/host.json`, the same machine as the Inkling 2 Mini and Qwen3.8-27B evidence on this
card). No other model server ran; an unrelated `hf download` (~1.2 GiB RSS) and desktop apps were open.

## Results: 24-core vs 32-core, same launch

| | 24-core proof (2026-11-03) | **32-core proof (this run)** | change |
|---|---:|---:|---:|
| lab gates | 6/6 | **6/6** | |
| decode, lab speed gate | 62.8 tok/s | **67.0 tok/s** (2,009 tokens in 30.0 s, served tokenizer) | +7% |
| prefill, lab context gate (91,882 tok) | 412 tok/s (223.0 s) | **524 tok/s** (175.3 s) | +27% |
| peak MLX memory (lab run) | 22.05 GiB | 22.05 GiB | same |

Decode is bandwidth-bound and both machines have the same memory system, so it moves little; prefill is
compute-bound and scales with GPU cores (32/24 = 1.33×). Peak system memory during the run was 29.89 GiB including
other apps; server RSS 15.24 GiB; swap 1.36–1.80 GiB (`evidence/memwatch.jsonl`, `evidence/memwatch-mlx.jsonl`).

Gate details: chat answered red, yellow, blue (finish `stop`); reasoning returned separately, answer `391`;
`get_weather(city="Paris")` and the reply quotes the tool's random code `228eed31`; context recalled `58213` from a
91,882-token prompt; the speed gate's story ran to its natural end at 7,631 tokens in 127.9 s.

## Hill-climb (`bench.py`, same settings as the Inkling 2 Mini and Qwen3.8-27B runs)

Temperature 1.0, c1 decode = median of 6 × 512-token answers from the skill's fixed prompt panel, prefill = prompt
tokens ÷ TTFT on a unique cold prompt; every candidate between controls (raw: `ledger.jsonl`, `logs/`, `configs/`).

| run | c1 decode | prefill ~2.9K | prefill ~22.5K | verdict |
|---|---:|---:|---:|---|
| base-a (pinned launch) | 68.91 | 956.6 | 891.0 | |
| ctl-a | 68.09 | 970.2 | 892.0 | control |
| `--prefill-step-size 4096` | 68.44 | 998.1 | 900.4 | +3% / +1% prefill: too small to change the launch |
| `--prefill-step-size 1024` | 68.21 | 945.5 | 863.0 | rejected |
| ctl-b | 68.10 | 971.1 | 893.5 | control |
| `--prefill-step-size 512` | 68.40 | 858.1 | 808.5 | rejected |
| `--kv-bits 8` | 70.35 | 988.5 | 788.0 | +3% decode, −12% long prefill, +2 GiB peak: rejected |
| ctl-c | 68.68 | 972.1 | 893.6 | control |

Controls stayed within 1%. TTFT on a short prompt: 103–140 ms. No speculative drafter exists for this model (every
small LFM2.5 MLX checkpoint has a 65,536-token vocabulary, this one 128,000), so the pinned launch is the plateau.

Ceilings (`ceiling-lfm25-24b-a2b-m1max.json`, the skill's `ceiling.py`): ~2.45B active parameters per token read
~2.0 GB at 6.5 bits per weight, giving 170 tok/s decode at 4K; measured c1 is 40% of it. Prefill practical ceiling 940 tok/s at 50% MFU;
measured ~970 (51%).

## Files

| file | role |
|---|---|
| `../../../lab-runs/apple-m1-max-32gb.lfm2.5-24b-a2b.mlx-lm-lfm2.5-24b-a2b-6bit-128k.128k.20261121T154935.json` | the run record (moved unchanged from `lab/runs/`; proof log `sha256:14a3102d40e33a50`) |
| `evidence/lab-try.log`, `evidence/run-server.log` | lab console output and the server log |
| `evidence/run-server.sh` | the measured launch, paths in `$HOME` form (same wrapper as `../run-server.sh`) |
| `evidence/memwatch.jsonl`, `evidence/memwatch.sh` | 1 Hz system memory, swap and server RSS |
| `evidence/memwatch-mlx.jsonl` | 1 Hz MLX active/peak/cache memory from `../memserve_lab.py` |
| `evidence/server-start.timestamp`, `evidence/host.json`, `evidence/pip-freeze.txt` | start time, machine facts, environment |
| `ledger.jsonl`, `configs/`, `logs/*.bench.log`, `tools/` | every `bench.py` run, its config and output, and the serve/stop/sweep scripts |
| `logs/*.mem.jsonl` | 1 Hz MLX memory per hill-climb run (peak 19.61 GiB at the pinned launch, 21.99 GiB with `--kv-bits 8`) |
| `ceiling-lfm25-24b-a2b-m1max.json` | ceiling inputs |

Logs and scripts that name the running account's home directory have it replaced with `$HOME`.
