# Apple M4 Max 36 GB — owner-machine evidence

Evidence for `registry/recipes/apple/apple-m4-max-36gb/qwen3.8-27b.mlx-vlm.200k.json`
(engine `mlx-vlm-qwen3.8-27b-4bit-mtp-200k@bd7091b5a9e0`, weights
`mlx-community/Qwen3.8-27B-4bit@10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, MTP drafter
`mlx-community/Qwen3.8-27B-MTP-4bit` revision `b643c01b…`) on
`registry/cards/apple/apple-m4-max-36gb.json`. Produced on the owner's machine — the same
Mac that produced the recorded run. The recipe's proof still binds the recorded run; this
directory adds the machine evidence requested in review.

## Two runs, one machine — read this first

1. **The recorded run (2026-10-16 20:55–21:32 MST / 2026-10-17 03:55–04:32 UTC)** produced the
   recipe's proof. It was driven by `lab/lab.py try`, whose gate harness (load, chat, reasoning,
   tools, context, speed — all passed) records results inside the run record at
   `data/lab-runs/apple-m4-max-36gb.qwen3.8-27b.mlx-vlm-qwen3.8-27b-4bit-mtp-200k.200k.20261017T043236.json`
   but persists **no memwatch or probe artifact** — that harness has no monitor. That is why none
   from that night exists. The run record's payload without its trailing newline hashes to
   `61943262030a8b32…`, the recipe's `proof.log` value; whole-file sha256 is
   `7f477b5247416520e6f3e4503631a40dc99347ee359e51c2aa497bfcf4269a5f` (unchanged by the move
   from `lab/runs/`). Its own server log (`original-run-server.log`) reports the context gate
   server-side: `prompt_tokens=166667 … elapsed=1916.211s rate=87.0 tok/s`, matching the run
   record's `prefill: 87` and anchoring these logs to that run.
2. **The reproduction run (2026-10-22 22:15–23:02 MST / 2026-10-23 05:15–06:02 UTC)** produced
   everything in this directory. Same machine, same venv (`pip-freeze.txt`), same pinned local
   weights, same flags, and the **byte-identical frozen launcher**
   `registry/launches/mlx-vlm-qwen3.8-long-prefill.py` — sha256
   `f8cd617060c26a5618d75a314931717779a72848f443c18a645e1db2d165f94a`, equal to
   `launch_config_sha256` in the run record. Two deliberate deltas from the recorded launch,
   neither on the model path: the repo's 1 Hz MLX memory sampler
   (`data/sources/qwen38-apple-mlx/memserve.py`) threads alongside the server (`memserve_lab.py`
   runs the launcher's own `__main__` sequence with the monitor added), and `caffeinate -dimsu`
   keeps the Mac awake. Probes are the repo's own `data/sources/qwen38-apple-mlx/probes.py`.

## Recorded vs reproduction

| | recorded (2026-10-16/17) | reproduction (2026-10-22/23) |
|---|---|---|
| harness | `lab/lab.py` gates | `data/sources/qwen38-apple-mlx/probes.py` |
| gates | load chat reasoning tools context speed — 6/6 | chat reasoning tools vision mtp — 5/5, long — pass |
| context prompt tokens | 166,667 (server-reported) | 166,723 (server-reported) |
| peak GPU memory | 20.78 GiB (`peak_gib`) | 20.796 GiB (memwatch, post-reset); server `timings.peak_memory` 22.33 GB decimal = 20.79 GiB |
| prefill @ ~166k | 87.0 tok/s (1,916.2 s) | 76.1 tok/s (2,190.1 s) |
| decode, short prompts | 25.5 tok/s (lab speed gate) | 26.5–32.1 tok/s across the five short gates |
| MTP drafter | enabled | enabled — 361/399 drafts accepted (90%) |

Vision and MTP were not in the recorded run's gate set; they are additional coverage from the
reproduction, not replacements. The recorded run's context-gate decode is not separately
recorded (1,922.9 s total vs 1,916.2 s prefill); the reproduction's long gate decoded at
9.2 tok/s at 166k context.

Prefill ran 13% slower than the recorded run. Reported as measured — single runs on a shared
machine vary with background and thermal state, and nothing in the comparison depends on
matching it.

## Peak attribution (memwatch.jsonl)

`memwatch.jsonl` is the 1 Hz sampler log for the whole session (2,600 samples, GiB). The peak
counter was reset at 2026-10-23T05:22:10Z (`memwatch-reset.timestamp`), immediately before the
long needle request — the first sample after the reset reads `peak_gb: 0.0` — so every
`peak_gb` from 05:22:11Z on is attributable to the long gate alone: **max 20.796 GiB**
(max active 20.68 GiB). Samples before the reset (model load + short gates) peak at 16.22 GiB.
mlx-vlm reports per-request `timings.peak_memory` in decimal GB (22.33 for the long request);
the sampler logs GiB. The two agree once converted.

## Long-gate token math

`probes.py`'s `gate_long` builds `target // 20` filler sentences, so `--long-tokens` is not
the prompt length. `verify_long_tokens.py` replicates the exact prompt construction
(`rng.Random(0)`, same filler lines, same needle position) and counts with the served tokenizer:
`--long-tokens 191540` → 9,577 sentences → **166,671 raw tokens**; the server reported
**166,723** prompt tokens for the request (chat-template overhead included) — 0.03% from the
recorded run's 166,667.

## Environment notes

- Neither the recorded run nor the reproduction raised `iogpu.wired_limit_mb` (sysctl = 0,
  system default); the ~20.8 GiB peak fit the default budget of this 36 GB machine.
- AC power throughout (battery charging, 18% → 56% over the session).
- Host facts in `host.json` (Mac16,6, Apple M4 Max, 14 cores, 36 GB, macOS 27.0.1 build
  26A434, arm64); serial number, hardware UUID, and provisioning UDID intentionally omitted.
- `run-server.sh` and `verify_long_tokens.py` are committed as executed, with the absolute
  home paths of the running account parameterized to `$HOME` form; at run time they resolved
  to the paths used on 2026-10-22 and model-serving behavior is unchanged.

## Files

| file | role | sha256 |
|---|---|---|
| `memwatch.jsonl` | 1 Hz MLX memory log, whole session (GiB) | `46808b4f1062f3db41f114aa6938e274c59bde1bda1e5b038f7e513d334ac5f3` |
| `memwatch-reset.timestamp` | peak-reset boundary (2026-10-23T05:22:10Z) | `f592d2a2acc06a34f175f4a6fb29e6597678e5d2764f663bead5c56e43539f0d` |
| `probe-gates.json` | short gates: chat, reasoning, tools, vision, mtp — 5/5 | `2fb7b89a5b65afb3e6dfef443009225f7bbd87f3b947a6bd3094a8206275312c` |
| `probe-long-166723.json` | long needle gate — pass, 166,723 prompt tokens | `55bfa2624dde84a73b4eec33ef88e3f8f81a758c2c300cd48604596b5dfd7642` |
| `probe-long.log` | long-gate console output + exit code | `61403bef5473f94788c1bd7412812ce114c46b00ed863462ffb736c58c647180` |
| `repro-run-server.log` | reproduction server log (uvicorn/MLX) | `c6ce06af13bd60bac150f589c5dd91d57c0a2a8ff24e85609d83f72a5bdd6fb2` |
| `original-run-server.log` | the recorded run's server log | `323cc58d3c6b0e292c988be07a0090a462ecb417ac2dbfc6d6d503fce7358d03` |
| `original-run-lab.log` | the recorded run's lab gate summary | `fef8ac9d1f67d8edb1df970e75a18a4254c1c81b5aec6de1120686b22a827abe` |
| `host.json` | machine facts (identifiers omitted) | `9df7ecfe661b2ef35e09a099a2cebaddde5d4f0c49b1d78ccebb2b0ace97bf31` |
| `pip-freeze.txt` | pinned reproduction environment | `b2188db87d6ca4b2d49d59d3402869ba14521639a055eaf23cf21bfa19ef04e8` |
| `run-server.sh` | reproduction launch (executed; paths parameterized) | `96d20607e6e7ee668989d0558fe342834aea22878119ffa51557fb7d23580d0d` |
| `memserve_lab.py` | capture wrapper: launcher `__main__` + sampler thread | `9919e686e50ab3c70b4c46f24863eb1a0b34a26d78ce9fda872c9714a9abffd5` |
| `verify_long_tokens.py` | long-gate token-math verification | `0fd26ab00c3dd9f70f4f73321371ffe323a97e000445d17673fb9ddf5a9ffae6` |