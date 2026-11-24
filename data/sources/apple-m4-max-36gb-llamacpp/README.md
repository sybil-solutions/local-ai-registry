# Apple M4 Max 36 GB: owner-machine evidence for llama.cpp (GGUF) recipes

This is owner-machine evidence for the GGUF recipes on `registry/cards/apple/apple-m4-max-36gb.json`.
All of it was produced on the same Mac (M4 Max, 36 GB unified memory, macOS 27.2.1) by
`lab/lab.py try --on endpoint`, which ran against a local `llama-server` started from the frozen launch.

## Shared tooling (this directory)

| file | role |
|---|---|
| `run-server.sh` | starts `llama-server` from a launch's `command` with a 1 Hz memory sampler beside it |
| `memwatch_llamacpp.py` | the 1 Hz sampler: server pid RSS/footprint, `vm_stat` free/wired/compressed, swapouts, and `iogpu.wired_limit_mb`, written as JSON lines. The repo's `memserve.py` monitor is MLX-only. |
| `preflight.sh` | the clean-machine gate run before every test: wired limit = 32768, AC power, no other llama server, heavy apps closed |

## Requirement: raised GPU wired limit

Every recipe here was measured with `sudo sysctl -w iogpu.wired_limit_mb=32768`. This setting
resets on reboot, and the launch keeps `wired_limit_reserve_mb: 4096` (32 GiB of the 36 GiB total).
Each `host-at-start.txt` records the value that was in effect.

## Per-recipe evidence

### `ornith-1.5-35b-a3b.llama.cpp.256k`: `ornith-1.5-35b-a3b/`

- Run record: `data/lab-runs/apple-m4-max-36gb.ornith-1.5-35b-a3b.llama.cpp-metal-ornith-1.5-35b-a3b-q4-k-m-256k.256k.20261118T013724.json`.
  Whole-file sha256 `819b99ba1c97c6e4…`. The payload without its trailing newline hashes to `2a2fd73ec92e1ff2…`,
  which is the recipe's `proof.log`. The file was copied byte-identical from `lab/runs/`.
- Run window: 2026-11-17 18:17:49 to 18:37:40 MST (2026-11-18 01:17 to 01:37 UTC). Covered by `memwatch.jsonl`
  (988 samples), `server.log`, `lab-try.log`, and `host-at-start.txt` (argv equals the launch `command`).
- Gates: load, chat, reasoning, tools, context (213,570 prompt tokens in 1129.7 s), speed. All 6 passed.
  Decode was 50.7 tok/s and prefill was 189 tok/s.
- Memory: server RSS peaked at 22.97 GiB, and swapouts grew by 108,616 pages over the run.
- Weights: `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612…`, `Ornith-1.5-35B-Q4_K_M.gguf`. Local sha256 equals the HF LFS oid at that revision.
- Engine: official ggml-org llama.cpp b11671 macOS arm64 release tarball. Its sha256 is pinned in the launch `install`.

Provenance note: `proof.launch_sha256` binds the launch only. These logs are tied to the run by
their timestamps, which fall inside the run window, and by the argv in `host-at-start.txt`. They are not hash-bound.
