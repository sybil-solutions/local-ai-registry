# Inkling 2 Mini on Apple M1 Max 32 GB: mlx-serve 26.11.4, MLX 4-bit, q8 KV, 128K

Machine: MacBook Pro (MacBookPro18,2), Apple M1 Max (32-core GPU), 32 GB unified memory, macOS 26.6.2. No sysctl changes, no sudo. One model server at a time.

## Pins

| Part | Pin |
|---|---|
| Engine | [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) v26.11.4 = `45df50e9adf7975bf31ee198c7440ec511c6162e`, release asset `mlx-serve-bin-macos-arm64.tar.gz` sha256 `2079e8649b4a25aed267bf43117ef23b026c833453f0cdd0eff2d328aa1e0832` (matches the GitHub release digest; binary reports `mlx-serve 26.11.4 (fast)`, MLX 0.32.5) |
| Weights | `mlx-community/Inkling-2-Mini-4bit@9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62`, every LFS file sha256-checked against the Hub |
| Launch | `registry/launches/mlx-serve-inkling-2-mini-4bit-kv8-128k.json` |

## Results

Same `ctxbench` prompts and lengths as the llama.cpp recipe (#466): a needle in filler text, temperature 0, 512 generated tokens, medians of 3 (64K and 128K: 1 run). Prefill and decode are the server's own `timings`.

| Setup | Decode tok/s (short / 8K / 25K / 64K / 128K) | Prefill tok/s (8K / 25K / 64K / 128K) | TTFT 8K | Peak system memory |
|---|---|---|---|---|
| **mlx-serve, q8 KV (this recipe)** | **46.3 / 39.8 / 36.6 / 31.5 / 25.2** | **387 / 367 / 321 / 265** | **21.9 s** | 28.8 GiB (server RSS 15.6 GiB) |
| mlx-serve, f16 KV | 49.9 / 43.2 / 38.7 / 31.3 / refused | 389 / 370 / 325 / — | 21.8 s | 28.0 GiB (server RSS 15.3 GiB) |
| llama.cpp UD-Q4_K_XL, q8 KV (#466) | 20.3 / 19.3 / 18.0 / 15.7 (66K) / — | 186 / 146 / — / — | 45.6 s | 29.3 GiB |

Every needle answer at 8K, 25K, 64K and 128K was correct. Swap stayed at the 428 MB it held before the runs.

- **128K:** with an f16 KV cache, mlx-serve's admission check refused the 131,049-token prompt (needs 7,031 MB, 5,654 MB free). With `--kv-quant 8` it ran to the end. That run used `--ctx-size 136000` so that the 131,050-token prompt plus 512 output tokens fit. The recipe keeps the registry's 131072 window; its lab context gate (91,888 tokens) passed at that window.
- **Draft decoding:** none on this machine. The Inkling plugin loads its bundled draft head or a DFlash2 drafter only on Apple9+ GPUs (M3 and later): `draftsOn` in `src/plugins/inkling2/plugin.zig` says that "on an M1 a few-row verify costs so much more than a step that drafted decode ran under serial." With `mtp.safetensors` (from `thinkingmachines/Inkling-2-Mini@c9d028302cd0f035fdf8e7e91f089a4ce94d2195`) or `incoai/Inkling-2-Mini-DFlash2@4f9088ea66c21f2b99ed657d1941c306b03e84d2` in the pack, the server logs `spec=disabled (inkling2_loop serves serial-only)`. Prompt-lookup decoding is also forced off for this architecture, so `--no-pld` changes nothing; it is passed to make that explicit.
- **Absolute model path:** mlx-serve 26.11.4 opens a pack only by absolute path (`FsPackDir.open` refuses a relative one, and the server stops with `PluginDeclinedNoLegacy`). The launch's two-line launcher passes `$PWD/models/...`.

## Quality

Six fixed prompts (intervals, bug fix, SQL, counting, scheduling, trace) on the recipe's server: 6/6 PASS, all `finish=stop`. See `evidence/quality.jsonl`.

## Evidence

- `evidence/ctxbench-q8kv.jsonl`, `evidence/ctxbench-f16kv.jsonl`: one line per request, with the server's `timings`.
- `evidence/memwatch-*.jsonl`: system memory, swap and server RSS once per second.
- `evidence/server-*.log`: the server's logs for each run.
- `evidence/lab-try.log`: the `lab.py try --on endpoint` output for the recipe's launch.
