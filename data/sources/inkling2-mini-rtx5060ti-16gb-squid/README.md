# Inkling-2-Mini on an RTX 5060 Ti 16 GB (`squid5060ti`, NVFP4 + Q4 KV, 128K)

Evidence for `registry/recipes/nvidia/rtx-5060-ti-16gb/inkling-2-mini.squid5060ti.128k.json`. Every run here was on one desktop PC with an **NVIDIA GeForce RTX 5060 Ti 16 GB** (`sm_120a`, 36 SMs, 32 MiB L2, driver 621.01), Windows 11 + WSL2 Ubuntu 26.04, CUDA 13.3, Zig 0.17.1.

## The launch

- **Engine:** [`omerdikyol/squid5060ti`](https://github.com/omerdikyol/squid5060ti) pinned at commit `5945ebb53aedde8a77bec657509a9a51a35fa470` (16 GB zero-spill fork of [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090), zero Python at runtime).
- **Weights:** `thinkingmachines/Inkling-2-Mini@7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` (26.4B looped MoE, 3.4B active, 48 logical visits, DSpark MTP).
- **16 GB Blackwell adaptations vs upstream `squid5090`:**
  1. Host-pinned `embed.weight` (`-0.86 GiB` VRAM) and `markov_w1` (`-98 MiB` VRAM) via `cudaHostAllocMapped`.
  2. Zero-copy host-pinned load staging (`d_stage`, `d_fold_bf16`) eliminating the `879 MiB` load-time VRAM spike.
  3. `prefill_batch_max = 128`, `prefill_local_attn_sub = 128`, and 16K cap on DSpark's single global KV layer so full **131,072-token (128K) Q4 KV** fits in `14.97 GiB` engine VRAM (`13.59 GiB` weights + `1.38 GiB` runtime).
  4. Default `--spec-threshold 0.90` tuned for 376.4 GB/s GDDR7 via interleaved A/B controls.
  5. Zero-malloc `POST /v1/token/encode` & `/tokenize` endpoints using the native Zig BPE tokenizer for exact server token counting.

## Results

### 1. Official Registry Lab (`lab/lab.py try --on endpoint`, clean build at `5945ebb`, 128K context)

Evidence: `lab/runs/rtx-5060-ti-16gb.inkling-2-mini.squid5060ti-inkling-2-mini-128k.128k.20261124T104530.json`.

| Gate | Result |
|---|---|
| `load` | PASS (`/v1/models` lists `inkling-2-mini`) |
| `chat` | PASS (`red, yellow, blue`, finish `stop`) |
| `reasoning` | PASS (thinking separated from content) |
| `tools` | PASS (`get_weather(city="Paris")` → reply quotes tool code) |
| `context` | PASS (`58213` exact needle recall from **91,888-token** prompt; **920 tok/s** cold prefill) |
| `speed` | PASS (**107.4 tok/s** first 30 s, exact server tokenizer via `/v1/token/encode`) |

Peak VRAM across 27 five-second `nvidia-smi` samples (`evidence/vram-samples.jsonl`): **15,941 / 16,311 MiB** (including ~590 MiB Windows WDDM desktop).

### 2. Hill-climb (`bench.py` from `sybil-solutions/inference-optimisation-skill`, medians of 6, `temp=0.6`, thinking on)

*(Measured with `prefill_batch_max = 512` at 48K–64K context during hill-climbing; `bench.py` numbers and `lab/lab.py` gate numbers come from different harnesses and are reported separately.)*

| Metric | `llama.cpp` Baseline (GGUF) | `squid5060ti` (NVFP4 + Q4 KV) | Ceiling (measured 376.4 GB/s, 51.1 TFLOPS) |
|---|---:|---:|---:|
| c1 decode (median of 6) | `45.61 tok/s` | **125.9–132.6 tok/s** *(code: 151–174)* | `160–200 tok/s` |
| c1 decode (`--mtp 0`, greedy) | — | **110.4–111.0 tok/s** | `133 tok/s` (83.0%) |
| Cold prefill ~4K (3,300 tok) | `411.0 tok/s` | **1,629–1,755 tok/s** | `3,649 tok/s` |
| Cold prefill ~16K (12,898 tok) | `362.5 tok/s` | **1,950–2,061 tok/s** | `3,359 tok/s` (61.4%) |
| Short TTFT | `254.6 ms` | **78.0–82.9 ms** | — |

See `THEORY.md` for hardware measurements and ceilings, `PLAN.md` for all accepted and rejected levers (32K pruned draft vocab, 36-SM `LM_HEAD_BLOCKS=144`, `KV_SPLIT_K=16`), and `ledger.jsonl` for raw runs.
