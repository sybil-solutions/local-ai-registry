# Inkling-2-Mini on an Apple M4 Max 64 GB (mlx-lm, 256K)

Evidence for `registry/recipes/apple/apple-m4-max-64gb/inkling-2-mini.mlx-lm.256k.json`. Every run here was on one
MacBook Pro, Apple M4 Max (16-core CPU, 40-core GPU), 64 GB, macOS 26.7.1, on AC power, host serve (no container).

Inkling-2-Mini: 24 layers, of which layers 6–17 run three times per token (48 layer visits), 128 experts (top 8),
17.3 GB as `mlx-community/Inkling-2-Mini-4bit`.

## The launch

- **Engine:** `mlx_lm.server` from ml-explore/mlx-lm#2086 at commit `0c4c99192230f1943dd468d297f0b8a64ae7b7c2`
  (Inkling 2 Mini support; not in a release yet), mlx 0.32.4. Installed from the upstream repository URL.
- **Launcher** (`registry/launches/mlx-lm-inkling2-serve.py`):
  - Inkling 2's response format, which mlx-lm does not know yet: `<|content_thinking|>` → `reasoning`,
    `<|content_text|>` → `content`, `<|content_invoke_tool_json|>` → OpenAI tool calls. Without it the reasoning and
    tools gates fail.
  - `/v1/token/encode`, a served-model-only `/v1/models` and the model's sampling defaults, as in the LFM2.5 launcher.
  - `INKLING_FAST`: decode-only `mx.compile` fusion of the small elementwise chains in each layer visit (message
    gates, decode attention mask, MoE combine, short convolutions, attention pre/post). Prefill runs mlx-lm's own
    code unchanged.
- **Window:** 262,144 tokens, one request at a time (`--decode-concurrency 1 --prompt-concurrency 1
  --prompt-cache-size 1`). KV is 32 KiB/token (8 full-attention visits); the other 40 visits use a 1,024-token window.

## Results

### Registry lab (`lab/lab.py try --on endpoint`), on a clean build of the recipe launch

`data/lab-runs/apple-m4-max-64gb.inkling-2-mini.mlx-lm-inkling-2-mini-4bit-256k.256k.20261121T014811.json`. The
build is a fresh venv from the launch's `pip` list only, the launcher written from the rendered launch (sha256
`e04fd3dd…`), weights at the pinned revision (`evidence/clean-build.log`, `evidence/clean-build-pip-freeze.txt`).

| Gate | Result |
|---|---|
| load | `/v1/models` lists the served model |
| chat | `red, yellow, blue`, finish `stop` |
| reasoning | thinking separate (127 chars), answer `391` |
| tools | `get_weather(city="Paris")`, then the reply quotes the tool's code |
| context | `58213` from a 184,730-token prompt (70 % of 262,144; the gate's filler is denser in Inkling's tokenizer than the gate assumes) in 449 s |
| speed | **40.2 tok/s**, 1,208 tokens in the first 30 s, counted with the server's tokenizer |

Prefill (context gate): 411 tok/s. Peak footprint during the run: 34.0 GiB (`evidence/lab-run-host-memory.jsonl`, 5 s samples).

**Clean build = hill-climb best.** Back to back with `bench.py` (n=6): decode 39.98 / **39.53** / 40.80 tok/s
(hill-climb / clean / hill-climb). A further interleaved run of the clean build with and without the recipe's
one-request flags: decode 40.7–41.2, prefill 765–828 tok/s at 3.3K: the flags change nothing, and the clean build's
prefill matches the hill-climb range (730–825). The single-request prefill cells vary about ±10 % run to run.

### Hill-climb (`bench.py` from sybil-solutions/inference-optimisation-skill, medians of 6, temperature 1.0, thinking on)

Each candidate ran between two controls; accept = beats both, nothing regresses > 3 %.

| Metric | Control | Control | **Recipe launch fast paths** | Ceiling |
|---|---:|---:|---:|---:|
| Decode, 1 request (tok/s) | 33.76 | 35.20 | **40.35** (40.1–40.7) | ~146 at 4K |
| Decode, 4 requests, aggregate | 91.0 | 89.5 | 94.3 | — |
| Prefill, 3,302-token prompt | 839 | 779 | 816 | 753 (50 % MFU) |
| Prefill, 25,554-token prompt | 792 | 751 | 775 | 659 |
| TTFT, short prompt (ms) | 227 | 222 | 207 | — |

Long context, recipe fast paths, one cold request each:

| Prompt | Needles | Prefill | Decode at that context | Peak footprint |
|---:|---:|---:|---:|---:|
| 102,426 tokens | 3/3 | 535 tok/s | 27.7 tok/s | 28.6 GiB |
| 242,039 tokens (92 % of 256K) | 3/3 | 330 tok/s | 20.8 tok/s | 40.4 GiB |

Peak footprint is the process's `footprint` (includes the Metal / IOAccelerator residency), sampled every 2 s.
40.4 GiB leaves ~20 GB of a 64 GB Mac for macOS: one 256K request at a time, not two.

### Ceilings (`THEORY.md`)

Measured 467 GB/s read bandwidth and 14.8 TFLOPS bf16 (MLX, this machine). Decode reads ~3.06 GB per token
(experts 1.84, attention and gates 0.96, lm_head 0.26) → ~146 tok/s at 4K. The recipe reaches ~28 % of it: each of
the ~3,300 dependent Metal kernels per token costs ~7 µs of latency, while the expert GEMVs already run at
bandwidth. Prefill is at its practical ceiling at every length measured.

## What the fast paths change, and how they were checked

| Patch | Effect alone | Numerics |
|---|---|---|
| gates, mask, moe, conv, attn (compiled elementwise chains) | 28.6 → 25.7 ms/token | bit-identical logits; 144/144 greedy tokens equal |
| gates3 (q/k/v message gates batched; forget+message projections in one matmul) | +6.5 % | KL 1.5e-4 on one step; greedy tokens equal |
| Reference: same unpatched model, prefill step 512 vs 256 | — | KL 4.4e-3, max \|Δlogit\| 2.66 |

The first version also fused prefill and cost ~11 % prefill (rejected); the fast paths are decode-only since.
Rejected (in `PLAN.md`): fusing the q/gate/r/sink projections into one GEMV (no gain, perturbs numerics), and
custom Metal kernels for the message gates (slower in the model than the compiled version).

## Files

- `PLAN.md` — every experiment in order, with the numbers and why it was kept or rejected.
- `THEORY.md` — hardware measurements and the decode / prefill / KV ceilings.
- `ledger.jsonl` — every `bench.py` run.
- `tools/` — the scripts behind the numbers (`hwbench.py`, `stepbench.py`, `ablate.py`, `chainbench.py`, ...).
- `evidence/` — the clean-build log and `pip freeze`, the server log and memory samples of the lab run, the
  clean-build vs hill-climb A/B.
