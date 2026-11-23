# Qwen3.8-27B on Apple silicon (mlx-vlm 0.7.3)

> **Pre-lab evidence (2026-09-25/26).** Measured before the recipe lab existed, with an earlier version of the
> launcher now at `registry/launches/mlx-vlm-qwen3.8-long-prefill.py`. The two memory patches (fused prefill
> attention, segmented prompt embeddings) are unchanged in the current launcher, which adds `/v1/token/encode`,
> a served-model-only `/v1/models` and the model's sampling defaults. The recipes themselves come from
> `lab/lab.py try` runs (see `data/lab-runs/apple-*`). This folder keeps what the lab gates do not cover:
> 200k-token needles, an image plus a 199k-token needle in one request, peak memory, and the dead ends.
>
> Re-checked on the final launcher (`lab-launcher-*`): vision on both builds, image plus needle at ~8k tokens on
> both builds, and on the 3-bit build an image plus a needle in **199,501 tokens in one request, thinking on:
> answer correct, peak 18.79 GiB** of a 24 GB Mac's 20 GiB budget. The probe's answer cap is 4096 tokens there,
> because the final launcher turns thinking on by default.

Evidence for the `qwen3-8-27b-mlx-{4bit,3bit}-mtp-*-mlx-vlm-tp1` recipes. Every run here was on **one
MacBook Pro M5 Max 48 GB** (40-core GPU, macOS 26.6.2), host serve, no container. No 24 GB or 32 GB
Mac was available: those tiers are shown to *fit* by measuring peak GPU memory on the 48 GB machine
and comparing it with the GPU budget the recipe sets, so their recipes stay `candidate`.

## Final results (recipe launch: launcher asset + recipe flags, `--max-kv-size 200000`)

| Tier | Weights | MTP drafter | KV cache | Needle, 199,831-token prompt | Peak GPU memory | GPU budget | Short gates |
|---|---|---|---|---|---|---|---|
| 32 GB | `mlx-community/Qwen3.8-27B-4bit` (4-bit, g64) | `mlx-community/Qwen3.8-27B-MTP-4bit` | 4-bit uniform g64 | found | 21.70 GiB | 28 GiB | chat, reasoning, tools, vision, MTP (251/266 drafts accepted) |
| 24 GB | `leonsarmiento/Qwen3.8-27B-3bit-mlx` (3-bit g64; embeddings/lm_head 4-bit, vision tower 8-bit) | `lukaskremla/Qwen3.8-27B-MTP-3bit-MLX` | 4-bit uniform g64 | found | 18.54 GiB | 20 GiB | chat, reasoning, tools, vision, MTP (245/277) |

- **Peak GPU memory** is `mx.get_peak_memory()` over the whole long request. mlx-vlm reports it in
  decimal GB (`timings.peak_memory`); `memserve.py` logs GiB. The two agree once converted.
- **GPU budget** is `iogpu.wired_limit_mb` from the recipe's launch step: 20480 MiB on 24 GB and
  28672 MiB on 32 GB. macOS wires about two thirds of RAM by default, which is too little.
- **Process overhead:** outside MLX's allocations the server process uses about 0.3 GB
  (`footprint`: 19 GB total, nearly all `IOAccelerator`). A 24 GB Mac therefore keeps ~5 GiB for
  macOS and everything else. Close large applications before a 200k-token request.
- **Vision and 200k in one request (24 GB build):** the synthetic image plus a needle haystack,
  199,473 tokens in total, answered `JESH9682, red, 7394` (all correct) at 18.93 GiB peak
  (`final-24gb-3bit-long-vision-199473.json`). The 32 GB build ran vision on short prompts; its
  200k run is text-only, with 6 GiB of headroom.
- **Speed:** measured while other applications used most of the CPU and swap (a Docker VM at ~490% CPU).
  Treat them as lower bounds for an M5 Max; they do not transfer to a base M5 or M5 Pro.
  Prefill at 200k: 160 tok/s (4-bit), 104 tok/s (3-bit). Acceptance decode: 22.0 / 12.9 tok/s.

## What the launcher changes

A plain `python -m mlx_vlm.server` works for 32 GB (23.97 GiB peak at 200k) but peaks at
**20.97 GiB** for the 3-bit build, over a 24 GB Mac's 20 GiB budget (`history-unpatched-*`).
Two runtime patches fix it:

1. **Fused prefill attention.** The full-attention layers have head_dim 256. MLX picks its unfused SDPA
   for that during prefill, and mlx-vlm's quantized-KV path always computes Q·Kᵀ with `quantized_matmul`.
   Prefill chunks now dequantize the layer's K/V and call `mx.fast.scaled_dot_product_attention(...,
   force_fused=True)`. Decode keeps the quantized path. At 200k on the 3-bit build this cut peak memory
   from 20.97 to 20.35 GiB, and prefill from 30 to 17 minutes; background load differed between the
   two runs (`history-fused-only-*`). The idea comes from PR #105, which forces the fused kernel on an M3 Max.
2. **Segmented prompt embeddings.** The full-prompt `inputs_embeds` (200k × 5120 × bf16 ≈ 2 GB)
   stays resident until prefill ends: the request kwargs hold a reference, and `prompt_step`'s
   `[:, n:]` views keep the buffer alive. The batch now owns it alone and holds it in 8192-token
   segments that are freed as prefill consumes them. At 200k: 20.35 → 18.49 GiB. At 64k:
   14.92 → 14.33 GiB active (`prefill-memory-diagnostics.json`, `3b-fused-64k` vs `3b-fused-seg-64k`).

Other settings:

- `--prefill-step-size 128`: at the default 2048, the unpatched 4-bit build reached 23.2 GiB active by
  43k tokens and swapped, so the run was aborted (~45 GB extrapolated to 200k). At 32k, 128 costs no
  prefill speed.
- `MLX_VLM_TOKEN_QUEUE_TIMEOUT=7200`: a 200k prefill outlasts the default 600 s wait for the first token.

## Tried and not used

- **3-bit KV cache:** mlx-vlm 0.7.3 sizes packed KV buffers as `dim // (32 // bits)`, which is wrong
  for 3 bits (25 vs 24 words). With that patched it passed the needle but saved nothing at 200k
  (`history-fused-kv3-*`).
- **TurboQuant KV:** no memory or speed advantage over uniform at 32k.
- **Quantizing the last full-attention layer (kept bf16 by mlx-vlm):** breaks the MTP drafter, which
  shares that layer's cache.
- **`lukaskremla/Qwen3.8-27B-2bit-MLX`:** refused a plain tool call and misread the image digits.
- **`lukaskremla/Qwen3.8-27B-3bit-MLX` (vision tower also 3-bit):** misread 7394 as 7304.

## Reproduce

    python -m venv .venv && .venv/bin/pip install mlx-vlm==0.7.3 huggingface_hub
    MEMLOG=mem.jsonl ./serve.sh 24gb           # or 32gb; pins both repos by commit
    python probes.py --gates chat,reasoning,tools,vision,mtp --out gates.json
    python probes.py --gates long --long-tokens 229700 --out long.json   # 199,831-token needle
