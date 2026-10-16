# RTX Spark N1X 128 GB: vLLM owner runs

This folder holds the evidence for the six `rtx-spark-n1x-128gb` recipes. The registry can't rent this card, so the runs were done on the owner's machine:

- **Card:** NVIDIA RTX Spark N1X. nvidia-smi names it `NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)` and reports its memory as `[N/A]`: 128 GiB unified with the host, 127.6 GiB online, 119.5 GiB `MemTotal`.
- **Host:** aarch64, 20 cores (Cortex-X925), a laptop running a desktop session. Weights sit on btrfs (`compress=zstd:3`).
- **Software:** Linux 7.2.8, NVIDIA driver 615.78.08, Docker 29.8.2 with CDI devices from nvidia-container-toolkit 1.20.1.

## Runs

Each recipe has one proof, from `lab.py try --on endpoint`, with all six gates passing. Its `log` hash matches its record in `data/lab-runs/rtx-spark-n1x-128gb.*`, all from 2026-10-15, 03:46–05:53 UTC.

**qwen3.8-27b-awq-bf16-int4 is borderline on speed, and both its runs are here.** The same eager launch was gated twice. At 03:25 (`…T032530`) it passed every gate except speed, at 14.9 tok/s. At 05:53 (`…T055305`) it passed everything at 15.7 tok/s; that's the run its proof cites. Over the whole answer it decodes about 32 tok/s. The gate's 30-second window from the first token catches the start of the answer, where DFlash accepts about 2 draft tokens per step (6–14 tok/s); later, acceptance rises to 6–7 (30–40 tok/s). The first run was started by hand without the memory log.

Each recipe is the GB10 recipe's settings with two changes:

- **`mem` 0.7 instead of 0.8.** On unified memory, vLLM compares the request with free host memory. The desktop holds about 22 GiB, so 0.8 (95.6 GiB) refuses to start.
- **`--safetensors-load-strategy eager` added to `extra`.** vLLM's default mmap'd loading copied weights into GPU memory at about 60 MB/s here, even with the files in page cache. A 27 GB checkpoint took 26.5 min, past the plugin's 30-minute start deadline. Eager loads the same checkpoint in seconds.

Each launch ran the way the Local AI plugin runs it:

- `docker run --security-opt no-new-privileges --device nvidia.com/gpu=<uuid> --shm-size <launch shm>`
- the plugin's verified weights mounted read-only at `/models`
- the args `lab.py` renders for the recipe

Before each run, the rendered args were checked to equal, argument for argument, the ones the plugin's export serves for the card, plus the eager flag. The gates ran against the engine's bridge address. The plugin's gateway would have cut the 213K-token context gate off at its 600 s upstream timeout.

## Files

- **`memwatch/<model>.csv`:** one per launch, every 5 s from `docker run` until the container was removed.
  - Columns: `t_s` (seconds since `docker run`), `mem_available_mib` (host `MemAvailable`), `gpu_used_mib` (the sum of `nvidia-smi --query-compute-apps=used_memory`, which reports per-process allocation on this iGPU).
  - Host and GPU share the same memory, so the two columns overlap: the engine's allocation is also missing from `MemAvailable`.

## Summary

`MemAvailable` covers the whole machine, including the desktop session.

| recipe | decode tok/s | prefill tok/s | context prompt | weight load | GPU allocation peak (MiB) | host MemAvailable start → min (GiB) |
|---|---|---|---|---|---|---|
| `gemma-4-12b-it-awq-int4.vllm.128k` | 47.7 | 546 | 106258 | 21 s | 81760 | 98.2 → 11.4 |
| `gemma-4-26b-a4b-it-awq-4bit.vllm.256k` | 64.0 | 325 | 213576 | 35 s | 78065 | 97.7 → 11.2 |
| `gemma-4-26b-a4b-it-awq-8bit.vllm.256k` | 52.1 | 327 | 213576 | 41 s | 87989 | 96.9 → 9.6 |
| `qwen3.8-27b-awq-fp8.vllm.256k` | 23.8 | 329 | 213612 | 39 s | 85961 | 97.1 → 11.0 |
| `gemma-4-31b-it-awq-4bit.vllm.256k` | 28.3 | 82 | 213576 | 30 s | 85354 | 97.1 → 8.2 |
| `qwen3.8-27b-awq-bf16-int4.vllm.256k` | 15.7 | 330 | 213612 | 72 s | 87115 | 97.3 → 9.7 |

Load is vLLM's "Model loading took". The plugin's whole start, from `docker run` to the API answering, took 182–288 s for each.
