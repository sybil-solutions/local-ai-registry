# Intel Arc Pro B65 32 GB: llama.cpp SYCL owner runs

This folder holds the evidence for the nine `intel-arc-pro-b65-32gb` recipes. The registry can't rent this card, so the runs were done on the owner's machine:

- **Card:** ASRock Arc Pro B65 Creator 32 GB, PCI id `8086:e222` (`lspci`: `Intel Corporation Battlemage G31 [Arc Pro B65]`)
- **Host:** Ryzen 7 5700X, 50 GB RAM, PCIe 4.0 x4
- **Software:** Linux 7.2.5, Docker 29.7.2

The machine also has an RTX 3080. The runs didn't use it.

## Runs

Each recipe has two proofs, both from `lab.py try --on endpoint` against the frozen launch, all six gates passing. Their `log` hashes match the 18 records in `data/lab-runs/intel-arc-pro-b65-32gb.*`.

1. **2026-10-14, 18:27–20:39 UTC:** 32K at 18:27–18:35, then 64K and 128K at 19:57–20:39. These had no memory logging.
2. **2026-10-14, 22:46–23:48 UTC:** all nine launches again, with the logging below. Record names end `T2246…` to `T2348…`.

Every launch was started the way the SDK renders it, `docker run --device /dev/dri …`, published on 127.0.0.1 only. The pinned GGUF was checked by sha256 against the Hugging Face LFS oid before each launch.

## Files

- **`probe.json`:** the recipe kit's `tools/probe.py --models-dir ~/.lmstudio/models`, at kit commit `37930974`, taken at the start of run 2. The kit's GPU list includes the RTX 3080, which wasn't used.
- **`memwatch/<model>.<ctx>.csv`:** the kit's `tools/memwatch.sh`, unmodified, one per launch.
  - Started before `docker run` and stopped after the container was removed.
  - Columns: `t, mem_available_gb, swap_used_gb, gpu_used_mib`.
  - **Its GPU column isn't the B65.** It sums `nvidia-smi`, which on this machine is the idle RTX 3080.
- **`b65-vram/<model>.<ctx>.csv`:** B65 memory, once a second, for as long as the container ran.
  - Columns: `t, b65_vram_resident_mib`.
  - It's the server's `drm-resident-vram0` from the xe driver's DRM fdinfo, summed over unique `drm-client-id`. It's read inside the container with `docker exec … cat /proc/[0-9]*/fdinfo/*`.
  - The card reports 32,656 MiB to SYCL.

## Run 2 summary

Host `MemAvailable` covers the whole machine, which was a desktop session with about 14.5 GB of unrelated swap in use. Use it as an upper bound on what the server added, not as an exact need.

| recipe | decode tok/s | prefill tok/s | context prompt | B65 VRAM peak (MiB) | host MemAvailable start → min (GB) |
|---|---|---|---|---|---|
| `qwen3.8-27b.llama.cpp.32k` | 23.6 | 617 | 25,782 | 18,496 | 39.3 → 34.2 |
| `gemma-4-31b.llama.cpp.32k` | 21.4 | 372 | 25,746 | 23,926 | 39.4 → 31.5 |
| `muse-glimmer-30b.llama.cpp.32k` | 25.4 | 607 | 21,127 | 17,826 | 39.4 → 37.3 |
| `qwen3.8-27b.llama.cpp.64k` | 23.6 | 664 | 52,625 | 20,713 | 39.5 → 32.2 |
| `gemma-4-31b.llama.cpp.64k` | 21.4 | 378 | 52,589 | 26,614 | 39.5 → 29.1 |
| `muse-glimmer-30b.llama.cpp.64k` | 25.1 | 263 | 43,186 | 17,857 | 39.5 → 36.9 |
| `qwen3.8-27b.llama.cpp.128k` | 23.6 | 575 | 106,294 | 24,906 | 39.4 → 29.0 |
| `gemma-4-31b.llama.cpp.128k` | 21.4 | 326 | 106,258 | 31,990 | 39.5 → 35.3 |
| `muse-glimmer-30b.llama.cpp.128k` | 25.1 | 667 | 87,285 | 19,458 | 39.5 → 36.1 |

- **Gemma 4 31B at 128K is tight.** It peaks at 31,990 of 32,656 MiB, leaving about 650 MiB free. It passed with nothing else on the card. If the B65 also drives a display, use the 64K recipe.
- **The host-RAM drop wasn't broken down by process.** llama-server keeps a host prompt cache (`--cache-ram`, 8 GiB by default), which may account for much of it. The weights are memory-mapped from the read-only `/models` mount. The 128K Gemma log shows its prompt state at 10.7 GB, over that limit, so the server skipped caching it (`… exceeds cache size limit 8192.000 MiB, skipping`), and that run's drop is the smallest of the Gemma runs.
- **Prefill is the registry's context-gate figure:** the prompt divided by the whole request time, thinking included. So it moves with how long the model thinks (Muse: 263 tok/s at 64K, 667 at 128K).
