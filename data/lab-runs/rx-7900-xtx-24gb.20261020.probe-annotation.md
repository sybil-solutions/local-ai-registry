Probe and memwatch notes for this run (2026-10-20, rx-7900-xtx-24gb)
=============================================================

`probe.json` is the raw output of the kit's `tools/probe.py`, taken before the server started (with
`storage.path` shortened to `~/models`; the models folder was in a scratch directory on the same disk), and
`memwatch.csv` covers the whole run. Two kit tools needed a workaround on this machine; neither changes what
was measured.

probe.py and an installed nvidia-smi
------------------------------------
This host has the `nvidia-smi` binary installed but no NVIDIA driver loaded. `probe.py:gpus()` calls
`nvidia-smi --query-gpu=...` and splits each output line into eight fields. Without a driver, the command
prints a one-line error ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver") on
stdout, and the unpack raises `ValueError: not enough values to unpack (expected 8, got 1)` before the AMD
branch is reached.

The probe was run unmodified, with a stub `nvidia-smi` (`#!/bin/sh` / `exit 9`, no output) first on PATH and
`LC_ALL=C`. With that, `gpus()` falls through to its `rocm-smi --json` branch, which reports both AMD
devices: card0 is the RX 7900 XTX (gfx1100, 25,753,026,560 B VRAM) and card1 is the Ryzen 7 7800X3D's iGPU
(gfx1036, 512 MiB). `HIP_VISIBLE_DEVICES=0` keeps the iGPU away from the server.

memwatch on AMD
---------------
The kit's `tools/memwatch.sh` fills `gpu_used_mib` from `nvidia-smi`, which is empty on AMD.
`memwatch.csv` comes from `tools/memwatch-amd.sh` in the recipe repo (kamm3r/exl3-rdna3-recipe@1fb72bd).
It has the same first four columns, computed the same way. `gpu_used_mib` and the added `gpu_total_mib` are
read once a second from amdgpu's `mem_info_vram_used` and `mem_info_vram_total` for the RX 7900 XTX
(`/sys/class/drm/card1/device`, PCI device 0x744c; the DRM card numbers differ from rocm-smi's). These are whole-card figures, so the desktop's own VRAM
(about 520 MiB at the start) is included in the 21,134 MiB peak.
