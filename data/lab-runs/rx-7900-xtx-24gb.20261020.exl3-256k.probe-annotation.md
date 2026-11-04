Probe and memwatch notes for the two 256K EXL3 runs (2026-10-20, rx-7900-xtx-24gb)
===================================================================================

Each run has its own files, `rx-7900-xtx-24gb.20261020.exl3-256k-<4.0|3.0>bpw.probe.json` and `.memwatch.csv`. The probe
is the raw output of the kit's `tools/probe.py`, taken before the server started (with `storage.path` shortened to
`~/models`; the models folder was in a scratch directory on the same disk), and the memwatch covers the whole run.
Two kit tools needed a workaround on this machine; neither changes what was measured.

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
`memwatch.csv` comes from `tools/memwatch-amd.sh` in the recipe repo (kamm3r/exl3-rdna3-recipe@7b10053).
It has the same first four columns, computed the same way. `gpu_used_mib` and the added `gpu_total_mib` are
read once a second from amdgpu's `mem_info_vram_used` and `mem_info_vram_total` for the RX 7900 XTX
(`/sys/class/drm/card1/device`, PCI device 0x744c; the DRM card numbers differ from rocm-smi's). These are whole-card figures, so the desktop's own VRAM is included in the peaks: 22,118 MiB for 4.0 bpw (937 MiB in
use at the start) and 21,638 MiB for 3.0 bpw (553 MiB at the start).
