Probe annotation for this run (2026-10-12, rx-9070-xt-16gb)
==========================================================

`probe-20261012.json` is the raw output of the kit's `tools/probe.py`, kept exactly as
written. Read it with this in mind: its `gpus` array is empty, and that is not a
measurement failure.

Why it is empty
---------------
`tools/probe.py:gpus()` detects an AMD card in exactly one way, the `rocm-smi --json`
branch at the end of the function. This machine runs the card through the `amdgpu`
kernel driver and RADV with no ROCm installed, so `rocm-smi` is not present and the
function returns no GPU. The NVIDIA and Intel branches do not apply either.

What the same machine reports through sysfs
-------------------------------------------
These values were read directly from `/sys/class/drm` and `lspci` after the probe, and
are the facts the empty array stands for:

  card                      /sys/class/drm/card1
  vendor                    0x1002 (Advanced Micro Devices)
  device                    0x7550
  lspci                     Advanced Micro Devices, Inc. [AMD/ATI] Navi 48
                            [Radeon RX 9070/9070 XT/9070 GRE] (rev c0)
  pci                       0000:07:00.0
  kernel driver             amdgpu
  vram_total                16,304 MiB (15.92 GiB)
  vram_used at the probe     459 MiB
  vulkan                    present; no ROCm, no /dev/kfd used by this recipe

That the card is what the recipe claims is corroborated twice over: the engine is the
`amd-vulkan` backend, which reaches the GPU only through `/dev/dri`, and the six gates
ran against a server that allocated 15.53 GiB on this device (see the memwatch CSV).

The rest of the probe file is unaffected and should be read as-is:

  host    kernel 7.2.9-2-cachyos, 12 CPUs, no AVX-512
  ram     33.59 GB total, 26.7 GB available at probe time, 0 swap used
  storage /mnt/raid0/GGUF, ext4 on /dev/md127, 124.98 GB free, not NVMe (two SATA SSDs)

The CPU name field is empty in the raw file for a second, unrelated reason. `probe.py`
reads `lscpu` without forcing a locale, so on this pt_BR host it printed translated
labels and the `Model name:` regex matched nothing. Under `LC_ALL=C` the same command
reports `AMD Ryzen 5 5600 6-Core Processor`, which is the CPU that ran this.
