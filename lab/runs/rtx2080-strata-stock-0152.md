# Swift 1.5 Flash-Next on RTX 2080: official Strata 0.1.52

This recipe uses the unmodified official CUDA 13 engine from Strata v0.2.7
and the unmodified v0.2.7 Python server. It does not use the owner's custom
Strata executable, private CUDA patches, hot-session patches, or routing prior.
The configuration is specific to one headless 8 GB RTX 2080 at a 165 W cap.
The host is an i5-9600K with six cores and 64 GB DDR4-3200.

## Pinned inputs and preparation

Clone `Niko1221/Strata` at `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`.
Run the accompanying `rtx2080-strata-stock-0152-setup.sh` from that checkout.
It installs the checkout's pinned requirements, verifies the official engine
archive and executable, downloads and checks the exact Swift shards, and uses
only the pinned upstream model/MTP preparation tools. Write the JSON config
from the rendered launch, then start its command from the Strata checkout.
The script is syntax-checked; the hardware run reused the owner's separately
hash-verified pinned input files and ran the same upstream preparation tools
in an isolated directory rather than downloading the large inputs twice.

The official engine archive is
`strata-engine-0.1.52-linux-x86_64-cuda-cuda13.tar.gz`, SHA256
`4177c1a8ffc59bf66074093b22ac55ed477bcbe84f8bd12e378b0f1236da55c7`.
The executable is SHA256
`9d415609c6abf40352345e20c75fc2205db0bc7acb804a75d12ab4ac10ee16ad`.
Its release BUILD.json reports source `f-027-rel a42cc3d7df70`, CUDA 13.0,
and architectures 75, 86, 89, 120. The server source is the v0.2.7 commit above.
No source compilation or custom binary is required. The official component requires NVIDIA driver 580.65 or newer; this host uses 615.78.08.

Swift weights are pinned to
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF@99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`:

- Shard 1: 39,788,473,344 bytes, SHA256
  `ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b`.
- Shard 2: 28,363,693,824 bytes, SHA256
  `e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb`.

The MTP tensors are fetched by upstream byte ranges from original Qwen revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and checked against the upstream
per-tensor SHA256 table. MTP Q2_0 and runtime packs are regenerated with the
unmodified tools. `gguf-py` comes from llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, the version pinned by Strata setup.

## Runtime choices and host requirements

128K context; int8 KV with 32K resident on the GPU; MTP four drafts; adaptive
expert cache; PCIe fraction 0.35; 512-token prefill chunks. Conversation
parking is disabled to avoid reserving RAM for unrelated sessions. The config sets upstream SM75, fused-prefill and KV-prefetch switches;
actual kernel selection remains the upstream runtime's.
`STRATA_ROUTE_TAIL_SKIP=0` retains all selected experts; `STRATA_ROUTE_PRIOR=0` disables the upstream small-card residency bias. No vision is claimed by this text-only launch.

`needs.host_ram_gb` declares 48 GB of available RAM, not 48 GB installed RAM.
The run samples MemAvailable every 500 ms starting before model launch through
all acceptance gates; the measurements below record the largest decline and
safety headroom. `disk_gb: 90` includes conservative setup space for weights, generated
packs, MTP source/packed/runtime files and the official engine. Models and PLE
must be on local NVMe, since PLE is mapped during serving.

Host programs are supported in the registry catalog/SDK but currently excluded
from the Omarchy plugin export. This PR does not change that export policy.
The SDK's host install field is an instruction, so follow the setup script;
it does not automatically prepare a Strata installation.

## Acceptance

The normal custom Strata service is stopped for the attempt and restored after
it. Upstream `lab/lab.py try --on endpoint` runs all gates unchanged with no
answer or thinking caps. The lab writes the recipe only after every gate passes.
The passing acceptance proof accompanies these notes; host-memory results
are recorded below.

A first startup with 4096-token chunks and auto expert-cache sizing left no cache slots on 8 GB. The accepted candidate uses 512-token chunks to reserve room for the expert cache without custom memory-loan patches.

The upstream verifier needs a profile-filled cache. This launch uses `data/expert-profile.bin` from the pinned source tree rather than an owner-learned profile.

All six gates passed on 2026-10-29 UTC. Lab decode: **40.4 tok/s**. The
context gate recalled the correct access code from **106,294 prompt tokens**
in **329.8 seconds** including generation (the lab prefill field is 322 tok/s,
not an engine-only rate). The streamed speed answer ran to its natural end.

The before-start MemAvailable was **59,814,916,096 bytes** and the minimum
through startup and all gates was **19,494,662,144 bytes**. The largest decline
was **40.320253952 GB**, leaving **7.68 GB** of headroom under the declared
48 GB available-RAM requirement. The full 500 ms trace is retained locally; these notes preserve the measured
baseline, minimum and headroom. The owner's usual Strata service was restored afterward.

The model metadata date uses the original fine-tune repository's first-publication timestamp (2026-09-22); the pinned Swift 1.5 GGUF repository was created 2026-09-24. Source: https://huggingface.co/api/models/ukisai/Swift-Qwen3.8-Flash-Next.

Storage check: both Swift GGUF shards are regular files on `/dev/mapper/omarchy_root[/@home]` (Btrfs), backed by `nvme0n1p2` / `nvme0n1` with NVMe transport and rotation flag 0.
