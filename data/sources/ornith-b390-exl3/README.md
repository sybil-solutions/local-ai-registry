# Ornith 1.5 EXL3 on Intel Arc B390 / 64 GB unified memory

Ornith-1.5-35B-A3B, EXL3 4 bpw, FP8 E4M3 KV, 262,144-token context,
one NEXTN/MTP speculative draft, 1,024-token prefill chunks, image input,
Qwen reasoning/tool parsers and prefix caching. One concurrent request.

Test machine: Dell XPS 14 DA14260, Core Ultra X7 358H, Intel Arc B390 (Xe3),
64 GB LPDDR5X-9600. CPU and GPU share the same RAM; capacities are not additive.
Linux xe driver; Docker with render-node passthrough, ordinary bridge networking
and private 2 GiB shared memory. No privileged container or host IPC.

## Reproduce

The launch's `build.commit` pins this registry's source tree. From that checkout:

```bash
docker build -f data/sources/ornith-b390-exl3/Dockerfile -t ornith-b390-recipe .
```

Use `python3 lab/lab.py render <recipe>` or the SDK for the pinned weight
download, environment, arguments and launch command. This is a source-build
recipe, supported by the registry and SDK. The plugin does not export build
recipes; publishing an attested OCI image and rerunning acceptance on that
digest is required before plugin distribution.

The Dockerfile fetches trellis-serve at
`72fd2d4ca8e8de3fce03ab09e3ed1692957c2f8c`, applies the included patch and
compiles both SYCL libraries against the pinned SGLang XPU base's Torch 2.13.
The patch adds K4 grouped MoE decode and XMX/DPAS prefill, handles even-K
expert layout, limits K4 GEMV MR to 2 on B390 and samples speculative target
distributions on CPU to avoid an extremely slow first XPU top-k call.
The included independent CPU-reference kernel checks passed all 24 cases.

Weights: `ultimatechris/Ornith-1.5-35B-A3B-EXL3-4bpw` at
`6507c778b466b78eaaad0d7d552f98625bd97881`. All routed experts are K4;
the vision encoder remains BF16 and MTP weights are present.

## Memory and operational limits

The 264,192-token pool allocates 2,705,326,080 bytes of target KV and
270,532,608 bytes of MTP KV: 2.771484375 GiB combined, versus 5.54296875 GiB
for the equivalent BF16 pool. Recurrent state, weights and runtime buffers
are separate. Driver-owned XPU allocations are not fully reflected in the
container's cgroup memory. Process PSS plus unique DRM residency was about
31.1 GiB during the original sweep; this is an approximate combined footprint.

All context tokens, including image tokens and generated output, share the
262,144-token limit. Long cold prefill is slow. Prefix caching avoids repeated
work for matching document prefixes, but does not make a new document cheap.
Image input is tested separately from the maximum-length text test.

A forced container stop during an earlier long prefill caused an xe GPU fault.
Compute recovered after closing the affected processes; no desktop reset or
reboot was used. Let requests finish before stopping/restarting. For cancellation,
use `/abort_request` with `{"abort_all":true}` then POST `/flush_cache?timeout=30`
and wait for it to drain; do not kill an active model worker.

## Evidence

The recipe is generated only by `lab/lab.py try --on endpoint`, using its
unmodified load/chat/reasoning/tools/context/speed gates. Lab responses are
uncapped and run to EOS. The Lab run record is retained here after acceptance.
Original exploratory tuning records, if included, are separate from the Lab
proof: synthetic repeated-word documents, three retrieval markers, thinking
disabled and 256 output tokens per cell. They do not establish general model
quality or controlled FP8-versus-BF16 speed differences.
