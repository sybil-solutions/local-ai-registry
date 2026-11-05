# RTX 2080 Ti host launch: Qwen3.5-9B EXL3, Q4 KV

Qwen3.5-9B EXL3 4.00bpw (TheMelonGod/Qwen3.5-9B-exl3 @22ef1303, the registry's
own reference weight for this model) on one stock RTX 2080 Ti (11264 MiB,
compute cap 7.5), Q4 KV cache, 65536 context tokens per request, batch 2, a
133120-token pool. TabbyAPI @01f939c with ExLlamaV3 1.6.6, Torch 2.8.0+cu128,
Triton 3.4.0, Python 3.12. The card file matches vramGb 11 only, so a 22 GiB
memory-modified card cannot match this recipe.

The lab run passed all six gates: load, chat, reasoning, tools, context,
speed — decode 77.4 tok/s, prefill 4045 tok/s. Recipe:
`registry/recipes/nvidia/rtx-2080-ti-11gb/qwen3.5-9b.tabbyapi.64k.q4.json`.
Evidence: `lab/runs/rtx-2080-ti-11gb.qwen3.5-9b.tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4.64k.20261105T113656.json`.

Why a host launch, not the `tabbyapi-exl3` container image: the container
image pins ExLlamaV3 1.5.1 and the rented host cannot run Docker (the
instance container lacks the capability to create the Docker NAT chain), so
the measured launch runs TabbyAPI from source on the host, exactly as the
Ornith Turing host launch does. The frozen profile is
`tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4@9213b8b94873`, variant q4.

Use Linux Python 3.12 on a compute node. Keep HOME unchanged and all
caches/data/logs on project storage. Do not run installation, hashing,
extraction or inference on a cluster login node. No sudo or shared/base-package
changes are required.

Obtain registry main commit 9e828b871a0828f850e3e8e3cd01cb0bb3f41842 and apply
this change. Download all 12 files in adjacent model-download-evidence.json
from TheMelonGod/Qwen3.5-9B-exl3 revision
22ef1303062e0f6d0b282440f8c1f685947f4938; the helper verifies every full SHA256.
The review bundle includes artifacts/virtualenv.pyz, SHA256
215e29d6c8e2a4bf5f509828368629bbe7eddb847dcfee862db0f84e6df40047. This binary is
outside the registry patch. The mutable bootstrap URL is rejected if its
payload changes.

Run these commands from the patched registry. Replace the four /actual paths
with your own absolute compute storage paths. RUN must be a new directory.

```sh
REG=/actual/patched-registry
RUN=/actual/new-qwen-runroot
MODEL=/actual/downloaded-model
PYZ=/actual/unpacked-review-bundle/artifacts/virtualenv.pyz
DOC="$REG/docs/rtx2080ti-qwen3.5-9b-exl3-q4-host"
PROFILE="$REG/registry/launches/tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4.json"

python3.12 "$DOC/reproduce_host.py" setup --run-dir "$RUN" --model-dir "$MODEL" --profile "$PROFILE" --manifest "$DOC/model-download-evidence.json" --virtualenv-pyz "$PYZ"
python3.12 "$DOC/reproduce_host.py" serve --run-dir "$RUN" --model-dir "$MODEL" --profile "$PROFILE" --manifest "$DOC/model-download-evidence.json"
```

In a second terminal in that same GPU allocation, run the original gates
against the service. The helper invokes the unchanged lab with no --set or
output cap. Warm the server with one chat call first: the very first
generation after load once returned no content (finish `repetition`); every
later call, including the recorded run, answers normally.

```sh
REG=/actual/patched-registry
RUN=/actual/new-qwen-runroot
MODEL=/actual/downloaded-model
DOC="$REG/docs/rtx2080ti-qwen3.5-9b-exl3-q4-host"
PROFILE="$REG/registry/launches/tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4.json"
python3.12 "$DOC/reproduce_host.py" gates --run-dir "$RUN" --model-dir "$MODEL" --profile "$PROFILE" --manifest "$DOC/model-download-evidence.json" --registry "$REG"
cd "$REG"
make
make check
```

Shell variables must be set in each terminal. Stop the service after
measurement. The exact command is `runtime/bin/python -u tabby/main.py
--config launch/tabbyapi-exl3-turing-qwen3.5-9b-64k2-q4.yml`, bound to
127.0.0.1:5000. This validation used Node v22.14.0; full make check requires
node in PATH, outside the inference launch.

No external PR, bounty claim, maintainer acceptance or payment has been
established by this document. Passing these service gates does not establish
any certification beyond the six measured gates on the named card.
