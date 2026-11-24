#!/usr/bin/env bash
# Inkling-2-Mini at 397,312 tokens on one RTX 3090 (maya-ik2, every expert in VRAM): engine + patch, vision encoder,
# tokenizer, pinned weights, the NVFP4-tail GGUF (sha256-checked) and the server config. Run in an empty directory;
# it creates maya-inkling-mini/ (the launch's working directory). Needs CUDA 12.x nvcc, cmake >= 3.21, ninja,
# python >= 3.10, git, huggingface_hub (hf), ~36 GB disk, ~40 GB RAM for the requant step.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_COMMIT=cc41ab12af88e8c57b0f36250997f28f8763a3cf
TAIL_SHA=ee205a600be110910ae0b94d88a5b3671d2f2440dda36f5d282e7c82311db651
git clone https://github.com/alesha-pro/maya-inkling-mini
cd maya-inkling-mini
git checkout "$ENGINE_COMMIT"
git apply "$HERE/maya-ik2-session2-k13-full.patch"
python3 -m venv .venv
.venv/bin/pip install -q numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub
# engine (sm_86)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build --target maya-ik2
# vision encoder (CPU) on llama.cpp b12590, as the repo's guide builds it
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision --target maya-vision
# a shared ggml-base from the same llama.cpp for the requant (its NVFP4 quantizer)
cmake -S third_party/llama.cpp-b12590 -B build-ggml -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build-ggml --target ggml-base
# pinned weights
.venv/bin/hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f --local-dir models/Inkling-2-Mini-GGUF-ae9327c5
.venv/bin/hf download peasantsmith/Inkling-2-Mini-Maya-GGUF drafter/mtp.safetensors \
  --revision 37027719327de5f574684b525ad88fe50c4ff222 --local-dir models/Inkling-2-Mini-Maya-GGUF-37027719
.venv/bin/hf download peasantsmith/Inkling-2-Mini-Maya-GGUF vision/mmproj-Inkling-2-Mini-BF16.gguf \
  --revision 7ab64bac7090c9f77df551ac0fac98119494a2a5 --local-dir models/Inkling-2-Mini-Maya-GGUF-7ab64bac
# tokenizer
.venv/bin/python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-GGUF-ae9327c5/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja --check
# the NVFP4 tail: experts of blocks 20-23 Q8_0 -> NVFP4 (+ per-expert .scale), everything else byte for byte
GGUF_PY=third_party/llama.cpp-b12590/gguf-py LIBGGML_BASE="$PWD/build-ggml/bin/libggml-base.so" \
  .venv/bin/python "$HERE/requant.py" models/Inkling-2-Mini-GGUF-ae9327c5/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  models/Inkling-2-Mini-UD-Q4_K_XL-nvfp4tail.gguf 'blk\.2[0-3]\.ffn_(gate_up|down)_exps\.weight' NVFP4
echo "$TAIL_SHA  models/Inkling-2-Mini-UD-Q4_K_XL-nvfp4tail.gguf" | sha256sum -c -
# the server config the launch names
cp "$HERE/ik2-388k.json" docs/inkling2-mini/ik2-388k.json
echo "ready: run the launch's command in $PWD"
