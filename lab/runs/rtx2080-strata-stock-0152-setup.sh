#!/usr/bin/env bash
# Run in an unmodified Strata checkout at the revision below.
set -euo pipefail
expected=5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d
[ "$(git rev-parse HEAD)" = "$expected" ]
git diff --quiet HEAD -- src include serve tools
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
fetch() {
  local target=$1 hash=$2 url=$3
  if [ -f "$target" ]; then
    echo "$hash  $target" | sha256sum -c -
    return
  fi
  mkdir -p "$(dirname "$target")"
  curl --fail --location --retry 3 --continue-at - --output "$target.part" "$url"
  echo "$hash  $target.part" | sha256sum -c -
  mv "$target.part" "$target"
}
archive=strata-engine-0.1.52-linux-x86_64-cuda-cuda13.tar.gz
fetch "$archive" 4177c1a8ffc59bf66074093b22ac55ed477bcbe84f8bd12e378b0f1236da55c7 \
  "https://github.com/Niko1221/Strata/releases/download/v0.2.7/$archive"
mkdir -p engine
tar xzf "$archive" -C engine
echo '9d415609c6abf40352345e20c75fc2205db0bc7acb804a75d12ab4ac10ee16ad  engine/strata' | sha256sum -c -
engine/strata --version
repo=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d
first=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
second=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
fetch "data/models/Swift-IQ2_XS/$first" ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b "$repo/$first"
fetch "data/models/Swift-IQ2_XS/$second" e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb "$repo/$second"
if [ ! -d third_party/llama.cpp/.git ]; then
  git init third_party/llama.cpp
  git -C third_party/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C third_party/llama.cpp fetch --depth 1 origin 3cf03257f219afbe7334045ff7c6a06ac68c627d
git -C third_party/llama.cpp checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
export STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py"
.venv/bin/python tools/iq_pack.py --gguf "data/models/Swift-IQ2_XS/$first" --out data/packs/swift-iq2_xs
# mtp_fetch.py pins the checkpoint revision and verifies the original tensor hashes.
.venv/bin/python tools/mtp_fetch.py fetch --out data/mtp
.venv/bin/python tools/mtp_fetch.py verify --out data/mtp
.venv/bin/python tools/mtp_pack.py --src data/mtp --experts q2_0 --out data/mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf data/mtp/mtp-q2_0.gguf --out data/mtp/rt
printf '%s\n' 'Preparation complete. Write registry-rtx2080.json from the rendered launch, then run its command.'
