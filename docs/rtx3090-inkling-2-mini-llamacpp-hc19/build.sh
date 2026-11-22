#!/bin/bash
# Build the llama-server this launch runs: unsloth's llama.cpp release b12588-mix-c0c3de8 plus inkling2-hc19.patch.
# Usage: build.sh [dest]   (default dest: llama.cpp-inkling2-hc19, next to the weights' models/ directory)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${1:-llama.cpp-inkling2-hc19}
url=https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/llama.cpp-source-b12588-mix-c0c3de8.tar.gz
sha=92ee8439f9ae51a40f15c168ac64e1d657bf8b08d301c96f8fb2d58625573607
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
curl -fL -o "$tmp/src.tar.gz" "$url"
echo "$sha  $tmp/src.tar.gz" | sha256sum -c -
mkdir -p "$dest"
tar -xzf "$tmp/src.tar.gz" -C "$dest" --strip-components=1
cd "$dest"
git apply --check "$here/inkling2-hc19.patch"
git apply "$here/inkling2-hc19.patch"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-server
echo "built $(pwd)/build/bin/llama-server"
