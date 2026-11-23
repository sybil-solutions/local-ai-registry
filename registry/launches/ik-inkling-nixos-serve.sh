#!/usr/bin/env bash
set -euo pipefail
# Run from the directory containing models/, after downloading the pinned GGUF.
model="$(pwd)/models/Inkling-2-Mini-Uncensored-GGUF-40886ab6/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf"
printf '%s  %s\n' '23a8e6792481b6f1d0ebad51399e9ee2a2c2be8525b09ed1ae6737811159f892' "$model" | sha256sum --check
server=$(nix build --no-link --print-out-paths --impure --expr '
  let
    config = builtins.getFlake "github:ImLunaHey/nixos-configs/3d4450eb893ead81311a3bce2d809277a1932b66";
    pkgs = import config.inputs.nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
  in import (config.outPath + "/packages/llama-cpp-inkling") { inherit pkgs; }
')
export LD_LIBRARY_PATH=/run/opengl-driver/lib
exec "$server/bin/llama-server" --model "$model" \
  --alias inkling-2-mini-abliterated --host 0.0.0.0 --port 8080 \
  --metrics --jinja --ctx-size 131072 --parallel 1 --n-gpu-layers 99 \
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 1024 --ubatch-size 128 --swa-compress
