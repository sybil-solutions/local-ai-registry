#!/usr/bin/env bash
# The measured launch for the Apple M1 Max 32 GB (32-core GPU) Inkling 2 Mini evidence, paths in $HOME form.
# Exactly the command of registry/launches/llama.cpp-metal-unsloth-inkling-2-mini-ud-q4-k-xl-128k-apple-m1-max-32gb.json,
# run from a directory holding llama-b12588-mix-c0c3de8/ (the unpacked pinned prebuilt) and models/ (the pinned weights).
# On top, and only on top: caffeinate so the Mac cannot sleep during the gates.
set -u
cd "$HOME/lai-work/inkling/run"
exec caffeinate -dimsu llama-b12588-mix-c0c3de8/llama-server \
  -m models/Inkling-2-Mini-GGUF-ae9327c5/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -c 131072 -np 1 -ngl 99 -fa on --fit off -ub 256 -ctk q8_0 -ctv q8_0 --cache-ram 0 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --jinja --alias inkling-2-mini --host 127.0.0.1 --port 8080
