#!/usr/bin/env bash
# The measured launch for the Apple M1 Max 32 GB (32-core GPU) Qwen3.8-27B evidence, paths in $HOME form.
# Exactly the command of registry/launches/llama.cpp-metal-unsloth-qwen3.8-27b-ud-q4-k-xl-mtp3-128k-apple-m1-max-32gb.json,
# run from a directory holding llama-b12588-mix-c0c3de8/ (the unpacked pinned prebuilt) and models/ (the pinned weights).
# On top, and only on top: caffeinate so the Mac cannot sleep during the gates.
set -u
cd "$HOME/lai-work/qwen/run"
exec caffeinate -dimsu llama-b12588-mix-c0c3de8/llama-server \
  -m models/Qwen3.8-27B-GGUF-4af51b35/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -c 131072 -np 1 -ngl 99 -fa on --fit off -ub 256 -ctk q8_0 -ctv q8_0 --cache-ram 0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --jinja --alias qwen3.8-27b --host 127.0.0.1 --port 8080
