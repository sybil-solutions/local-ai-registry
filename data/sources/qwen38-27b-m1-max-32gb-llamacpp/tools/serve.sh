#!/usr/bin/env bash
# Start llama-server for one experiment, detached, logging to logs/<name>.server.log.
#   serve.sh <name> [extra llama-server args...]
# Base launch: 128K context, one sequence, all layers on Metal, flash attention on, Qwen's thinking-mode sampling.
# Same engine build as the Inkling 2 Mini run (unslothai/llama.cpp b12588-mix-c0c3de8 source build).
set -euo pipefail
name=$1; shift
root=$HOME/lai-work/qwen
mkdir -p "$root/exp/logs"
cd "$root"
nohup caffeinate -dimsu "$HOME/lai-work/inkling/src/tree/llama.cpp-b12588-mix-c0c3de8/build/bin/llama-server" \
  -m weights/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 131072 -np 1 -ngl 99 -fa on -fit off --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --alias qwen3.8-27b --host 127.0.0.1 --port 8080 "$@" \
  > "exp/logs/$name.server.log" 2>&1 < /dev/null &
echo "$!" > exp/logs/server.pid
t0=$(date +%s)
until curl -sf http://127.0.0.1:8080/v1/models > /dev/null; do
  sleep 1
  kill -0 "$(cat exp/logs/server.pid)" 2> /dev/null || { echo "server exited"; tail -20 "exp/logs/$name.server.log"; exit 1; }
done
echo "healthy after $(( $(date +%s) - t0 )) s"
