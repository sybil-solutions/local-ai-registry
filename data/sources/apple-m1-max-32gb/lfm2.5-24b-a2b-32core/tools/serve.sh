#!/usr/bin/env bash
# Start the pinned mlx-lm launch for one experiment, detached, logging to logs/<name>.server.log, with the
# 1 Hz MLX memory log (memserve_lab.py) in logs/<name>.mem.jsonl.
#   serve.sh <name> [extra mlx_lm.server args...]
# Base: the registry launch mlx-lm-lfm2.5-24b-a2b-6bit-128k (128K max tokens, one request at a time).
# exp/memserve_lab.py is a copy of ../../memserve_lab.py and run/lfm25_mlx_serve.py a copy of
# registry/launches/mlx-lm-lfm2.5-serve.py (sha256 7003b19c...5024f), both unchanged.
set -euo pipefail
name=$1; shift
root=$HOME/lai-work/lfm
mkdir -p "$root/exp/logs"
cd "$root/run"
HF_HUB_OFFLINE=1 MEMLOG="$root/exp/logs/$name.mem.jsonl" LAUNCHER="$root/run/lfm25_mlx_serve.py" \
nohup caffeinate -dimsu "$root/.venv/bin/python" "$root/exp/memserve_lab.py" \
  --model models/LFM2.5-24B-A2B-MLX-6bit-1deec53e --max-tokens 131072 --decode-concurrency 1 \
  --prompt-concurrency 1 --prompt-cache-size 1 --host 127.0.0.1 --port 8080 "$@" \
  > "$root/exp/logs/$name.server.log" 2>&1 < /dev/null &
echo "$!" > "$root/exp/logs/server.pid"
t0=$(date +%s)
until curl -sf http://127.0.0.1:8080/v1/models > /dev/null; do
  sleep 1
  kill -0 "$(cat "$root/exp/logs/server.pid")" 2> /dev/null || { echo "server exited"; tail -20 "$root/exp/logs/$name.server.log"; exit 1; }
done
echo "healthy after $(( $(date +%s) - t0 )) s"
