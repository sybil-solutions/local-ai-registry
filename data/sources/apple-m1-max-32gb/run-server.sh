#!/usr/bin/env bash
# The measured launch for the Apple M1 Max 32 GB evidence (2026-11-03), paths in $HOME form.
# Same command, flags, environment and launcher as registry/launches/mlx-lm-lfm2.5-24b-a2b-6bit-128k.json; on top,
# and only on top: memserve_lab.py's once-per-second MLX memory log, and caffeinate so the Mac cannot sleep.
set -u
LAR=${LAR:-$HOME/Projects/local-ai-registry}
cd "$HOME/lai-work/run"  # holds models -> the pinned weights directory
exec caffeinate -dimsu env HF_HUB_OFFLINE=1 \
  MEMLOG="$HOME/lai-work/memwatch.jsonl" \
  LAUNCHER="$LAR/registry/launches/mlx-lm-lfm2.5-serve.py" \
  "$HOME/lai-work/.venv/bin/python" "$LAR/data/sources/apple-m1-max-32gb/memserve_lab.py" \
  --model models/LFM2.5-24B-A2B-MLX-6bit-1deec53e --max-tokens 131072 --decode-concurrency 1 \
  --prompt-concurrency 1 --prompt-cache-size 1 --host 127.0.0.1 --port 8080
