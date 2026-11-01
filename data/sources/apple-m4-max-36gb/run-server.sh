#!/usr/bin/env bash
# Reproduction launch for the Apple M4 Max 36 GB owner-machine evidence (2026-10-22).
# Identical to the recorded run's launch (~/ai-work/m4max/run_4bit.sh): same venv,
# same pinned local weights, same flags, same launcher
# (registry/launches/mlx-vlm-qwen3.8-long-prefill.py,
#  sha256 f8cd617060c26a5618d75a314931717779a72848f443c18a645e1db2d165f94a).
# Differences from the recorded launch, and why:
#   - memserve_lab.py runs the launcher with its __main__ lab routes + sampling defaults,
#     plus the repo's once-per-second MLX memory monitor
#     (data/sources/qwen38-apple-mlx/memserve.py), writing t/active_gb/peak_gb/cache_gb
#     to $MEMLOG.
#   - caffeinate -dimsu so the Mac cannot sleep mid-prefill.
set -u
# Paths parameterized for portability; on the owner machine at run time these
# resolved to the exact absolute paths used on 2026-10-22 (no other difference).
LAR=${LAR:-$HOME/dev/lar}
M4MAX=${M4MAX:-$HOME/ai-work/m4max}
EVID=${EVID:-$(cd "$(dirname "$0")" && pwd)}
cd "$M4MAX/run"
exec caffeinate -dimsu env -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  MLX_VLM_TOKEN_QUEUE_TIMEOUT=7200 \
  MEMLOG="$EVID/memwatch.jsonl" \
  LAUNCHER="$LAR/registry/launches/mlx-vlm-qwen3.8-long-prefill.py" \
  "$M4MAX/venv/bin/python" "$EVID/memserve_lab.py" \
  --model models/Qwen3.8-27B-4bit-10c35caa \
  --draft-model models/Qwen3.8-27B-MTP-4bit-b643c01b \
  --draft-kind mtp --draft-block-size 2 --kv-bits 4 --kv-quant-scheme uniform --kv-group-size 64 \
  --max-kv-size 204800 --max-tokens 8192 --enable-thinking --max-num-seqs 1 \
  --prefill-step-size 128 --host 127.0.0.1 --port 8080