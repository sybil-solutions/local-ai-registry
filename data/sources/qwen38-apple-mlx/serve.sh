#!/usr/bin/env bash
# Reference launcher for the Qwen3.8-27B Apple recipes, as used for the evidence in this folder.
# Usage: ./serve.sh 32gb|24gb [port]    (expects mlx-vlm==0.7.3 in ./.venv)
# MEMLOG=file.jsonl wraps the server with memserve.py (MLX memory once a second, GiB).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  32gb) TARGET=mlx-community/Qwen3.8-27B-4bit;      TARGET_REV=10c35caafbb80f7dc6a7a432cdd11af10a6d4818
        DRAFT=mlx-community/Qwen3.8-27B-MTP-4bit;   DRAFT_REV=b643c01b6d3b094e325edb6ebd832e16c486c575 ;;
  24gb) TARGET=leonsarmiento/Qwen3.8-27B-3bit-mlx;  TARGET_REV=5fc234d9e6080b8388a11286380e801b7c9f535c
        DRAFT=lukaskremla/Qwen3.8-27B-MTP-3bit-MLX; DRAFT_REV=9d061a0661258e75b401a11ac9fa22fc648e039d ;;
  *) echo "usage: $0 32gb|24gb [port]" >&2; exit 2 ;;
esac
PORT="${2:-8080}"
# A 200k-token prefill takes longer than the default 600 s wait for the first token.
export MLX_VLM_TOKEN_QUEUE_TIMEOUT="${MLX_VLM_TOKEN_QUEUE_TIMEOUT:-7200}"

# `hf download` prints the snapshot directory of that exact commit; serving it keeps the revision pinned.
TARGET_DIR=$(.venv/bin/hf download "$TARGET" --revision "$TARGET_REV" --quiet)
DRAFT_DIR=$(.venv/bin/hf download "$DRAFT" --revision "$DRAFT_REV" --quiet)

ARGS=(--model "$TARGET_DIR" --draft-model "$DRAFT_DIR" --draft-kind mtp
      --kv-bits 4 --kv-quant-scheme uniform --kv-group-size 64
      --max-kv-size 200000 --max-num-seqs 1 --prefill-step-size 128
      --host 127.0.0.1 --port "$PORT")
LAUNCHER=../../../registry/launches/mlx-vlm-qwen3.8-long-prefill.py
if [ -n "${MEMLOG:-}" ]; then
  LAUNCHER="$LAUNCHER" exec .venv/bin/python memserve.py "${ARGS[@]}"
fi
exec .venv/bin/python "$LAUNCHER" "${ARGS[@]}"
