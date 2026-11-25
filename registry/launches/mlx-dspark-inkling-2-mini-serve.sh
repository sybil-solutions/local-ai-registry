#!/bin/sh
# Starts mlx-dspark's OpenAI server for Inkling 2 Mini.
# Usage: sh inkling2_dspark_serve.sh <mlx-dspark serve arguments>
#
# Finds the 32,768-row draft vocabulary that ships inside the mlx-dspark package and hands it to the
# engine (MLX_DSPARK_INKLING_DRAFT_VOCAB). The rest of the configuration is the launch's command and env.
set -eu
PYTHON="${PYTHON:-python3}"
VOCAB="$("$PYTHON" -c 'import mlx_dspark, os; print(os.path.join(os.path.dirname(mlx_dspark.__file__), "assets", "vocab-32768.npy"))')"
if [ ! -f "$VOCAB" ]; then
  echo "mlx-dspark draft vocabulary not found at $VOCAB; install the pinned mlx-dspark commit" >&2
  exit 1
fi
export MLX_DSPARK_INKLING_DRAFT_VOCAB="$VOCAB"
exec "$PYTHON" -m mlx_dspark serve "$@"
