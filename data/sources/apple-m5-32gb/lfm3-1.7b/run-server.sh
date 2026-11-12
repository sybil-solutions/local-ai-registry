#!/bin/bash
# The measured launch: the registry's command, environment and packages, started from the folder that holds the
# launcher and the weights (models/LFM3-1.7B-MLX-8bit-4d6d94c9, LiquidAI/LFM3-1.7B-MLX-8bit at commit 4d6d94c9...).
cd "$HOME/projects/lfm3-m5.noindex"
export PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 MLX_VLM_TOKEN_QUEUE_TIMEOUT=7200
exec python3 lfm3_mlx_serve.py --model models/LFM3-1.7B-MLX-8bit-4d6d94c9 --max-tokens 262144 --enable-thinking --max-num-seqs 1 --prefill-step-size 128 --host 127.0.0.1 --port 8080
