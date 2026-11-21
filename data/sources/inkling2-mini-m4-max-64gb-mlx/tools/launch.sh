#!/usr/bin/env bash
# launch.sh <server args...>: start inkling2_mlx_serve.py detached, write run/server.pid. Env vars pass through.
cd "$(dirname "$0")/.."; mkdir -p run logs
HF_HUB_OFFLINE=1 nohup .venv/bin/python inkling2_mlx_serve.py "$@" > logs/server.log 2>&1 < /dev/null &
echo $! > run/server.pid
