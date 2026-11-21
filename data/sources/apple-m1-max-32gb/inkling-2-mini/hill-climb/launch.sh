#!/usr/bin/env bash
# launch.sh NAME [llama-server args...]: start the pinned b12600 llama-server detached on 127.0.0.1:8090.
set -euo pipefail
W=/tmp/inkling-skill/work; name=$1; shift
nohup "$W/bin/llama-b12600/llama-server" -m "$HOME/models/Inkling-2-Mini-IQ4_NL-db61c5b4/Inkling-2-Mini-IQ4_NL.gguf" \
  --alias inkling --host 127.0.0.1 --port 8090 --no-ui "$@" > "$W/logs/$name.log" 2>&1 &
echo $! > "$W/server.pid"
