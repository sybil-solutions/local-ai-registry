#!/usr/bin/env bash
# Start the rendered launch exactly: its argv and env, from this directory, with the clean venv's python3 first on PATH.
cd "$(dirname "$0")"
eval "$(python3 -c 'import json,shlex; c=json.load(open("cmd.json")); print(" ".join(f"export {k}={shlex.quote(v)};" for k,v in c["env"].items())); print("CMD=(" + " ".join(shlex.quote(a) for a in c["command"]) + ")")')"
PATH="$PWD/.venv/bin:$PATH" nohup "${CMD[@]}" > logs/server.log 2>&1 < /dev/null &
echo $! > server.pid
