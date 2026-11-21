#!/usr/bin/env bash
# stop.sh: graceful SIGTERM to the pidfile's server, 60 s timeout, never pattern-kill.
W=/tmp/inkling-skill/work; pid=$(cat "$W/server.pid" 2>/dev/null) || exit 0
kill -0 "$pid" 2>/dev/null || exit 0
sleep 2; kill -TERM "$pid"
for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || exit 0; sleep 1; done
kill -KILL "$pid"
