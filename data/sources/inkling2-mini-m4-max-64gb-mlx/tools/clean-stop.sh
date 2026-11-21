#!/usr/bin/env bash
# Graceful stop of the clean-build server: wait for the log to go quiet (max 120 s), SIGTERM, wait 30 s, then SIGKILL.
cd "$(dirname "$0")"; pid=$(cat server.pid 2>/dev/null) || exit 0
kill -0 "$pid" 2>/dev/null || exit 0
for i in $(seq 1 24); do a=$(wc -c < logs/server.log); sleep 5; [ "$a" = "$(wc -c < logs/server.log)" ] && break; done
kill -TERM "$pid"; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"; true
