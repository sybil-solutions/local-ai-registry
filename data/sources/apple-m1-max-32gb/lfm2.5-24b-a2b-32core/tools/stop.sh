#!/usr/bin/env bash
# Stop the experiment server gracefully (SIGTERM, wait up to 60 s); call only when no request is in flight.
set -u
pidfile=$HOME/lai-work/lfm/exp/logs/server.pid
[ -f "$pidfile" ] || exit 0
pid=$(cat "$pidfile")
pkill -TERM -P "$pid" 2> /dev/null  # the server runs as caffeinate's child
kill -TERM "$pid" 2> /dev/null
for _ in $(seq 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
rm -f "$pidfile"
