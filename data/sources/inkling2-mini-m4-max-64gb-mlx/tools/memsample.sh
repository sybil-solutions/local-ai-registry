#!/usr/bin/env bash
# Footprint samples (incl. Metal residency) of the server every 5 s -> logs/host-memory.jsonl, until it exits.
cd "$(dirname "$0")"; pid=$(cat server.pid)
while kill -0 "$pid" 2>/dev/null; do
  b=$(footprint -p "$pid" -f bytes --noCategories 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="Footprint:"){print $(i+1); exit}}')
  echo "{\"t\": \"$(date -u +%FT%TZ)\", \"footprint_bytes\": ${b:-0}}" >> logs/host-memory.jsonl; sleep 5
done
