#!/bin/bash
# On pop-os: free memory and per-card GPU memory every 2 s, until the file named as $2 exists.
out="$1"; stop="$2"
while [ ! -e "$stop" ]; do
  echo "$(date -u +%FT%TZ) mem_available_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo) gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//') gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')" >> "$out"
  sleep 2
done
