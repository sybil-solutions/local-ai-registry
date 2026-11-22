#!/bin/bash
# Sample GPU memory/power/clock and host MemAvailable once a second into a CSV, until killed.
# Usage: memwatch.sh <out.csv>
set -u
out=$1
echo "time,gpu_mem_used_mib,gpu_util_pct,power_w,sm_clock_mhz,host_mem_available_mib" > "$out"
while :; do
  g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ' ')
  h=$(awk '/^MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  echo "$(date -u +%FT%T),$g,$h" >> "$out"
  sleep 1
done
