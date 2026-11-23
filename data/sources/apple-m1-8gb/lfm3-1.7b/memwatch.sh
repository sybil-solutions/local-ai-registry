#!/bin/bash
# Every 10 s: the server's physical footprint (MB, includes GPU memory), free and swap-used memory, 1-min load.
PID="$1"; OUT="$2"
while kill -0 "$PID" 2>/dev/null; do
  fp=$(footprint -p "$PID" 2>/dev/null | awk '/phys_footprint:/ {print $2, $3; exit}')
  free=$(vm_stat | awk '/Pages free/ {gsub("\\.","",$3); print int($3*16384/1048576)}')
  swap=$(sysctl -n vm.swapusage | awk '{print $6}')
  echo "$(date -u +%H:%M:%S) footprint=$fp free_mb=$free swap_used=$swap load=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "$OUT"
  perl -e 'select(undef,undef,undef,10)'
done
