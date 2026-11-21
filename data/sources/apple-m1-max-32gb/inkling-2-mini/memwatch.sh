#!/usr/bin/env bash
# memwatch.sh PID OUT: 1 Hz samples of the server's memory (phys_footprint incl. Metal buffers + resident mmapped weights), GiB, JSONL.
pid=$1; out=$2
while kill -0 "$pid" 2>/dev/null; do
  footprint -p "$pid" 2>/dev/null | awk -v t="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
    function b(v,u){return v*((u~/GB/)?2^30:(u~/MB/)?2^20:(u~/KB/)?2^10:1)}
    /Footprint:/{for(i=1;i<=NF;i++) if($i=="Footprint:") fp=b($(i+1),$(i+2))}
    /mapped file$/{mf=b($3,$4)}
    END{printf "{\"t\":\"%s\",\"footprint_gib\":%.2f,\"mapped_weights_gib\":%.2f,\"total_gib\":%.2f}\n", t, fp/2^30, mf/2^30, (fp+mf)/2^30}' >> "$out"
  sleep 1
done
