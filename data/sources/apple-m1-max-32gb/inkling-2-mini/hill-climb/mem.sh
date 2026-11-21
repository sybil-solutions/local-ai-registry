#!/usr/bin/env bash
# mem.sh: server memory in bytes = phys_footprint (Metal KV/compute buffers, heap) + resident mmapped weights.
pid=$(cat /tmp/inkling-skill/work/server.pid 2>/dev/null) || { echo 0; exit; }
footprint -p "$pid" 2>/dev/null | awk '
  function b(v,u){return v*((u~/GB/)?2^30:(u~/MB/)?2^20:(u~/KB/)?2^10:1)}
  /Footprint:/{for(i=1;i<=NF;i++) if($i=="Footprint:") fp=b($(i+1),$(i+2))}
  /mapped file$/{mf=b($3,$4)}
  END{printf "%d\n", fp+mf}'
