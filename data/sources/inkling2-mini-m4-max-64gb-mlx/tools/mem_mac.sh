#!/usr/bin/env bash
# Physical footprint (bytes) of the model server, including Metal/IOAccelerator residency (unified memory).
pid=$(cat "${1:-run/server.pid}" 2>/dev/null) || { echo 0; exit 0; }
kill -0 "$pid" 2>/dev/null || { echo 0; exit 0; }
footprint -p "$pid" -f bytes --noCategories 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="Footprint:"){print $(i+1); exit}}'
