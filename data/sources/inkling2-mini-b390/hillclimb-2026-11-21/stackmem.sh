#!/usr/bin/env bash
# stackmem.sh CONTAINER -> bytes: sum unique drm-resident-gtt per xe client + container cgroup anon
n=${1:-inkling}
pids=$(docker top "$n" -eo pid 2>/dev/null | tail -n +2)
[ -z "$pids" ] && { echo 0; exit; }
cid=$(docker inspect -f '{{.Id}}' "$n")
sudo -S -p '' bash -c '
tot=0; declare -A seen
for p in '"$(echo $pids)"'; do for f in /proc/$p/fdinfo/*; do
  id=$(awk "/drm-client-id/{print \$2}" $f 2>/dev/null); [ -z "$id" ] && continue; [ -n "${seen[$id]}" ] && continue; seen[$id]=1
  kb=$(awk "/drm-resident-gtt/{s+=\$2} END{print s+0}" $f); tot=$((tot+kb*1024)); done; done
cg=$(ls -d /sys/fs/cgroup/system.slice/docker-'"$cid"'.scope 2>/dev/null)
anon=$(awk "/^anon /{print \$2}" $cg/memory.stat 2>/dev/null); echo $((tot+${anon:-0}))
' < "${SUDO_PASS_FILE:?set SUDO_PASS_FILE to a file holding the sudo password}"
