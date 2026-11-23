#!/bin/bash
set -euo pipefail
# Build by-path aliases inside the container; the host needs only render-node access.
mkdir -p /dev/dri/by-path
for node in /dev/dri/renderD*; do
  [ -e "$node" ] || continue
  name=${node##*/}
  device=$(readlink -f "/sys/class/drm/$name/device")
  pci=${device##*/}
  ln -sfn "../$name" "/dev/dri/by-path/pci-$pci-render"
done
exec "$@"
