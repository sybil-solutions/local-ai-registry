#!/usr/bin/env bash
# Print system memory in use (wired + active + compressed pages) in bytes, for bench.py's memory watchdog on macOS.
# Unified memory: Metal buffers over mmapped GGUF weights are file-backed and do not show in a process footprint,
# so the watchdog reads the whole system; compare against the idle baseline recorded before each server start.
set -euo pipefail
vm_stat | awk '
  /page size of/ { ps = $8 }
  /Pages active/ { a = $3 }
  /Pages wired down/ { w = $4 }
  /occupied by compressor/ { c = $5 }
  END { gsub(/\./, "", a); gsub(/\./, "", w); gsub(/\./, "", c); printf "%d\n", (a + w + c) * ps }'
