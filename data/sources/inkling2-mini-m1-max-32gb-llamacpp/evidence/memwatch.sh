#!/usr/bin/env bash
# 1 Hz memory log for macOS (the recipe kit's memwatch.sh reads /proc and is Linux-only).
# Each line: UTC time, system wired+active+compressed GiB, swap used MiB, llama-server RSS GiB.
#   memwatch.sh OUT.jsonl
set -u
out=$1
while :; do
  sys=$(vm_stat | awk '/page size of/ {ps=$8} /Pages active/ {a=$3} /Pages wired down/ {w=$4} /occupied by compressor/ {c=$5}
    END {gsub(/\./,"",a); gsub(/\./,"",w); gsub(/\./,"",c); printf "%.2f", (a+w+c)*ps/2^30}')
  swap=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
  rss=$(ps -axo rss=,comm= | awk '/llama-server$/ {printf "%.2f", $1/1048576; exit}')
  printf '{"t":"%s","sys_gib":%s,"swap_used_mib":%s,"server_rss_gib":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sys" "${swap:-0}" "${rss:-0}" >> "$out"
  sleep 1
done
