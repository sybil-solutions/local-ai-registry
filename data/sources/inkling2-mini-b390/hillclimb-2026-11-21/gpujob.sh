#!/usr/bin/env bash
# gpujob.sh OUTFILE cmd...  — runs cmd holding the queue lock
cd "$(dirname "$0")"; out=$1; shift
exec 9>.queue.lock; flock 9
"$@" > "$out" 2>&1; echo "rc=$?" >> "$out"
