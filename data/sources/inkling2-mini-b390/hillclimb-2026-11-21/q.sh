#!/usr/bin/env bash
# serialized queue: q.sh name...  (flock: one GPU job at a time)
cd "$(dirname "$0")"
exec 9>.queue.lock; flock 9
for n in "$@"; do python3 hc.py run configs/$n.json $HC_FLAGS > runs-$n.log 2>&1; done
echo QUEUE_DONE "$@"
