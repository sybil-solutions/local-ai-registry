#!/usr/bin/env bash
# queue.sh NAME...: run bench.py on configs/NAME.json one at a time (serialized by a mkdir lock), then stop the server.
W=/tmp/inkling-skill/work; cd "$W"
until mkdir "$W/.lock" 2>/dev/null; do sleep 5; done
trap 'rmdir "$W/.lock"' EXIT
for n in "$@"; do python3 -I bench.py run "configs/$n.json" ${BENCH_ARGS:-} > "logs/bench-$n.out" 2>&1; done
bash "$W/stop.sh"
