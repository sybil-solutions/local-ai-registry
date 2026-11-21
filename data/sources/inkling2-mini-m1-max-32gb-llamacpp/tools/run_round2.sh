#!/usr/bin/env bash
# Round 2: control, --no-mmap, -fa off, control (ub 256, q8_0 KV, no draft).
set -u
cd "$HOME/lai-work/inkling/exp"
until grep -q SWEEP_DONE logs/sweep.log; do sleep 5; done
COMMON=(-ub 256 -ctk q8_0 -ctv q8_0)
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" || { echo "fail $name"; return; }; python3 -I ../harness/bench.py run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log) $(grep -o 'healthy after [0-9]* s' logs/serve-$name.txt 2>/dev/null)"; }
run base256-b "${COMMON[@]}"
run nommap "${COMMON[@]}" --no-mmap
run faoff -ub 256 -ctk q8_0 -ctv q8_0 -fa off
run base256-c "${COMMON[@]}"
./stop.sh
echo ROUND2_DONE
