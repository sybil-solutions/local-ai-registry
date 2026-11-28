#!/usr/bin/env bash
# Round 3: load mode. -lm none (weights copied, not mmapped) and -lm mmap+mlock, then a control.
set -u
cd "$HOME/lai-work/inkling/exp"
until grep -q ROUND2_DONE logs/round2.log; do sleep 5; done
COMMON=(-ub 256 -ctk q8_0 -ctv q8_0)
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" > "logs/serve-$name.txt" 2>&1 || { echo "fail $name: $(tail -3 logs/serve-$name.txt)"; return; }; python3 -I ../harness/bench.py run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log) $(cat logs/serve-$name.txt)"; }
run lmnone "${COMMON[@]}" -lm none
run mlock "${COMMON[@]}" -lm mmap+mlock
run base256-d "${COMMON[@]}"
./stop.sh
echo ROUND3_DONE
