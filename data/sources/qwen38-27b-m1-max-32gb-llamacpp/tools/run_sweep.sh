#!/usr/bin/env bash
# Round 1: built-in MTP head at 1/2/3 draft tokens between controls; round 2: larger ubatch for prefill.
# Base: 128K, -ub 256, q8_0 KV (same as the Inkling recipe). Waits for the baseline bench to finish first.
set -u
cd "$HOME/lai-work/qwen/exp"
while pgrep -f 'bench.py run configs/base-a.json' > /dev/null; do sleep 5; done
COMMON=(-ub 256 -ctk q8_0 -ctv q8_0)
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" > "logs/serve-$name.txt" 2>&1 || { echo "fail $name: $(tail -3 logs/serve-$name.txt)"; return; }; python3 -I "$HOME/lai-work/inkling/harness/bench.py" run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log) $(cat logs/serve-$name.txt)"; }
run ctl-a "${COMMON[@]}"
run mtp1 "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 1
run mtp2 "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 2
run mtp3 "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 3
run ctl-b "${COMMON[@]}"
run ub512 -ub 512 -ctk q8_0 -ctv q8_0
run ub1024 -ub 1024 -b 2048 -ctk q8_0 -ctv q8_0
run ctl-c "${COMMON[@]}"
./stop.sh
echo SWEEP_DONE
