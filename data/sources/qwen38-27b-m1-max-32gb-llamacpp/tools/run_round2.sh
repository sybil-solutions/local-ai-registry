#!/usr/bin/env bash
# Round 2: longer MTP drafts (4, 5), a repeat of the best so far (3), and a control.
set -u
cd "$HOME/lai-work/qwen/exp"
until grep -q SWEEP_DONE logs/sweep.log; do sleep 5; done
COMMON=(-ub 256 -ctk q8_0 -ctv q8_0)
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" > "logs/serve-$name.txt" 2>&1 || { echo "fail $name: $(tail -3 logs/serve-$name.txt)"; return; }; python3 -I "$HOME/lai-work/inkling/harness/bench.py" run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log) $(cat logs/serve-$name.txt)"; }
run mtp4 "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 4
run mtp5 "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 5
run mtp3-b "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max 3
run ctl-d "${COMMON[@]}"
./stop.sh
echo ROUND2_DONE
