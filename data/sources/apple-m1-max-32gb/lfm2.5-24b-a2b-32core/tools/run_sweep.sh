#!/usr/bin/env bash
# LFM2.5 hill-climb: prefill step size and 8-bit KV between controls (base launch = the registry recipe).
set -u
cd "$HOME/lai-work/lfm/exp"
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" > "logs/serve-$name.txt" 2>&1 || { echo "fail $name"; return; }; python3 -I "$HOME/lai-work/inkling/harness/bench.py" run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_32768": [0-9.]*' logs/$name.bench.log)"; }
run ctl-a
run pss4096 --prefill-step-size 4096
run pss1024 --prefill-step-size 1024
run ctl-b
run pss512 --prefill-step-size 512
run kv8 --kv-bits 8
run ctl-c
./stop.sh
echo SWEEP_DONE
