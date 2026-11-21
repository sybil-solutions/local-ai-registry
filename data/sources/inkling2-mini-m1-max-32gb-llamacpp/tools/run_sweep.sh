#!/usr/bin/env bash
# Serialized sweep: control, DSpark draft lengths 1/2/3, control. One server at a time; stop only after bench returns.
set -u
cd "$HOME/lai-work/inkling/exp"
COMMON=(-ub 128 -ctk q8_0 -ctv q8_0)
SPEC=(-md weights/dspark-Inkling-2-Mini-Q8_0.gguf --spec-type draft-dspark -ngld 99 -ctkd q8_0 -ctvd q8_0)
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" || return; python3 -I ../harness/bench.py run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log)"; }
run base128-a "${COMMON[@]}"
run dspark1 "${COMMON[@]}" "${SPEC[@]}" --spec-draft-n-max 1
run dspark2 "${COMMON[@]}" "${SPEC[@]}" --spec-draft-n-max 2
run dspark3 "${COMMON[@]}" "${SPEC[@]}" --spec-draft-n-max 3
run base128-b "${COMMON[@]}"
./stop.sh
echo SWEEP_DONE
