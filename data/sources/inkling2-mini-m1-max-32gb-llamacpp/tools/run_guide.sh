#!/usr/bin/env bash
# The Unsloth guide's llama.cpp launch (32K context, default -ub 512 and f16 KV, DSpark Q8_0 with 3 drafts, text only),
# A/B against the same launch without the drafter, interleaved. Starts once the lab gate server is gone.
set -u
cd "$HOME/lai-work/inkling/exp"
while curl -sf http://127.0.0.1:8080/v1/models > /dev/null; do sleep 5; done
run() { name=$1; shift; ./stop.sh; ./serve.sh "$name" "$@" > "logs/serve-$name.txt" 2>&1 || { echo "fail $name: $(tail -3 logs/serve-$name.txt)"; return; }; python3 -I ../harness/bench.py run "configs/$name.json" --skip-cN > "logs/$name.bench.log" 2>&1; echo "done $name: $(grep -o '"c1_decode": [0-9.]*' logs/$name.bench.log) $(grep -o '"prefill_4096": [0-9.]*' logs/$name.bench.log)"; }
run guide32k-nodraft-a -c 32768
run guide32k-dspark3 -c 32768 -md weights/dspark-Inkling-2-Mini-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 3 -ngld 99
run guide32k-nodraft-b -c 32768
./stop.sh
echo GUIDE_DONE
