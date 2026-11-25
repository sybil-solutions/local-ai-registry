#!/usr/bin/env bash
# macOS port of skill-scripts/queue.sh (no flock/setsid): one GPU job at a time via a mkdir lock.
#   nohup bash tools/queue_mac.sh base cand base > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
LOCK=/tmp/inference-hillclimb.lockdir
until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
trap 'rmdir "$LOCK"' EXIT
for n in "$@"; do
  pmset -g batt | head -1 | grep -q "AC Power" || { echo "ON BATTERY, stopping before $n"; break; }
  [ -n "${PRESTOP:-}" ] && bash -c "$PRESTOP"
  echo "$(date +%T) start $n $(pmset -g batt | head -1)"
  .venv/bin/python skill-scripts/bench.py run "configs/$n.json" ${BENCH_FLAGS:-} > "logs/run-$n-$(date +%H%M%S).log" 2>&1
  echo "$(date +%T) done $n: $(tail -1 ledger.jsonl | cut -c1-300)"
done
bash tools/stop.sh
echo QUEUE_DONE "$@"
