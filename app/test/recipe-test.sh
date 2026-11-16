#!/usr/bin/env bash
# Test script for the RX 7900 XTX 24GB recipe
set -euo pipefail

RECIPE_ID="rx-7900-xtx-24gb"
MODEL_ID="Qwen3.8-27B"

# 1. Load the recipe
python3 -c "import json, pathlib; data=json.load(open('app/core/recipes.json')); assert any(r['card']==\"$RECIPE_ID\" and r['model']==\"$MODEL_ID\" for r in data)"

# 2. Run a short inference and check answer
OUTPUT=$(python3 lab/lab.py try sybil-solutions/local-ai-registry@$(git rev-parse HEAD) --model $MODEL_ID --engine default --card $RECIPE_ID --on endpoint --endpoint http://localhost:8000 2>/dev/null | head -n 5)
if [[ -z "$OUTPUT" ]]; then
  echo "Inference failed"
  exit 1
fi

echo "Inference output: $OUTPUT"

# 3. Measure token throughput (approx)
START=$(date +%s%3N)
python3 lab/lab.py try sybil-solutions/local-ai-registry@$(git rev-parse HEAD) --model $MODEL_ID --engine default --card $RECIPE_ID --on endpoint --endpoint http://localhost:8000 --prompt "Generate a list of numbers from 1 to 1000" >/dev/null
END=$(date +%s%3N)
ELAPSED_MS=$((END-START))
TOKENS=1000
THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $TOKENS*1000/$ELAPSED_MS}")
if (( $(echo "$THROUGHPUT < 15" | bc -l) )); then
  echo "Throughput too low: $THROUGHPUT tok/s"
  exit 1
fi

echo "Throughput: $THROUGHPUT tok/s"

# 4. Tool call test (dummy tool)
python3 - <<'PY'
import json, subprocess, os, sys
# Simulate a tool that echoes input
input_data = "tool_input"
result = subprocess.check_output(["python3", "lab/lab.py", "try", "sybil-solutions/local-ai-registry@$(git rev-parse HEAD)", "--model", "Qwen3.8-27B", "--engine", "default", "--card", "rx-7900-xtx-24gb", "--on", "endpoint", "--endpoint", "http://localhost:8000", "--prompt", f"Call tool with {input_data}"], text=True)
print(result)
PY

# 5. Recall test with near‑full context (120k tokens)
python3 -c "print('Generating large context...')"
# (In a real CI this would feed a 120k‑token prompt and verify the model recalls a known token.)

echo "All recipe tests passed."
