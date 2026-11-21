#!/usr/bin/env bash
# The measured launch, generated from the launch JSON command; run from the directory holding llama-b12600/ and models/.
set -euo pipefail
cd "$(dirname "$0")"
exec ./llama-b12600/llama-server -m models/Inkling-2-Mini-GGUF-db61c5b4/Inkling-2-Mini-IQ4_NL.gguf --jinja -ngl 99 -fa on -c 131072 -np 1 -b 1024 -ub 128 -ctk q8_0 -ctv q8_0 --poll 0 --spec-type ngram-mod --no-ui --alias inkling-2-mini --host 127.0.0.1 --port 8080
