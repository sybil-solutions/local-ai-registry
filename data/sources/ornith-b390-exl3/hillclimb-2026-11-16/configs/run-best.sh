#!/usr/bin/env bash
# Start Ornith with the best hill-climbed config on http://127.0.0.1:8000/v1 (model: ornith-1.5-35b-a3b)
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import json, subprocess, os
c = json.load(open("best-launch.json"))
cache = os.path.expanduser("~/.cache/ornith-hc")
for d in ("sycl", "neo", "inductor", "triton"): os.makedirs(f"{cache}/{d}", exist_ok=True)
subprocess.run(["docker", "rm", "-f", "ornith"], capture_output=True)
env = {"SYCL_CACHE_PERSISTENT": "1", "SYCL_CACHE_DIR": "/hc-cache/sycl", "NEO_CACHE_PERSISTENT": "1",
       "NEO_CACHE_DIR": "/hc-cache/neo", "TORCHINDUCTOR_CACHE_DIR": "/hc-cache/inductor",
       "TRITON_CACHE_DIR": "/hc-cache/triton", **c["env"]}
args = ["docker", "run", "-d", "--name", "ornith", "--restart", "no", "--device", "/dev/dri/renderD128",
        "--shm-size", "4g", "-p", "127.0.0.1:8000:8000",
        "-v", os.path.expanduser("~/Models/Ornith-1.5-35B-A3B-EXL3-4bpw") + ":/models:ro", "-v", f"{cache}:/hc-cache"]
for k, v in env.items(): args += ["-e", f"{k}={v}"]
args += ["--entrypoint", "/opt/ornith-entrypoint.sh", c["image"], "python3", "-m", "sglang.launch_server",
         "--model-path", "/models", "--served-model-name", "ornith-1.5-35b-a3b", "--host", "0.0.0.0", "--port", "8000", *c["args"]]
subprocess.run(args, check=True)
print("started; ready in ~45 s: curl -s localhost:8000/health")
PY
