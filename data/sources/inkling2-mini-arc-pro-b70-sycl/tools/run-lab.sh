#!/usr/bin/env bash
# Start the container from the registry launch, run `lab.py try` against it, then stop it.
# Usage: GGUF=<path to Inkling-2-Mini-Q4_K_M.gguf> [LAUNCH=<launch id>] [REGISTRY=<registry checkout>] [OUT=<output dir>] run-lab.sh
# The image must be tagged localhost/llama-inkling-b70:<first 7 characters of the launch's build.commit>.
set -u
R=$(realpath "${REGISTRY:-.}"); P=$(realpath -m "${OUT:-./lab-out}"); mkdir -p "$P"
GG=$(realpath "${GGUF:?set GGUF to the pinned Inkling-2-Mini-Q4_K_M.gguf}")
ID=${LAUNCH:-llama.cpp-sycl-inkling-2-mini-q4-k-m-128k-arc-pro-b70}
W=lmstudio-community/Inkling-2-Mini-GGUF@23c938b028945d4b0fa9f71fb603b5cd1a683e71

mapfile -t CMD < <(python3 - "$R/registry/launches/$ID.json" "$GG" <<'PY'
import json, sys
L = json.load(open(sys.argv[1])); w = L["weights"][0]
c = ["podman", "run", "--rm", "--name", "inkling-lab", "--device", "/dev/dri", "-p", "18080:%d" % L["port"], "--shm-size", L["shm"]]
for k, v in L["env"].items():
    c += ["-e", f"{k}={v}"]
c += ["-v", f"{sys.argv[2]}:{w['at']}/{w['files']}:ro", "localhost/llama-inkling-b70:" + L["build"]["commit"][:7]] + L["args"]
print("\n".join(c))
PY
)
echo "${CMD[*]}" > "$P/cmd.txt"
"${CMD[@]}" > "$P/server.log" 2>&1 &

for i in $(seq 1 600); do curl -sf http://127.0.0.1:18080/health >/dev/null && break; sleep 1; done
echo "healthy after ${i}s" > "$P/load.txt"

# peak device memory: the largest drm-*-vram0 value of the server's DRM clients, once a second
( while podman container exists inkling-lab 2>/dev/null; do
	for q in $(pgrep -f /app/llama-server); do cat /proc/$q/fdinfo/* 2>/dev/null; done |
		awk '/^drm-(total|resident)-vram0:/{s=$2; if($3=="KiB")s/=1024; else if($3=="GiB")s*=1024; if(s>m)m=s} END{printf "%d\n", m}' >> "$P/vram.samples"
	sleep 1
done ) &

(cd "$R" && python3 lab/lab.py try "$W" --model inkling-2-mini --engine "$ID" --card intel-arc-pro-b70-32gb --on endpoint \
	--endpoint http://127.0.0.1:18080 --gpu "Intel Arc Pro B70 32 GB (owner)") > "$P/lab.out" 2>&1
echo "lab exit $?" >> "$P/lab.out"
podman stop -t 60 inkling-lab >/dev/null 2>&1
wait
sort -n "$P/vram.samples" | tail -1 > "$P/vram.max"
