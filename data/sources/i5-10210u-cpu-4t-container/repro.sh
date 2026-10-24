#!/usr/bin/env bash
# Reproduce the CPU recipe lfm2.5-2.6b.llama.cpp.8k.4t (x86-64-avx2-cpu) with bounded steps and owned cleanup.
#
#   MODEL_DIR=/path/to/LFM2.5-2.6B-GGUF-84022ce7 LAB_DIR=/path/to/local-ai-registry \
#   GPU_NAME="<CPU model>, four threads, 4 GiB container cap, no GPU" \
#     bash data/sources/i5-10210u-cpu-4t-container/repro.sh
#
# Success (exit 0) needs all of: the lab's six gates (lab.py exits 0), the stock speed prompt repeated at >= 15 tok/s,
# exactly seven chat requests that each end on their own (check_run.py on the observer's raw bytes) with no output or
# thinking limit sent, and cleanup: the server log saved, then the container removed and confirmed gone. Anything
# else exits 1. GPU_NAME is required: it is written into the proof, so it must name the CPU actually used.
#
# The container is started from the lab's own rendering of the launch (lab.py try --dry-run), and the script stops
# unless that rendering's launch_sha256 is the accepted one, so the image, argv, memory cap and mounts cannot drift.
#
# Settings (environment): RUNTIME (docker; podman works), PORT (server, default 18090; the observer takes PORT+1),
# LIFETIME (whole run, default 3600 s), HEALTH_SECS (300), LAB_SECS (2700), REPEAT_SECS (900). The image must already
# be present (pull it first).
#
# observer.py and check_run.py are task tools next to this script, not part of the lab: the observer relays bytes
# unchanged between the lab and the server and keeps a copy; check_run.py reads that copy afterwards.
set -euo pipefail

# Each helper we start runs under setsid as the leader of its own process group, tracked as "pid start-time"
# (/proc/<pid>/stat field 22). The group is signalled only while the leader's pid AND start time still match, so a
# recycled pid is never hit; TERM first, KILL to the whole group after 5 s, then the leader is reaped.
pid_start() { [ -r "/proc/$1/stat" ] && sed -E 's/^.*\) //' "/proc/$1/stat" | cut -d' ' -f20; }
running() {  # running PID START: same process (pid and start time) and not yet a zombie
  local f; f=$(sed -E 's/^.*\) //' "/proc/$1/stat" 2>/dev/null) || return 1
  [ "$(cut -d' ' -f20 <<< "$f")" = "$2" ] && [ "${f%% *}" != Z ]
}
fenced_stop() {  # fenced_stop PID START: never touches a pid whose start time differs from the one recorded
  local pid=$1 start=$2 i
  [ -n "$pid" ] && [ -n "$start" ] || return 0
  if [ "$(pid_start "$pid")" != "$start" ]; then  # gone: reap it if it was ours; alive: another process, leave it
    [ -e "/proc/$pid" ] || wait "$pid" 2>/dev/null || true
    return 0
  fi
  if running "$pid" "$start"; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for i in 1 2 3 4 5 6 7 8 9 10; do running "$pid" "$start" || break; sleep 0.5; done
    running "$pid" "$start" && kill -KILL -- "-$pid" 2>/dev/null
  fi
  wait "$pid" 2>/dev/null || true   # ours, stopped or a zombie: reap it
}
if [ -n "${REPRO_LIB_ONLY:-}" ]; then return 0 2>/dev/null || exit 0; fi  # the test harness sources the helpers

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIFETIME=${LIFETIME:-3600}
# Cleanup budget, worst case: start delay 12 (a foreground health probe) + helpers 2 x 6 + label check 12 + logs 22
# + rm 22 + final inspect 12 = 92 s. The outer KILL comes 120 s after TERM, so cleanup always runs to its end.
KILL_AFTER=120
if [ -z "${REPRO_INNER:-}" ]; then  # the whole run lives under one hard deadline: TERM at LIFETIME, KILL 120 s later
  exec timeout --kill-after=$KILL_AFTER "$LIFETIME" env REPRO_INNER=1 bash "${BASH_SOURCE[0]}" "$@"
fi

: "${MODEL_DIR:?set MODEL_DIR to the weights directory}" "${LAB_DIR:?set LAB_DIR to a local-ai-registry checkout}"
: "${GPU_NAME:?set GPU_NAME to the CPU actually used, e.g. \"Intel Core i5-10210U (4 cores, 8 threads), four threads, 4 GiB container cap, no GPU\"}"
RUNTIME=${RUNTIME:-docker}; PORT=${PORT:-18090}; OBS_PORT=$((PORT + 1))
HEALTH_SECS=${HEALTH_SECS:-300}; LAB_SECS=${LAB_SECS:-2700}; REPEAT_SECS=${REPEAT_SECS:-900}
WEIGHTS=LiquidAI/LFM2.5-2.6B-GGUF@84022ce711b28455e8c4fc364ce68c00cf995875
ENGINE=llama.cpp-lfm2.5-2.6b-qad-q4-0-8k-cpu-docker-4t
LAUNCH_SHA256=f47439fecb04920684f9e6fa115763986025ee8ba8b2f54e739dd929fb9a49e4
LABEL_KEY=io.local-ai-registry.repro

# --- preflight: pins, inputs, free ports, image present
( cd "$MODEL_DIR" && echo "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03  LFM2.5-2.6B-QAD-Q4_0.gguf" | sha256sum -c --quiet - )
( cd "$LAB_DIR" && echo "b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca  lab/lab.py" | sha256sum -c --quiet - )
LAB_ARGS=("$WEIGHTS" --model lfm2.5-2.6b --engine "$ENGINE" --card x86-64-avx2-cpu --on endpoint)
# the launch exactly as the lab renders it; refuse anything but the accepted launch hash
RENDER=$(mktemp "${TMPDIR:-/tmp}/cpu-4t-render.XXXXXXXX")
( cd "$LAB_DIR" && timeout -k 5 60 python3 -I lab/lab.py try "${LAB_ARGS[@]}" --endpoint http://unused --dry-run ) > "$RENDER"
timeout -k 5 60 python3 -I - "$RENDER" "$LAUNCH_SHA256" "$LAB_DIR" <<'EOF' > "$RENDER.args" || { echo "rendered launch is not the accepted one"; exit 1; }
import json, sys
sys.path.insert(0, sys.argv[3] + "/lab")
import lab
L = json.load(open(sys.argv[1]))["launch"]
if lab.launch_hash(L) != sys.argv[2] or L["weights"][0]["at"] != "/models" or L.get("env"):
    sys.exit(1)
r = L["resources"]
sys.stdout.write("\0".join([L["image"], L["entrypoint"], str(L["port"]), L["shm"], str(r["memoryBytes"]),
                             str(r["memorySwapBytes"])] + L["args"]) + "\0")
EOF
mapfile -d '' -t R < "$RENDER.args"; rm -f "$RENDER" "$RENDER.args"
IMAGE=${R[0]}; ENTRY=${R[1]}; CPORT=${R[2]}; SHM=${R[3]}; MEM=${R[4]}; MEMSWAP=${R[5]}; ARGV=("${R[@]:6}")
for p in "$PORT" "$OBS_PORT"; do
  if python3 -I -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)' "$p"; then
    echo "port $p is already in use"; exit 1; fi
done
timeout -k 5 60 "$RUNTIME" image inspect "$IMAGE" > /dev/null || { echo "image $IMAGE is not present; pull it first"; exit 1; }

# --- a fresh private run directory; its name is also the container name and label value
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cpu-4t-llamacpp.XXXXXXXX"); chmod 700 "$RUN_DIR"
RUN_ID=$(basename "$RUN_DIR"); CID_FILE=$RUN_DIR/cid
CHILD=""; CHILD_START=""; OBS_PID=""; OBS_START=""; CLEANED=""
echo "run $RUN_ID in $RUN_DIR"

owned_cid() {  # print the container id only if the cidfile is ours and the container carries our label
  local cid
  [ -s "$CID_FILE" ] || return 1
  cid=$(cat "$CID_FILE")
  [[ $cid =~ ^[0-9a-f]{64}$ ]] || return 1
  [ "$(timeout -k 2 10 "$RUNTIME" inspect -f "{{index .Config.Labels \"$LABEL_KEY\"}}" "$cid" 2>/dev/null)" = "$RUN_ID" ] || return 1
  echo "$cid"
}

gone() {  # 0 only when the runtime positively says the container does not exist; a hang or other error is not "gone"
  local out rc=0
  out=$(timeout -k 2 10 "$RUNTIME" inspect "$1" 2>&1 > /dev/null) || rc=$?
  [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 137 ] && grep -qi "no such" <<< "$out"
}

cleanup() {  # success needs the server log saved (command ok, file non-empty) AND the container confirmed gone
  [ -n "$CLEANED" ] && return; CLEANED=1
  fenced_stop "$CHILD" "$CHILD_START"; fenced_stop "$OBS_PID" "$OBS_START"
  local cid log_ok=0 rm_ok=0
  if cid=$(owned_cid); then
    if timeout -k 2 20 "$RUNTIME" logs "$cid" > "$RUN_DIR/server.log" 2>&1 && [ -s "$RUN_DIR/server.log" ]; then
      log_ok=1; echo "server log saved: $(wc -c < "$RUN_DIR/server.log") bytes"
    else echo "SERVER LOG NOT SAVED (the command failed, timed out or wrote nothing); removing the container anyway"; fi
    timeout -k 2 20 "$RUNTIME" rm -f "$cid" > /dev/null 2>&1 || true   # always attempted, whatever happened to the log
    if gone "$cid"; then rm_ok=1; echo "container $cid removed"; else echo "CLEANUP FAILED: $cid not confirmed removed"; fi
    [ "$log_ok$rm_ok" = 11 ] && echo ok > "$RUN_DIR/cleanup" || echo fail > "$RUN_DIR/cleanup"
  elif [ -e "$CID_FILE" ]; then
    echo "cidfile does not name a container labelled $RUN_ID (or the runtime did not answer); nothing touched"
    echo fail > "$RUN_DIR/cleanup"
  else
    echo ok > "$RUN_DIR/cleanup"   # no container was ever created
  fi
}
trap 'exit 124' TERM INT
trap cleanup EXIT

bounded() {  # run a bounded step in the background so a TERM reaches the traps at once (bash defers traps until a
  setsid "$@" & CHILD=$!; CHILD_START=$(pid_start "$CHILD"); local rc=0   # foreground child exits, not a wait)
  wait "$CHILD" || rc=$?; CHILD=""; CHILD_START=""; return $rc
}

# --- the server: exactly the rendered launch (image, entrypoint, argv, shm, memory cap), plus name and label
bounded timeout -k 10 300 "$RUNTIME" run -d --cidfile "$CID_FILE" --name "$RUN_ID" --label "$LABEL_KEY=$RUN_ID" \
  --memory "$MEM" --memory-swap "$MEMSWAP" --shm-size "$SHM" -p "127.0.0.1:$PORT:$CPORT" -v "$MODEL_DIR:/models:ro" \
  --entrypoint "$ENTRY" "$IMAGE" "${ARGV[@]}" > /dev/null
CID=$(owned_cid) || { echo "started container is not the one labelled $RUN_ID"; exit 1; }
deadline=$((SECONDS + HEALTH_SECS))
until timeout -k 2 10 curl -sf -m 5 -o /dev/null "http://127.0.0.1:$PORT/health"; do
  [ "$(timeout -k 2 10 "$RUNTIME" inspect -f '{{.State.Running}}' "$CID")" = true ] || { echo "server exited before healthy"; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "server not healthy within ${HEALTH_SECS}s"; exit 1; }
  sleep 3
done
echo "server healthy after ${SECONDS}s"

# --- the observer: lab -> 127.0.0.1:OBS_PORT -> server, bytes unchanged both ways
setsid timeout -k 10 "$LIFETIME" python3 -I "$HERE/observer.py" 127.0.0.1 "$OBS_PORT" 127.0.0.1 "$PORT" "$RUN_DIR/observer" \
  > "$RUN_DIR/observer.log" 2>&1 & OBS_PID=$!; OBS_START=$(pid_start "$OBS_PID")
deadline=$((SECONDS + 30))
until timeout -k 2 10 curl -sf -m 5 -o /dev/null "http://127.0.0.1:$OBS_PORT/health"; do
  [ "$SECONDS" -lt "$deadline" ] || { echo "observer did not start"; exit 1; }; sleep 1
done
rm -f "$RUN_DIR"/observer/conn-*  # the readiness probes are not part of the checked run

# --- the lab's six gates, unchanged, through the observer
LAB_OK=0
( cd "$LAB_DIR" && exec setsid timeout -k 30 "$LAB_SECS" env -u LAB_API_KEY -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy python3 -I lab/lab.py try "${LAB_ARGS[@]}" \
    --endpoint "http://127.0.0.1:$OBS_PORT" --gpu "$GPU_NAME" ) \
  > "$RUN_DIR/lab-try.log" 2>&1 & CHILD=$!; CHILD_START=$(pid_start "$CHILD")
wait "$CHILD" && LAB_OK=1; CHILD=""; CHILD_START=""
tail -n 2 "$RUN_DIR/lab-try.log"

# --- one repeat of the stock speed prompt: the lab's own stream_rate and prompt, imported from the unchanged lab.py
REPEAT_OK=0
( cd "$LAB_DIR" && exec setsid timeout -k 30 "$REPEAT_SECS" env -u LAB_API_KEY python3 -I -c '
import json, re, sys, time
sys.path.insert(0, "lab")
import lab
prompt = re.search(r"stream_rate\(endpoint, served, \"([^\"]+)\"\)", open("lab/lab.py").read()).group(1)
ep = sys.argv[1]
served = lab.call(ep, "/v1/models", timeout=30)[0]["data"][0]["id"]
tps, total, secs = lab.stream_rate(ep, served, prompt)
print(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prompt": prompt, "tps_first_30s": round(tps, 2),
                  "tokens_total": total, "seconds_total": round(secs, 1), "min_tps": lab.MIN_TPS}))
sys.exit(0 if tps >= lab.MIN_TPS else 1)' "http://127.0.0.1:$OBS_PORT" ) > "$RUN_DIR/speed-repeat.json" 2>&1 & CHILD=$!; CHILD_START=$(pid_start "$CHILD")
wait "$CHILD" && REPEAT_OK=1; CHILD=""; CHILD_START=""
cat "$RUN_DIR/speed-repeat.json"

# --- every chat request: no limits sent, one natural finish_reason each
CHECK_OK=0
timeout -k 5 120 python3 -I "$HERE/check_run.py" "$RUN_DIR/observer" --expect 7 > "$RUN_DIR/check.jsonl" && CHECK_OK=1
tail -n 1 "$RUN_DIR/check.jsonl"

fenced_stop "$OBS_PID" "$OBS_START"; OBS_PID=""
cleanup
CLEAN_OK=0; [ "$(cat "$RUN_DIR/cleanup")" = ok ] && CLEAN_OK=1
echo "{\"lab_six_gates\": $LAB_OK, \"speed_repeat\": $REPEAT_OK, \"seven_natural_ends\": $CHECK_OK, \"cleanup\": $CLEAN_OK}" \
  | tee "$RUN_DIR/verdict.json"
[ "$LAB_OK$REPEAT_OK$CHECK_OK$CLEAN_OK" = 1111 ]
