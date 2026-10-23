#!/usr/bin/env bash
# Reproduce the GTX 1650 SUPER LFM2.5-2.6B W4A16 vLLM recipe with bounded steps and owned cleanup.
#
#   MODEL_DIR=/path/to/LFM2.5-2.6B-AutoRound-W4A16-2244dd32 LAB_DIR=/path/to/local-ai-registry \
#     bash data/sources/gtx1650super-vllm-autoround/repro.sh
#
# Success (exit 0) needs all of: the lab's six gates (lab.py exits 0), the stock speed prompt repeated at >= 15 tok/s,
# exactly seven chat requests that each end on their own (check_run.py on the observer's raw bytes) with no output or
# thinking limit sent, and the container removed after its log was saved. Anything else exits 1.
#
# Settings (environment): RUNTIME (docker; podman works), GPU_ARGS (default "--gpus all"; WSL without the NVIDIA
# Container Toolkit: "--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib"),
# PORT (server, default 18100; the observer takes PORT+1), LIFETIME (whole run, default 5400 s), HEALTH_SECS (900),
# LAB_SECS (3600), REPEAT_SECS (1200). The image must already be present (pull it first; it is about 22 GB).
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
LIFETIME=${LIFETIME:-5400}
# Cleanup budget, worst case: start delay 12 (a foreground health probe) + helpers 2 x 6 + label check 12 + logs 22
# + rm 22 + final inspect 12 = 92 s. The outer KILL comes 120 s after TERM, so cleanup always runs to its end.
KILL_AFTER=120
if [ -z "${REPRO_INNER:-}" ]; then  # the whole run lives under one hard deadline: TERM at LIFETIME, KILL 120 s later
  exec timeout --kill-after=$KILL_AFTER "$LIFETIME" env REPRO_INNER=1 bash "${BASH_SOURCE[0]}" "$@"
fi

: "${MODEL_DIR:?set MODEL_DIR to the weights directory}" "${LAB_DIR:?set LAB_DIR to a local-ai-registry checkout}"
RUNTIME=${RUNTIME:-docker}; PORT=${PORT:-18100}; OBS_PORT=$((PORT + 1))
HEALTH_SECS=${HEALTH_SECS:-900}; LAB_SECS=${LAB_SECS:-3600}; REPEAT_SECS=${REPEAT_SECS:-1200}
read -ra GPU <<< "${GPU_ARGS:---gpus all}"
IMAGE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
WEIGHTS=plavno/LFM2.5-2.6B-AutoRound-W4A16@2244dd328189f0b58d9b938b576937ad7d216b7f
SERVED=LFM2.5-2.6B-AutoRound-W4A16-2244dd32
LABEL_KEY=io.local-ai-registry.repro

# --- preflight: pins, inputs, free ports, image present
( cd "$MODEL_DIR" && printf '%s\n' \
  "5792b854fb26a4b30e9dc55da1f1465db99b671ea425374511ba577fe9901d3a  model-00001-of-00002.safetensors" \
  "3694b8fee957f0168ddda96c5c7bed60d2d56f52eec11064b5920282baf97d5a  model-00002-of-00002.safetensors" \
  "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef  chat_template.jinja" | sha256sum -c --quiet - )
( cd "$LAB_DIR" && echo "b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca  lab/lab.py" | sha256sum -c --quiet - )
for p in "$PORT" "$OBS_PORT"; do
  if python3 -I -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)' "$p"; then
    echo "port $p is already in use"; exit 1; fi
done
timeout -k 5 60 "$RUNTIME" image inspect "$IMAGE" > /dev/null || { echo "image $IMAGE is not present; pull it first"; exit 1; }

# --- a fresh private run directory; its name is also the container name and label value
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gtx1650-vllm.XXXXXXXX"); chmod 700 "$RUN_DIR"
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

# --- the server: exactly the rendered launch (lab.py render), plus the GPU device flags, name and label
bounded timeout -k 10 300 "$RUNTIME" run -d --cidfile "$CID_FILE" --name "$RUN_ID" --label "$LABEL_KEY=$RUN_ID" \
  "${GPU[@]}" --shm-size 16g -p "127.0.0.1:$PORT:8000" -v "$MODEL_DIR:/models/$SERVED:ro" --entrypoint vllm "$IMAGE" \
  serve "/models/$SERVED" --served-model-name "$SERVED" --host 0.0.0.0 --port 8000 --max-model-len 8192 \
  --max-num-seqs 1 --gpu-memory-utilization 0.78 --kv-cache-dtype auto --enable-auto-tool-choice \
  --tool-call-parser lfm2 --reasoning-parser deepseek_r1 > /dev/null
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
    -u http_proxy -u https_proxy -u all_proxy python3 -I lab/lab.py try "$WEIGHTS" --model lfm2.5-2.6b --engine vllm \
    --card gtx-1650-super-4gb --set ctx=8192 --set mem=0.78 --set tools=lfm2 --set think=deepseek_r1 \
    --on endpoint --endpoint "http://127.0.0.1:$OBS_PORT" --gpu "${GPU_NAME:-NVIDIA GeForce GTX 1650 SUPER}" ) \
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
