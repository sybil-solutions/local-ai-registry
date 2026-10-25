#!/usr/bin/env bash
# Reproduce the RTX 4080 Laptop GPU proof of qwen3.5-9b.tabbyapi.128k: the frozen
# tabbyapi-qwen3.5-9b-exl3-4bpw-128k launch in rootless Podman, the unchanged stock lab through a
# pass-through finish-reason recorder, one uncapped repeat of the stock speed request, and a check that
# all seven chat completions ended naturally. Linux, NVIDIA driver, rootless Podman, flock, procps
# (pgrep), Python 3.11+. It changes no GPU, driver, clock, power or OS setting and never touches a process
# or container it did not start.
#
# Files it writes: only under STORE (the Podman storage, the lock and STORE/run-<id>/). The checkout is
# read, never written. The stock lab runs in a scratch copy of lab/ and registry/ at STORE/run-<id>/lab-root,
# so the recipe it rewrites with the new proof and its lab/runs/ output stay there; the run directory keeps
# copies (stock-recipe.json and the run JSON).
#
#   MODEL_DIR=/path/to/Qwen3.5-9B-EXL3-6hb-4bpw STORE=/path/to/roomy/dir \
#     bash data/sources/rtx4080-laptop-tabbyapi-exl3-container/repro.sh
#
# MODEL_DIR holds TheMelonGod/Qwen3.5-9B-exl3@e98453aa143a24fab9eb14b718daed02e2fa6eef: exactly the 13 files
# of weights.manifest, checked by size and hash. The one other entry allowed is a real `.cache` directory
# (what `huggingface-cli download --local-dir` leaves); anything else fails the check. STORE gets the Podman storage, the run lock and one run-<id>/ directory per
# run. Optional: LIBDIR (host driver libraries, default /usr/lib), SERVER_PORT (15000), REC_PORT (15001),
# PYTHON (python3), and the bounds REPRO_TIMEOUT (7200 s), HEALTH_TIMEOUT (900), LAB_TIMEOUT (5400),
# REPEAT_TIMEOUT (1800).
#
# Exit 0 only if the six lab gates, the repeat (>= 15 tok/s) and every postcondition pass AND the server
# log was saved non-empty AND this run's container is positively gone AND its helper processes are gone.
# 1 a lab gate failed; 3 the GPU is not idle, not the measured card, or (on a machine with a battery) not on AC
# power; 4 an input check failed; 5 the
# container did not start or is not verifiably this run's; 6 never healthy; 7 another run holds the
# lock; 8 lab passed but the repeat or a postcondition failed; 9 log saving or cleanup failed;
# 10 SYBIL_TEST_POWER_DIR was set (fixtures only: such a run never exits 0);
# 124/143 the overall bound or a signal ended the run (cleanup still ran to its verdict).
#
# Idle policy, checked before anything starts and again immediately before `podman run`: three samples one
# second apart, each with GPU utilization <= 15 % and memory used <= 1500 MiB (parish-media's thresholds)
# and no compute process; a failed, slow (10 s) or malformed query counts as busy. On a machine with a
# battery, a mains or USB supply must be online. Where `parish-media` is installed, its `status` must also
# exit 0 with ac_online true and not_ready null.
#
# Stages. Every long step (render, weights check, image check, pull and inspect, container start, the lab,
# the repeat, the postconditions) runs in the background under its own `timeout -k K B`, which puts it in
# its own process group, and the script waits for it. A signal interrupts that wait at once; cleanup then
# stops the stage first: TERM to its `timeout` (which passes it on to the stage and its group and KILLs
# them after K s), then KILL to whatever of that process group is left, then a check that nothing of it
# remains. A stage that ends by itself gets the same check. Only `pgrep` exit 1 counts as "empty"; a pgrep
# error or a missing pgrep fails closed (exit 9, the stage kept for cleanup, cleanup ok=0). Only short
# calls run in the foreground; the longest is `parish-media status` (timeout -k 5 20).
#
# Cleanup reserve. On REPRO_TIMEOUT the outer `timeout` sends TERM and allows CLEANUP_RESERVE seconds before
# KILL. Every cleanup step has its own `timeout -k 5 B` (worst case B + 5 s), so the reserve covers the worst
# complete path, with at most two container ids (this run makes one `podman run`; a cidfile id plus a
# label match is the most there can be):
#   either a short foreground call finishing (parish-media 20 + 5) 25, or stopping
#     the running stage (its -k grace, at most 30, + 5, then KILL of its group and
#     up to 5 s for it to empty) 40                                                  40
#   label enumeration (ps 10)                                                       15
#   per id x 2: owner check (inspect 10) 15, log (logs 30) 35, inspect (10) 15,
#     rm -t 10 (25) 30, absence (exists 10, ps 10) 30, kill (10) 15, rm -t 0 (20) 25,
#     absence 30 = 195                                                             390
#   final label check (ps 10)                                                       15
#   helper fence: TERM, 10 s wait, KILL                                             12
#   GPU readback (2 x nvidia-smi 10)                                                30
#   writes and stamps                                                                5
#   total                                                                          507  -> CLEANUP_RESERVE=540
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
: "${REPRO_TIMEOUT:=7200}" "${HEALTH_TIMEOUT:=900}" "${LAB_TIMEOUT:=5400}" "${REPEAT_TIMEOUT:=1800}"
if [ -z "${SYBIL_REPRO_BOUNDED:-}" ]; then  # the whole run, cleanup included, under one bound
  export SYBIL_REPRO_BOUNDED=1
  exec timeout -k 540 "$REPRO_TIMEOUT" bash "$0" "$@"
fi

# Pins (the frozen launch, the stock lab, the measured card)
IMAGE=ghcr.io/0xsero/tabbyapi-exl3@sha256:0f83e6198dc3be2561652d8df2525a7d1a69733e12fc8e25cbc5bc319a8a3ad1
WEIGHTS=TheMelonGod/Qwen3.5-9B-exl3@e98453aa143a24fab9eb14b718daed02e2fa6eef
RECIPE=registry/recipes/nvidia/rtx-4080-laptop-12gb/qwen3.5-9b.tabbyapi.128k.json
LAUNCH_SHA=72198e58f86afcf47cd0b7aae9d2c23dfc5f2af25440b260e2bca6a816e07097
CONFIG_SHA=d3efe710e28fe94a93a81a86bf52f74f95ab6740800890d78b6a4f9db7ee9d0c
LAB_SHA=b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca
GPU_NAME='NVIDIA GeForce RTX 4080 Laptop GPU'
: "${MODEL_DIR:?set MODEL_DIR}" "${STORE:?set STORE}"
: "${LIBDIR:=/usr/lib}" "${SERVER_PORT:=15000}" "${REC_PORT:=15001}" "${PYTHON:=python3}"

mkdir -p "$STORE/tmp" || exit 4
STORE=$(cd "$STORE" && pwd)
export TMPDIR=$STORE/tmp
P=(podman --root "$STORE/root" --runroot "$STORE/run" --tmpdir "$STORE/tmp")
RUNID=$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM
RUN=$STORE/run-$RUNID
mkdir "$RUN" || { echo "cannot create a fresh $RUN" >&2; exit 4; }
trap '' PIPE  # a closed log pipe must not kill cleanup
exec > >(trap '' INT TERM; exec tee -a "$RUN/repro.log") 2>&1  # the logger outlives a TERM so cleanup is logged
exec 7>&1  # the logger again: a signal can arrive inside a call whose output is redirected (a stage's log),
           # and the EXIT trap points stdout and stderr back here first. Closed for every child (7>&-).
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
stamp "run $RUNID in $RUN"

CLEANUP_RESERVE=540
RC=0; CREATED=0; CLEANED=0; IN_CLEANUP=0; SIGNALLED=; REC=; REC_T=; STAGE=; STAGE_T=
t() { local b=$1; shift; timeout -k 5 "$b" "$@" 7>&-; }  # every bounded call: worst case b + 5 s

# Start time of a process (field 22 of /proc/<pid>/stat, read after the command name); fails if it is
# gone or a zombie. Used to signal only processes this run started.
proc_start() {
  local s
  s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  s=${s##*) }
  set -- $s
  case "$1" in Z|X|x) return 1 ;; esac
  echo "${20}"
}
fenced_kill() {
  local pid=$1 want=$2 got i
  [ -n "$pid" ] && [ -n "$want" ] || return 0
  got=$(proc_start "$pid") || return 0
  [ "$got" = "$want" ] || { echo "fence: pid $pid is no longer this run's helper; not signalled"; return 0; }
  kill "$pid" 2>/dev/null
  for i in $(seq 1 50); do
    [ "$(proc_start "$pid")" = "$want" ] || return 0
    sleep 0.2
  done
  kill -9 "$pid" 2>/dev/null; sleep 0.5
  [ "$(proc_start "$pid")" = "$want" ] && { echo "fence: helper $pid survived KILL"; return 1; }
  return 0
}

# stage B K cmd...: run cmd in the background under `timeout -k K B` (its own process group) and wait for it.
# Redirections on the call apply to the stage. A trapped signal interrupts the wait at once; outside cleanup
# the trap ends the run, and cleanup's stop_stage stops the stage before anything else.
stage() {
  local b=$1 k=$2 rc
  shift 2
  timeout -k "$k" "$b" "$@" < /dev/null 7>&- 9>&- &
  STAGE=$!; STAGE_T=$(proc_start "$STAGE")
  while :; do
    wait "$STAGE"; rc=$?
    [ -n "$STAGE_T" ] && [ "$(proc_start "$STAGE")" = "$STAGE_T" ] || break  # still running: the wait was interrupted
  done
  # The stage's `timeout` has exited, but its process group can outlive it: on the stage's own deadline
  # `timeout` exits 124 as soon as its child is gone, before its -k KILL reaches a grandchild that ignores
  # TERM; a finished stage can also leave a background child behind. The identity stays set until the
  # group is confirmed empty, so a failure here still leaves cleanup something to find.
  empty_group "$STAGE" || fail 9 "stage process group $STAGE could not be confirmed empty"
  STAGE=; STAGE_T=
  return "$rc"
}
# Is anything left in process group $1? 0: yes; 1: confirmed empty (pgrep found no match); 2: unknown (pgrep
# failed or is missing). Unknown never counts as empty.
group_state() {
  local rc
  pgrep -g "$1" > /dev/null 2>&1; rc=$?
  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *) stamp "pgrep -g $1 failed (exit $rc): the stage's process group cannot be confirmed empty"; return 2 ;;
  esac
}
# KILL whatever is left in process group $1 and wait (up to 5 s) until it is confirmed empty. Fails (1) if it
# stays non-empty or if the query fails; a failed query sends no KILL, since the group can no longer be seen.
empty_group() {
  local g=$1 i rc
  group_state "$g"; rc=$?
  [ "$rc" -eq 1 ] && return 0
  [ "$rc" -eq 0 ] || return 1
  stamp "stage process group $g outlived its timeout; KILLing what is left"
  kill -KILL -- "-$g" 2>/dev/null
  for i in $(seq 1 25); do
    group_state "$g"; rc=$?
    [ "$rc" -eq 1 ] && return 0
    [ "$rc" -eq 0 ] || return 1
    sleep 0.2
  done
  return 1
}
# Stop the running stage: TERM its `timeout` (it passes TERM to the stage and its process group, then KILLs
# them after its -k grace), wait up to 35 s, KILL anything left in that process group, and require it empty.
# The group id is the stage's `timeout` pid; Linux does not reuse a pid while a process group still uses it.
stop_stage() {
  local pid=$STAGE want=$STAGE_T i
  [ -n "$pid" ] || return 0
  if [ "$(proc_start "$pid")" = "$want" ]; then
    stamp "stopping the running stage (timeout pid $pid)"
    kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 175); do
      [ "$(proc_start "$pid")" = "$want" ] || break
      sleep 0.2
    done
  fi
  empty_group "$pid" || { stamp "stage process group $pid could not be confirmed empty"; return 1; }
  wait "$pid" 2>/dev/null
  STAGE=; STAGE_T=
  return 0
}

# A container id is this run's only if podman reports exactly that id with this run's unique label.
is_owned() {
  local got
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || return 1
  got=$(t 10 "${P[@]}" inspect --format '{{.Id}} {{index .Config.Labels "sybil.run"}}' "$1" 2>/dev/null) || return 1
  [ "$got" = "$1 $RUNID" ]
}
# Positively gone: `container exists` answers "no" (exit 1, not an error or a timeout) and no container
# carries the label. Anything unknown is "not gone".
gone() {
  local rc left
  t 10 "${P[@]}" container exists "$1"; rc=$?
  [ "$rc" -eq 1 ] || return 1
  left=$(t 10 "${P[@]}" ps -a -q --filter "label=sybil.run=$RUNID") || return 1
  [ -z "$left" ]
}
save_logs() {  # $1 = owned id; success means podman exited 0 and the log is not empty
  t 30 "${P[@]}" logs --timestamps "$1" > "$RUN/server.log" 2>&1 && [ -s "$RUN/server.log" ] \
    || { stamp "server log NOT saved (or empty) for ${1:0:12}"; return 1; }
  t 10 "${P[@]}" inspect "$1" > "$RUN/container-inspect.json" 2>&1
  stamp "server log saved: $(wc -l < "$RUN/server.log") lines"
}
remove_owned() {  # always attempts removal of an owned id; escalates rm -> kill -> rm; requires positive absence
  local cid=$1
  t 25 "${P[@]}" rm -f -t 10 "$cid" > /dev/null 2>&1
  gone "$cid" && return 0
  stamp "rm did not remove ${cid:0:12}; escalating"
  t 10 "${P[@]}" kill "$cid" > /dev/null 2>&1
  t 20 "${P[@]}" rm -f -t 0 "$cid" > /dev/null 2>&1
  gone "$cid"
}

# On a machine with a battery, a mains or USB supply must be online. SYBIL_TEST_POWER_DIR (fixtures only)
# replaces /sys/class/power_supply and makes the run unable to exit 0.
on_ac() {
  local d=${SYBIL_TEST_POWER_DIR:-/sys/class/power_supply} s ty bat=0 on=0
  [ -n "${SYBIL_TEST_POWER_DIR:-}" ] && stamp "WARNING: SYBIL_TEST_POWER_DIR=$d; this run cannot qualify"
  [ -d "$d" ] || { [ -z "${SYBIL_TEST_POWER_DIR:-}" ] && return 0; return 1; }  # no power-supply class: no battery
  for s in "$d"/*; do
    ty=$(cat "$s/type" 2>/dev/null)
    case "$ty" in
      Battery) bat=1 ;;
      Mains|USB) [ "$(cat "$s/online" 2>/dev/null)" = 1 ] && on=1 ;;
    esac
  done
  [ "$bat" = 0 ] || [ "$on" = 1 ]
}
parish_media_ready() {  # only where parish-media is installed; same fail-closed reading as its own guard
  local pm out
  pm=$(command -v parish-media 2>/dev/null) || { [ -x "$HOME/.local/bin/parish-media" ] && pm=$HOME/.local/bin/parish-media; }
  [ -n "${pm:-}" ] || return 0
  out=$(t 20 "$pm" status 2>/dev/null) || { stamp "parish-media status failed"; return 1; }
  printf '%s' "$out" | "$PYTHON" -I -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(d, dict) and d.get("ac_online") is True and "not_ready" in d and d["not_ready"] is None else 1)' \
    || { stamp "parish-media not ready: $(printf '%s' "$out" | tr -s ' \n' ' ' | cut -c1-400)"; return 1; }
}
gpu_idle() {
  local i out apps
  for i in 1 2 3; do
    out=$(t 10 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1) \
      || { stamp "GPU utilization query failed or timed out: $out"; return 1; }
    [[ "$out" =~ ^[[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*$ ]] || { stamp "GPU utilization query malformed: '$out'"; return 1; }
    [ "${BASH_REMATCH[1]}" -le 15 ] && [ "${BASH_REMATCH[2]}" -le 1500 ] \
      || { stamp "GPU busy: ${BASH_REMATCH[1]}% utilization, ${BASH_REMATCH[2]} MiB used"; return 1; }
    apps=$(t 10 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) \
      || { stamp "GPU compute query failed or timed out: $apps"; return 1; }
    [ -z "$apps" ] || { stamp "GPU compute process present: $apps"; return 1; }
    [ "$i" -lt 3 ] && sleep 1
  done
  return 0
}
idle_or_fail() {  # $1 = where in the run
  on_ac || fail 3 "$1: not on AC power"
  parish_media_ready || fail 3 "$1: parish-media guard not ready"
  gpu_idle || fail 3 "$1: GPU not idle"
  stamp "$1: GPU idle, power ok"
}

cleanup() {
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  local ok=1 cid left ids=() t0=$SECONDS
  stamp "cleanup (reserve ${CLEANUP_RESERVE}s)"
  stop_stage || ok=0
  if [ "$CREATED" = 1 ]; then
    [ -s "$RUN/container.cid" ] && ids+=("$(cat "$RUN/container.cid")")
    # a create that never wrote its cidfile still carries this run's unique label
    for cid in $(t 10 "${P[@]}" ps -a -q --no-trunc --filter "label=sybil.run=$RUNID" 2>/dev/null); do
      [[ " ${ids[*]} " == *" $cid "* ]] || ids+=("$cid")
    done
    if [ "${#ids[@]}" -gt 2 ]; then stamp "more than two ids carry this run's label: ${ids[*]}"; ok=0; fi
    for cid in "${ids[@]:0:2}"; do  # the reserve covers two
      if ! is_owned "$cid"; then
        if gone "$cid"; then continue; fi
        stamp "container ${cid:0:12} is not verifiably this run's: left untouched"; ok=0; continue
      fi
      save_logs "$cid" || ok=0
      if remove_owned "$cid"; then stamp "container ${cid:0:12} removed (positively gone)"; else stamp "container ${cid:0:12} NOT removed"; ok=0; fi
    done
    left=$(t 10 "${P[@]}" ps -a -q --filter "label=sybil.run=$RUNID" 2>&1) && [ -z "$left" ] \
      || { stamp "a container with this run's label remains (or podman ps failed): $left"; ok=0; }
  fi
  fenced_kill "$REC" "$REC_T" || ok=0
  [ -n "$REC" ] && [ "$(proc_start "$REC")" = "$REC_T" ] && { stamp "recorder still running"; ok=0; }
  { date -u +%FT%TZ; t 10 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    t 10 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$RUN/cleanup-readback.txt" 2>&1
  echo "$ok" > "$RUN/cleanup.ok"
  if [ "$RC" -eq 0 ] && [ "$ok" != 1 ]; then RC=9; fi
  if [ "$RC" -eq 0 ] && [ -n "${SYBIL_TEST_POWER_DIR:-}" ]; then RC=10; fi
  if [ -n "$SIGNALLED" ] && [ "$RC" -eq 0 ]; then RC=$SIGNALLED; fi  # a deadline during cleanup still fails the run
  stamp "cleanup verdict: ok=$ok after $((SECONDS - t0))s of ${CLEANUP_RESERVE}s; exit $RC"
}
# A signal outside cleanup ends the run, and the EXIT trap then cleans up. A signal while cleanup is running
# (the overall deadline can land in it) is recorded and cleanup carries on to its verdict: the outer
# timeout's 540 s KILL reserve is what bounds it. The handler sets RC itself (keeping an earlier failure),
# so a signal at any point of cleanup, even after the verdict line, makes the process exit non-zero. The
# verdict line's text is expanded before such a late signal and can still say "exit 0"; the appended
# "signal ... during cleanup" record and the process exit status are what count.
on_signal() {
  if [ "$IN_CLEANUP" = 1 ]; then
    SIGNALLED=${SIGNALLED:-$1}
    [ "$RC" -eq 0 ] && RC=$1
    # straight to the run log: bash can run this inside the interrupted command's redirections (server.log)
    stamp "signal $(( $1 - 128 )) during cleanup at $(date +%s.%N): deferred; cleanup continues" >> "$RUN/repro.log" 2>&1
    return 0
  fi
  RC=$1; exit "$1"
}
trap 'S=$?; exec 1>&7 2>&7; IN_CLEANUP=1; [ "$RC" -eq 0 ] && RC=$S; cleanup; exit $RC' EXIT  # an unexpected exit keeps its status
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
fail() { RC=$1; shift; stamp "$*"; exit "$RC"; }

# 0. One run per STORE (fd 9 is closed for every child so no helper or container keeps the lock)
exec 9> "$STORE/repro.lock"
flock -n 9 || fail 7 "another run holds $STORE/repro.lock"

# 1. Inputs
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || fail 4 "Python 3.11 or newer is required"
command -v pgrep > /dev/null || fail 4 "pgrep (procps) is required: stage process groups are checked with it"
[ "$(sha256sum "$ROOT/lab/lab.py" | cut -d' ' -f1)" = "$LAB_SHA" ] || fail 4 "lab/lab.py is not the stock lab"
stage 60 5 "$PYTHON" -I -B -c '
import json, sys
sys.path.insert(0, sys.argv[4] + "/lab"); import lab  # -B: importing from the checkout writes no __pycache__ there
L = lab.render(json.load(open(sys.argv[1])))
json.dump(L, open(sys.argv[2], "w"), indent=1, sort_keys=True)
open(sys.argv[3], "w").write(L["config"]["text"])
print(lab.launch_hash(L))' "$ROOT/$RECIPE" "$RUN/launch.json" "$RUN/config.yml" "$ROOT" > "$RUN/launch.sha256" || fail 4 "cannot render $RECIPE"
[ "$(cat "$RUN/launch.sha256")" = "$LAUNCH_SHA" ] || fail 4 "rendered launch $(cat "$RUN/launch.sha256") is not the measured one"
[ "$(sha256sum "$RUN/config.yml" | cut -d' ' -f1)" = "$CONFIG_SHA" ] || fail 4 "config file is not the measured one"
stage 900 10 "$PYTHON" -I "$HERE/verify_weights.py" "$HERE/weights.manifest" "$MODEL_DIR" > "$RUN/weights-verify.txt" \
  || { cat "$RUN/weights-verify.txt"; fail 4 "weights do not match weights.manifest"; }
for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do [ -c "$d" ] || fail 4 "missing $d"; done
NAME_SEEN=$(t 10 nvidia-smi --query-gpu=name --format=csv,noheader 9>&-) || fail 3 "nvidia-smi failed"
[ "$NAME_SEEN" = "$GPU_NAME" ] || fail 3 "GPU is '$NAME_SEEN', not the measured '$GPU_NAME'"
DRIVER=$(t 10 nvidia-smi --query-gpu=driver_version --format=csv,noheader 9>&-) || fail 3 "nvidia-smi failed"
idle_or_fail "preflight" 9>&-
LIBS=()
for l in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4 "libnvidia-gpucomp.so.$DRIVER"; do
  f=$(readlink -f "$LIBDIR/$l" 2>/dev/null) && [ -f "$f" ] || fail 4 "host driver library $LIBDIR/$l does not resolve to a file"
  echo "$l -> $f $(sha256sum "$f" | cut -d' ' -f1)" >> "$RUN/driver-libs.txt"
  LIBS+=(-v "$f:/usr/local/nvidia/lib64/$l:ro")
done
for port in "$SERVER_PORT" "$REC_PORT"; do
  "$PYTHON" -c 'import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) else 1)' "$port" \
    || fail 4 "port $port is in use"
done
if ! stage 60 5 "${P[@]}" image exists "$IMAGE"; then
  stamp "pulling $IMAGE"
  stage 3000 10 "${P[@]}" pull "$IMAGE" || fail 4 "pull failed"
fi
stage 60 5 "${P[@]}" image inspect "$IMAGE" --format '{{.Digest}}' > "$RUN/image-digest.txt" || fail 4 "image inspect failed"
[ "$(cat "$RUN/image-digest.txt")" = "sha256:${IMAGE##*@sha256:}" ] || fail 4 "image digest mismatch"
{ date -u +%FT%TZ; free -b; t 10 nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv; } > "$RUN/baseline.txt" 2>&1 9>&-

# 2. Recorder in front of the server: forwards unchanged, logs every finish reason and raw body
"$PYTHON" -I "$HERE/finish_recorder.py" "$REC_PORT" "$SERVER_PORT" "$RUN/recorder" < /dev/null 7>&- 9>&- & REC=$!; REC_T=$(proc_start $REC)
[ -n "$REC_T" ] || fail 4 "recorder did not start"

# 3. Server: the rendered launch's image, entrypoint, args, env, shm, weights mount and config file; the
# GPU through its device nodes and the host driver's libraries (read-only), as measured
idle_or_fail "launch" 9>&-  # immediately before the container: the pull above can take long
stamp "starting container"
CREATED=1
stage 120 10 "${P[@]}" run -d --name "sybil-repro-$RUNID" --cidfile "$RUN/container.cid" --label "sybil.run=$RUNID" \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools "${LIBS[@]}" \
  -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8g -p "127.0.0.1:$SERVER_PORT:5000" \
  -v "$MODEL_DIR:/workspace/models/Qwen3.5-9B-EXL3-6hb-4bpw:ro" -v "$RUN/config.yml:/app/config.yml:ro" \
  --entrypoint /opt/venv/bin/python3 "$IMAGE" main.py --config /app/config.yml > /dev/null || fail 5 "podman run failed"
CID=$(cat "$RUN/container.cid" 2>/dev/null)
is_owned "$CID" || fail 5 "the started container is not verifiably this run's"
stamp "container $CID"

DEADLINE=$((SECONDS + HEALTH_TIMEOUT)); READY=no
while [ $SECONDS -lt $DEADLINE ]; do
  if timeout -k 2 5 curl --max-time 4 -sf "http://127.0.0.1:$SERVER_PORT/v1/models" > "$RUN/models.json"; then READY=yes; break; fi
  [ "$(t 10 "${P[@]}" inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" = true ] || { stamp "server exited during load"; break; }
  sleep 5
done
[ "$READY" = yes ] || fail 6 "server not healthy within ${HEALTH_TIMEOUT}s"
stamp "healthy"

# 4. The stock lab, unchanged, through the recorder, in a scratch copy of lab/ and registry/ so the checkout
# stays untouched (the lab rewrites the recipe with its new proof and writes lab/runs/)
LR=$RUN/lab-root
mkdir -p "$LR/lab/runs" || fail 4 "cannot make $LR"
(cd "$ROOT" && tar --exclude=./runs --exclude=__pycache__ -C lab -cf - . | tar -C "$LR/lab" -xf - && cp -a registry "$LR/registry") \
  || fail 4 "cannot copy lab/ and registry/ to $LR"
[ "$(sha256sum "$LR/lab/lab.py" | cut -d' ' -f1)" = "$LAB_SHA" ] || fail 4 "the scratch lab/lab.py is not the stock lab"
cmp -s "$ROOT/$RECIPE" "$LR/$RECIPE" || fail 4 "the scratch recipe differs from the checkout's"
cd "$LR"
stamp "lab try (in $LR)"
stage "$LAB_TIMEOUT" 30 "$PYTHON" lab/lab.py try "$WEIGHTS" --model qwen3.5-9b --engine tabbyapi-qwen3.5-9b-exl3-4bpw-128k \
  --card rtx-4080-laptop-12gb --on endpoint --endpoint "http://127.0.0.1:$REC_PORT" --gpu "$GPU_NAME" > "$RUN/lab-try.log" 2>&1
LAB=$?
cat "$RUN/lab-try.log"
echo "$LAB" > "$RUN/lab.exit"
stamp "lab exit $LAB"
find lab/runs -maxdepth 1 -name 'rtx-4080-laptop-12gb.*' -newer "$RUN/config.yml" -exec cp -p {} "$RUN/" \;
cmp -s "$ROOT/$RECIPE" "$LR/$RECIPE" || cp -p "$LR/$RECIPE" "$RUN/stock-recipe.json"

# 5. One independent repeat of the stock speed request with the lab's own stream_rate, uncapped
if [ "$LAB" -eq 0 ]; then
  stage "$REPEAT_TIMEOUT" 30 "$PYTHON" -c '
import json, sys, time
sys.path.insert(0, "lab"); import lab
ep = sys.argv[1]
served = lab.call(ep, "/v1/models", timeout=30)[0]["data"][0]["id"]
t = time.time()
tps, total, secs = lab.stream_rate(ep, served, "Write a detailed 600-word story about a lighthouse keeper.")
print(json.dumps({"at": t, "served": served, "tps": round(tps, 1), "tokens_total": total, "seconds_total": round(secs, 1), "min_tps": lab.MIN_TPS, "pass": tps >= lab.MIN_TPS}))
' "http://127.0.0.1:$REC_PORT" > "$RUN/speed-repeat.json" 2> "$RUN/speed-repeat.err"
  echo $? > "$RUN/repeat.exit"
  stamp "repeat exit $(cat "$RUN/repeat.exit"): $(cat "$RUN/speed-repeat.json")"
fi

# 6. Postconditions: exactly 7 chat completions, HTTP 200, no caps, every one ended naturally. Before the
# recorder stops, wait (at most 10 s) for its seventh chat row; its TERM handler then stops taking requests
# and lets in-flight handlers finish writing their rows before it exits.
if [ "$LAB" -eq 0 ]; then
  for i in $(seq 1 50); do
    [ "$(grep -c '"method": "POST", "path": "/v1/chat/completions"' "$RUN/recorder.jsonl" 2>/dev/null)" -ge 7 ] && break
    sleep 0.2
  done
fi
fenced_kill "$REC" "$REC_T"
stage 60 5 "$PYTHON" -I "$HERE/check_postconditions.py" "$RUN" > "$RUN/postconditions.out" 2>&1; PC=$?
stamp "postconditions exit $PC"
[ "$LAB" -eq 0 ] || fail 1 "a lab gate failed"
[ "$PC" -eq 0 ] || fail 8 "the repeat or a postcondition failed: $(grep -A3 failures "$RUN/postconditions.out" | tr -s ' \n' ' ')"
stamp "all gates, the repeat and every postcondition passed"
exit 0
