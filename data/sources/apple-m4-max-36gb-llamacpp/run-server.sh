#!/bin/zsh
# Start the llama-server argv of a registry host launch, from the campaign run dir, with a memory sampler.
# usage: run-server.sh <launch-id> <tag>
set -euo pipefail
LAUNCH=$1; TAG=$2
REPO=$HOME/dev/lar; RUN=$HOME/ai-work/m4max-campaign/run; LOGS=$HOME/ai-work/m4max-campaign/logs/$TAG
mkdir -p "$LOGS"
need=$(python3 -I -c "import json,sys;print(36*1024-json.load(open(sys.argv[1]))['wired_limit_reserve_mb'])" "$REPO/registry/launches/$LAUNCH.json")
have=$(sysctl -n iogpu.wired_limit_mb)
[[ "$have" == "$need" ]] || { echo "iogpu.wired_limit_mb=$have, launch needs $need: run sudo sysctl -w iogpu.wired_limit_mb=$need" >&2; exit 2; }
cd "$RUN"
cmd=("${(@f)$(python3 -I -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['command']))" "$REPO/registry/launches/$LAUNCH.json")}")
{ date; sysctl iogpu.wired_limit_mb hw.memsize; sw_vers; print -r -- "argv: ${cmd[*]}"; } > "$LOGS/host-at-start.txt"
"./${cmd[1]}" "${cmd[2,-1][@]}" > "$LOGS/server.log" 2>&1 &
SPID=$!
echo $SPID > "$LOGS/server.pid"
python3 -I $HOME/ai-work/m4max-campaign/memwatch_llamacpp.py $SPID "$LOGS/memwatch.jsonl" &
echo "server pid $SPID, logs $LOGS"
