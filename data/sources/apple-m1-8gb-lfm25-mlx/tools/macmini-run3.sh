#!/bin/bash
# Apple M1 8 GB qualification run v3, run on the target from its task directory (the parent of the tools/ directory holding
# this script). The task directory is built from a clean checkout:
#   tools/                      data/sources/apple-m1-8gb-lfm25-mlx/tools/, as committed (this script among them)
#   runtime/                    data/sources/apple-m1-8gb-lfm25-mlx/runtime/, as committed (weights manifest, patch, regression)
#   sdk-setup.sh, sdk-start.argv  generated from the SDK for this launch (README: Reproduction)
#   labtree/                    the registry checkout itself (`lab.py try` runs from it)
# Everything the run makes (rt312/, rt312-bundle/, runs/) is created inside the task directory.
# Every task process runs under groupwatch (identity-tracked process tree, lifetime cap); the host's own services are only
# observed. The runtime and weights come from the registry SDK's own rendered setup steps (sdk-setup.sh, an exact prefix of
# the SDK's one-script command for this launch); the server runs the SDK's exact start argv (sdk-start.argv).
#   macmini-run3.sh setup TAG    run sdk-setup.sh (runtime, packages, weights, launcher), then verify what it made
#   macmini-run3.sh verify TAG   verify the runtime, launcher and weights again (writes runs/TAG/verify.txt)
#   macmini-run3.sh start TAG    memwatch, snapshot loop, the server (memory guard), and the byte-transparent observer
#   macmini-run3.sh lab TAG      the registry lab, --on endpoint, through the observer
#   macmini-run3.sh probe TAG    tools/probe_m1.py (sampler, prompt limit, back-to-back near-limit prompts, a long answer)
#                                behind a second observer on 8082
#   macmini-run3.sh long TAG     tools/long_answer.py (a near-limit prompt, then a short and a long streamed answer) behind
#                                a second observer on 8082
#   macmini-run3.sh stop TAG     tear down only this run's recorded processes; final snapshot
#   macmini-run3.sh sampler TAG  after stop: tools/seedprobe.py, then a stock (unpatched) mlx_lm.server on 8083 for
#                                tools/pristine_sampler.py, under the same memory guard, stopped when it is done
#   macmini-run3.sh snap TAG X   one service/thermal snapshot labelled X
# The host's own services to observe come from the environment, so no host detail is in this file:
#   OBSERVE_SERVICES="label ..." (launchd gui/<uid> service labels)   OBSERVE_HTTP=<url answering HTTP while they run>
set -u
D=$(cd "$(dirname "$0")/.." && pwd -P); T=$2; L=$D/runs/$T; G="/usr/bin/python3 -I $D/tools/groupwatch.py"; U=$(id -u)
EP=http://127.0.0.1:8081   # the observer, in front of the server on 8080
MODEL=LiquidAI/LFM2.5-2.6B-MLX-4bit; REV=04efa23776ce61ec34ec95ec34c859854c89542b
ENGINE=mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1
snap() { { echo "== $(date -u +%FT%TZ) $1"
  for s in ${OBSERVE_SERVICES:-}; do launchctl print gui/$U/$s 2>/dev/null | grep -E "^\s+(state|pid|last exit code|runs) =" | tr -s ' ' | tr '\n' ';'; echo " <- $s"; done
  uptime; pmset -g therm 2>&1 | grep -vE "^$|Note:" | tr '\n' ';'; echo; pmset -g batt | head -1
  [ -z "${OBSERVE_HTTP:-}" ] || curl -s -o /dev/null -w "service_http=%{http_code}\n" --max-time 3 "$OBSERVE_HTTP"; } >> $L/services.txt 2>&1; }
host_state() { { date -u +%FT%TZ; sysctl hw.model hw.memsize iogpu.wired_limit_mb vm.swapusage; vm_stat; uptime; pmset -g therm; pmset -g batt
  ioreg -r -d 1 -c IOAccelerator | grep -oE '"(Device Utilization %|In use system memory)"=[0-9]+'
  for port in 8080 8081 8082 8083; do lsof -nP -iTCP:$port -sTCP:LISTEN || echo "port $port free"; done
  pgrep -fl "mlx_lm|lab/lab.py|groupwatch.py|memwatch_native|observer.py|probe_m1.py|long_answer.py|seedprobe.py|pristine_sampler.py|llama-server" || echo "no task processes"; } > "$1" 2>&1; }
macos_guard() {  # the mlx and mlx-metal wheels are tagged macosx_15_0: stop before any download on an older macOS
  v=$(sw_vers -productVersion 2>/dev/null); major=${v%%.*}
  case $major in ''|*[!0-9]*) echo "ABORT: cannot read the macOS version (sw_vers -productVersion gave '$v')" >&2; exit 3;; esac
  if [ "$major" -lt 15 ]; then echo "ABORT: macOS $v; this recipe needs macOS 15 or later (its mlx and mlx-metal wheels are tagged macosx_15_0)" >&2; exit 3; fi
  echo "macOS $v: at least 15, ok"; }
case $1 in
setup)
  macos_guard
  mkdir -p $L; : > $L/services.txt; snap before-setup; host_state $L/before-setup.txt
  cd $D && $G run --state $L/gw-setup --max-seconds 5400 -- /bin/bash -e $D/sdk-setup.sh > $L/setup.log 2>&1; echo "setup exit $?" >> $L/setup.log
  tail -3 $L/setup.log; bash $D/tools/macmini-run3.sh verify $T ;;
verify)
  mkdir -p $L; cd $D
  { echo "== interpreter"; rt312/python/bin/python3.12 -I -c 'import sys, platform; print(sys.version.split()[0], platform.mac_ver()[0], platform.machine())'
    echo "== installed distributions"; rt312/python/bin/python3.12 -I -c 'import sys; sys.path.insert(0, "rt312/site"); import importlib.metadata as m; print(len(list(m.distributions(path=["rt312/site"]))), "in rt312/site"); [print(f"{p}=={m.version(p)}") for p in ("mlx-lm", "mlx", "mlx-metal", "transformers", "huggingface_hub")]'
    echo "== launcher"; shasum -a 256 rt312/mlx_lm_unlimited_serve.py
    PYTHONPATH=rt312/site rt312/python/bin/python3.12 -s rt312/mlx_lm_unlimited_serve.py --verify-only; echo "verify-only exit $?"
    echo "== weights (the pinned snapshot, every file against the manifest)"; m=rt312/hf-cache/models--LiquidAI--LFM2.5-2.6B-MLX-4bit
    echo "refs/main: $(cat $m/refs/main)"; ls $m/snapshots
    (cd $m/snapshots/$REV && ls -A | sort > /tmp/.sybil-snap.$$ && cut -c67- $D/runtime/weights-04efa237.sha256 | sort | diff - /tmp/.sybil-snap.$$ && echo "file set = manifest ($(wc -l < /tmp/.sybil-snap.$$ | tr -d ' ') files)"; rm -f /tmp/.sybil-snap.$$; shasum -a 256 -c $D/runtime/weights-04efa237.sha256)
    echo "== launcher patch regression (CPU only; its stdout and stderr, merged, are also in regression-cpu.txt)"; P=$(mktemp -d $D/.regression.XXXXXX)
    mkdir $P/models && cp rt312/site/mlx_lm/server.py rt312/site/mlx_lm/utils.py $P/ && cp rt312/site/mlx_lm/models/cache.py $P/models/ \
      && patch -s -o $P/server.patched.py $P/server.py < $D/runtime/mlx_lm-0.31.3-unlimited-max-tokens.patch && shasum -a 256 $P/server.py $P/server.patched.py | sed "s#$P/##"
    rt312/python/bin/python3.12 -I $D/runtime/run_regression_cpu.py $P/server.py $P/server.patched.py > $L/regression-cpu.txt 2>&1; echo "regression exit $?"
    cat $L/regression-cpu.txt; rm -rf $P
    echo "== disk"; du -sh rt312 rt312-bundle 2>/dev/null; } > $L/verify.txt 2>&1
  tail -25 $L/verify.txt; host_state $L/after-setup.txt; snap after-setup ;;
start)
  mkdir -p $L; snap before; host_state $L/baseline.txt
  nohup $G run --state $L/gw-memwatch --max-seconds 9000 -- /usr/bin/python3 -I $D/tools/memwatch_native.py $L/memwatch.csv > /dev/null 2>&1 &
  nohup $G run --state $L/gw-snapper --max-seconds 9000 -- /bin/bash -c "while [ ! -f $L/snapper.stop ]; do sleep 30; [ -f $L/snapper.stop ] || bash $D/tools/macmini-run3.sh snap $T during; done" > /dev/null 2>&1 &
  sleep 2; cd $D
  # the SDK's start step, exactly: env assignments then the command (one argv per line in sdk-start.argv)
  argv=(); while IFS= read -r a; do argv+=("$a"); done < $D/sdk-start.argv
  nohup $G run --state $L/gw-server --max-seconds 8000 --min-free-pct 10 -- /usr/bin/env "${argv[@]}" > $L/server.log 2>&1 &
  up=0; for i in $(seq 1 90); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/health)" = 200 ] && { echo "healthy ~$((i*2))s"; up=1; break; }; sleep 2; done
  if [ "$up" != 1 ]; then echo "server did not become healthy; stopping this run's helpers" >&2
    touch $L/snapper.stop; for g in gw-server gw-memwatch gw-snapper; do [ -d $L/$g ] && $G stop --state $L/$g 2>&1 | tail -1; done; snap after-failed-start; exit 1; fi
  ps -o args= -p $(cat $L/gw-server/child.pid) > $L/server-argv.txt; cat $L/server-argv.txt
  nohup $G run --state $L/gw-proxy --max-seconds 8000 -- /usr/bin/python3 -I $D/tools/observer.py 127.0.0.1 8081 127.0.0.1 8080 $L/observer > $L/proxy.log 2>&1 &
  for i in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 $EP/health)" = 200 ] && { echo "proxy up"; break; }; sleep 1; done ;;
probe)
  P2=http://127.0.0.1:8082
  nohup $G run --state $L/gw-proxy-probe --max-seconds 6000 -- /usr/bin/python3 -I $D/tools/observer.py 127.0.0.1 8082 127.0.0.1 8080 $L/observer-probe > $L/proxy-probe.log 2>&1 &
  for i in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 $P2/health)" = 200 ] && { echo "probe proxy up"; break; }; sleep 1; done
  $G run --state $L/gw-probe --max-seconds 5400 -- /usr/bin/python3 -I $D/tools/probe_m1.py $P2 $MODEL $L/probe.json > $L/probe.log 2>&1; echo "probe exit $?" >> $L/probe.log
  $G stop --state $L/gw-proxy-probe 2>&1 | tail -1; cat $L/probe.log ;;
long)
  P2=http://127.0.0.1:8082
  nohup $G run --state $L/gw-proxy-long --max-seconds 4000 -- /usr/bin/python3 -I $D/tools/observer.py 127.0.0.1 8082 127.0.0.1 8080 $L/observer-long > $L/proxy-long.log 2>&1 &
  for i in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 $P2/health)" = 200 ] && { echo "long proxy up"; break; }; sleep 1; done
  $G run --state $L/gw-long --max-seconds 3300 -- /usr/bin/python3 -I $D/tools/long_answer.py $P2 $MODEL $L/long-answer.json > $L/long.log 2>&1; echo "long exit $?" >> $L/long.log
  $G stop --state $L/gw-proxy-long 2>&1 | tail -1; cat $L/long.log ;;
snap) mkdir -p $L; snap $3 ;;
sampler)
  cd $D; export HF_HUB_OFFLINE=1 HF_HUB_CACHE=rt312/hf-cache PYTHONPATH=rt312/site
  $G run --state $L/gw-seedprobe --max-seconds 600 -- rt312/python/bin/python3.12 -s $D/tools/seedprobe.py $L/seedprobe.json > $L/seedprobe.log 2>&1; echo "seedprobe exit $?" >> $L/seedprobe.log
  nohup $G run --state $L/gw-pristine --max-seconds 1800 --min-free-pct 10 -- rt312/python/bin/python3.12 -s -m mlx_lm.server --model $MODEL --host 127.0.0.1 --port 8083 > $L/pristine-server.log 2>&1 &
  up=0; for i in $(seq 1 90); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8083/health)" = 200 ] && { up=1; break; }; sleep 2; done
  [ "$up" = 1 ] && $G run --state $L/gw-pristine-probe --max-seconds 1500 -- /usr/bin/python3 -I $D/tools/pristine_sampler.py http://127.0.0.1:8083 $MODEL $L/pristine-sampler.json > $L/pristine-sampler.log 2>&1
  echo "pristine sampler exit $? (server healthy: $up)" >> $L/pristine-sampler.log
  $G stop --state $L/gw-pristine 2>&1 | tail -1; host_state $L/after-sampler.txt; snap after-sampler; cat $L/seedprobe.log $L/pristine-sampler.log ;;
lab)
  cd $D/labtree && $G run --state $L/gw-lab --max-seconds 3000 -- /usr/bin/env -u LAB_API_KEY -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    /usr/bin/python3 -I lab/lab.py try $MODEL@$REV --model lfm2.5-2.6b --engine $ENGINE \
    --card apple-m1-8gb --on endpoint --endpoint $EP --gpu "Apple M1 8GB (8-core GPU, Metal)" > $L/lab-try.log 2>&1; echo "lab exit $?" >> $L/lab-try.log; tail -3 $L/lab-try.log ;;
stop)
  touch $L/snapper.stop; for i in $(seq 1 40); do [ -f $L/gw-snapper/groupwatch.log ] && grep -q "EXIT\|ABORT" $L/gw-snapper/groupwatch.log && break; sleep 1; done
  for g in gw-server gw-proxy gw-proxy-probe gw-proxy-long gw-memwatch gw-snapper; do [ -d $L/$g ] && $G stop --state $L/$g 2>&1 | tail -1; done
  snap after; host_state $L/after.txt; tail -4 $L/after.txt; tail -4 $L/services.txt ;;
esac
