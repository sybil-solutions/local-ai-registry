#!/bin/zsh
# Clean-machine gate before every lab run ("pure results", Dan 2026-11-17).
# Refuses to pass unless: wired limit is 32768, no heavy user app is running, no other llama-server/ollama
# model is loaded, AC power. Writes a baseline snapshot into the run's log dir.
# usage: preflight.sh <tag>
set -uo pipefail
TAG=$1; LOGS=$HOME/ai-work/m4max-campaign/logs/$TAG; mkdir -p "$LOGS"; fail=0
[[ $(sysctl -n iogpu.wired_limit_mb) == 32768 ]] || { echo "FAIL wired_limit_mb=$(sysctl -n iogpu.wired_limit_mb)"; fail=1; }
# heavy user apps only: OneDrive's sync daemon, the Teams widget and an idle Docker backend measured <0.15 GiB
# each (2026-11-17) and are allowed; a running Docker VM is not. Chrome is allowed (Dan 2026-11-17, "run it with chrome")
for a in "OneDrive.app/Contents/MacOS/OneDrive" "Discord.app" "Superhuman.app" "Microsoft Teams.app/Contents/MacOS" "Slack.app"; do
  pgrep -f "/Applications/$a" >/dev/null && { echo "FAIL running: $a"; fail=1; }
done
pgrep -f "com.docker.virtualization|com.docker.krun" >/dev/null && { echo "FAIL Docker VM running"; fail=1; }
pgrep -fl "llama-b1[0-9]*/llama-server" && { echo "FAIL a campaign llama-server is still running"; fail=1; }
[[ -z "$(ollama ps 2>/dev/null | sed 1d)" ]] || { echo "FAIL ollama has a model loaded"; fail=1; }
pmset -g batt | grep -q "AC Power" || { echo "FAIL not on AC power"; fail=1; }
{ date; sysctl iogpu.wired_limit_mb vm.swapusage; vm_stat; ps -axo rss=,comm= | sort -rn | head -15; } > "$LOGS/preflight-baseline.txt"
(( fail )) && { echo "PREFLIGHT FAILED ($TAG)"; exit 1; }
echo "PREFLIGHT OK ($TAG): $(sysctl -n vm.swapusage)"
