#!/bin/bash
# AMD-SMI supplies telemetry for ROCm and Vulkan cards. A card registered for Vulkan must be exposed as amd-vulkan,
# with the same render node, so a Vulkan recipe selects it and Docker passes only /dev/dri (never /dev/kfd).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export HOME=$TMP/home XDG_RUNTIME_DIR=$TMP/run
mkdir -p "$HOME" "$TMP/bin" "$TMP/prefix/bin" "$TMP/prefix/share/local-ai"
cp "$ROOT/core/local-ai" "$TMP/prefix/bin/"
CLI=$TMP/prefix/bin/local-ai
DATA=$TMP/prefix/share/local-ai
ID=test-vulkan
PIN=ghcr.io/x/engine@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
shim() { printf '#!/bin/bash\n%s\n' "$2" >"$TMP/bin/$1"; chmod +x "$TMP/bin/$1"; }
shim nvidia-smi 'exit 0'
shim lspci 'exit 0'
shim amd-smi '
if [[ $1 == static ]]; then
  printf "%s" '\''{"gpu_data":[{"gpu":0,"asic":{"market_name":"AMD Radeon RX 9060 XT"},"bus":{"bdf":"0000:03:00.0"},"vram":{"size":{"value":16304}}},{"gpu":1,"asic":{"market_name":"AMD Radeon RX 9070 XT"},"bus":{"bdf":"0000:09:00.0"},"vram":{"size":{"value":16304}}}]}'\''
else
  printf "%s" '\''{"gpu_data":[{"gpu":0,"mem_usage":{"used_vram":{"value":100}},"temperature":{"edge":{"value":40}}},{"gpu":1,"mem_usage":{"used_vram":{"value":200}},"temperature":{"edge":{"value":41}}}]}'\''
fi'
jq -nc --arg id "$ID" --arg img "$PIN" '{schemaVersion:"omarchy-local-ai/recipes/2",registryCommit:("d"*40),
  gateway:{image:("ghcr.io/x/gateway@sha256:"+("b"*64))},hardware:{
    "rx-9060-xt-16gb":{match:{backend:"amd-vulkan",vramGb:16,names:["rx9060xt"]},recipes:[{id:$id,name:"Test",family:"qwen",format:"GGUF",sizeGb:1,cards:1,image:$img,servedName:"test",weights:[],launch:{arguments:[],environment:{},port:8080},serving:{ctxTokens:32768},capabilities:{}}]},
    "rx-9070-xt-16gb":{match:{backend:"amd-rocm",vramGb:16,names:["rx9070xt"]},recipes:[]}
  }}' >"$DATA/recipes.json"
export PATH=$TMP/bin:/usr/bin:/bin
"$CLI" snapshot >"$TMP/snapshot.json"
jq -e '(.gpus | map(del(.busy))) == [{backend:"amd-vulkan",index:0,product:"AMD Radeon RX 9060 XT",totalMiB:16304,usedMiB:100,tempC:40,node:"/dev/dri/by-path/pci-0000:03:00.0-render",key:"amd-vulkan:0",hw:"rx-9060-xt-16gb",vramGb:16,name:"RX 9060 XT"},
  {backend:"amd-rocm",index:1,product:"AMD Radeon RX 9070 XT",totalMiB:16304,usedMiB:200,tempC:41,node:"/dev/dri/by-path/pci-0000:09:00.0-render",key:"amd-rocm:1",hw:"rx-9070-xt-16gb",vramGb:16,name:"RX 9070 XT"}]' "$TMP/snapshot.json" >/dev/null || {
  echo "not ok - AMD Vulkan and ROCm cards keep separate backends" >&2; jq . "$TMP/snapshot.json" >&2; exit 1; }
echo "ok - AMD Vulkan and ROCm cards keep separate backends"
