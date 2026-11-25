#!/usr/bin/env bash
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
rm -rf /tmp/src && mkdir -p /tmp/src && cp -r /src/csrc /src/scripts /src/exl3xpu /tmp/src/ 2>/dev/null; cd /tmp/src
for mb in 32 48 64; do
  EXL3_MOE_FLAGS="-DEXL3_MOE_PREFILL_MB=$mb" EXL3_MOE_OUT=/out/_moe$mb.so bash scripts/build_moe.sh >/out/build$mb.log 2>&1 || { echo "build $mb failed"; cat /out/build$mb.log | tail -5; }
done
for mb in 32 48 64; do [ -f /out/_moe$mb.so ] && python3 /h/t_moe.py /out/_moe$mb.so mb$mb 2>&1 | grep -E "^mb|Error|error" ; done
python3 - <<'PY'
import torch
r = {m: torch.load(f"/out/res_mb{m}.pt") for m in (32, 48, 64) if __import__("os").path.exists(f"/out/res_mb{m}.pt")}
ref = r[32]
for m, v in r.items(): print(f"mb{m} vs mb32: rel={((v-ref).norm()/ref.norm()).item():.2e} maxabs={(v-ref).abs().max().item():.3e}")
PY
