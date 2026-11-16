# Best hill-climbed Ornith runtime (2026-11-15): K4 EXL3 kernels + oneDNN INT8 prefill + plugin fixes
FROM ornith-hc:dnnl
COPY sglang_plugin.py moe_offload.py ops.py /opt/trellis-serve/xpu/exl3xpu/
COPY dv256.json /opt/ornith/dv256.json
