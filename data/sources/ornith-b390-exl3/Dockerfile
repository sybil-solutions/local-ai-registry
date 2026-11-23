# Build from the registry checkout root, at the launch's pinned build.commit.
FROM lmsysorg/sglang@sha256:e4e10d66de374e6e8cc88136e8dbfc3182787070df8a48e581fb007cc7a51c36
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-registry"
# trellis-serve branch ornith-b390-exl3: K4 MoE kernels, MB64 prefill, GPU sampler, pruned draft head.
ARG TRELLIS_COMMIT=0fab6452b12a7e3f7a4dddf7c70f9dd7b5f0e7ab
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 https://github.com/sybil-solutions/trellis-serve.git "$TRELLIS_COMMIT" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$TRELLIS_COMMIT"
# oneDNN 2026.0.2 links against the image's libsycl.so.8 (2025.3 does not); used for INT8 prefill GEMMs and SDPA.
RUN pip install --no-deps onednn==2026.0.2 onednn-devel==2026.0.2
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu
WORKDIR /opt/trellis-serve/xpu
RUN rm -f exl3xpu/_C.so exl3xpu/_C_dnnl.so exl3xpu/_moe.so \
 && EXL3_MOE_FLAGS="-DEXL3_MOE_PREFILL_MB=64" bash scripts/build_moe.sh \
 && EXL3_NO_DNNL=1 bash scripts/build_ext.sh \
 && EXL3_DNNL_DIR=/opt/venv EXL3_OUT=exl3xpu/_C_dnnl.so bash scripts/build_ext.sh \
 && test -s exl3xpu/_C.so && test -s exl3xpu/_C_dnnl.so && test -s exl3xpu/_moe.so \
 && python3 -c "import torch; torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_C.so'); torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_moe.so')" \
 && python3 -c "import torch; torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_C_dnnl.so')"
# Pruned 256-block draft vocabulary for the MTP head (EXL3_DRAFT_VOCAB), built from the target's token statistics.
COPY data/sources/ornith-b390-exl3/draft-vocab-256.json /opt/ornith/dv256.json
COPY data/sources/ornith-b390-exl3/entrypoint.sh /opt/ornith-entrypoint.sh
RUN chmod 0755 /opt/ornith-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/opt/ornith-entrypoint.sh"]
