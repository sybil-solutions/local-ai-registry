# Inkling-2-Mini on Intel Arc B390 (Xe3): SGLang 0.5.20-XPU + trellis-serve inkling2 (native NVFP4/MXFP8 experts).
# Build from the registry checkout root, at the launch's pinned build.commit.
FROM lmsysorg/sglang@sha256:e4e10d66de374e6e8cc88136e8dbfc3182787070df8a48e581fb007cc7a51c36
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-registry"
# trellis-serve branch inkling2-mini-b390: SGLang model for Inkling2LoopForConditionalGeneration (registered by the
# exl3xpu SGLang plugin), SYCL NVFP4/MXFP8/Q8 GEMV + flash-decoding, ESIMD DPAS prefill attention, Triton fused ops.
ARG TRELLIS_COMMIT=5076a00d0f70f8d86bcb150552369932a1927330
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 https://github.com/sybil-solutions/trellis-serve.git "$TRELLIS_COMMIT" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$TRELLIS_COMMIT"
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu
WORKDIR /opt/trellis-serve/xpu
RUN rm -f exl3xpu/_inkling2.so \
 && bash scripts/build_inkling2.sh \
 && test -s exl3xpu/_inkling2.so \
 && python3 -c "import torch; torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_inkling2.so'); print(torch.ops.inkling2_xpu.moe_gemv2)"
COPY data/sources/inkling2-mini-b390/entrypoint.sh /opt/inkling2-entrypoint.sh
RUN chmod 0755 /opt/inkling2-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/opt/inkling2-entrypoint.sh"]
