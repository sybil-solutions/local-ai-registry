# Build from the registry checkout root, at the launch's pinned build.commit.
FROM lmsysorg/sglang@sha256:e4e10d66de374e6e8cc88136e8dbfc3182787070df8a48e581fb007cc7a51c36
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-registry"
ARG TRELLIS_COMMIT=72fd2d4ca8e8de3fce03ab09e3ed1692957c2f8c
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN git init -q /opt/trellis-serve \
 && cd /opt/trellis-serve \
 && git fetch -q --depth 1 https://github.com/sybil-solutions/trellis-serve.git "$TRELLIS_COMMIT" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$TRELLIS_COMMIT"
COPY data/sources/ornith-b390-exl3/k4-moe.patch /tmp/k4-moe.patch
RUN cd /opt/trellis-serve && git apply --check /tmp/k4-moe.patch && git apply /tmp/k4-moe.patch
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu
WORKDIR /opt/trellis-serve/xpu
RUN rm -f exl3xpu/_C.so exl3xpu/_moe.so \
 && bash scripts/build_moe.sh && bash scripts/build_ext.sh \
 && test -s exl3xpu/_C.so && test -s exl3xpu/_moe.so \
 && python3 -c "import torch; torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_C.so'); torch.ops.load_library('/opt/trellis-serve/xpu/exl3xpu/_moe.so')"
COPY data/sources/ornith-b390-exl3/entrypoint.sh /opt/ornith-entrypoint.sh
RUN chmod 0755 /opt/ornith-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/opt/ornith-entrypoint.sh"]
