# Karmic Kraken serving image for the glm-5.3-flash Studio recipe:
# serve entrypoint (the Studio controller appends `vllm serve` arguments) and the OnlineMoEMethodBase workspace fix.
FROM ghcr.io/local-inference-lab/vllm@sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522
COPY moe_base.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/online/moe_base.py
ENTRYPOINT ["/opt/venv/bin/lil-runtime-bootstrap","/opt/venv/bin/python","-m","vllm.entrypoints.cli.main","serve"]
