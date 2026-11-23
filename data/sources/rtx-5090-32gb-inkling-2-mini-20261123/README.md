# Inkling 2 Mini / RTX 5090 owner validation

Measured 23 November 2026 on one NVIDIA GeForce RTX 5090 (32607 MiB reported), NVIDIA driver 615.78.08, Ryzen 9 9950X3D, 128GB system RAM, Linux 7.2.8-9-omarchy. Model weights and serving image are pinned in the launch. No CPU weight offload and no rented/proxy GPU.

`acceptance.json` is the unmodified output of the repository lab, generated with:

```sh
python3 lab/lab.py try thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --model inkling-2-mini --engine sglang-inkling-2-mini-rtx5090-128k \
  --card rtx-5090-32gb --on endpoint --endpoint http://127.0.0.1:8102 \
  --gpu 'NVIDIA GeForce RTX 5090'
```

All six gates passed. The streamed lighthouse-story speed sample ended naturally after about 8 seconds: 149.3 accepted tokens/s, counted by the server tokenizer, 1188 tokens total. Draft proposals are not counted. This is the registry workload, not a coding-task or complete-answer latency claim. The reported 10210 prefill metric is the lab's 91890 input tokens divided by the entire 9.0-second context request, not kernel-only prefill throughput.

`near-limit-recall.json` records an additional prompt of 126048 input tokens. It places `apricot-lighthouse-7319` at the beginning, follows it with 63000 repetitions of `filler `, and requests the code at the end. The exact code was recovered. This check supplements the official context gate; it is not included in its prefill measurement.

Default oversized caches caused CUDA graph-capture OOM on this 32GB card. The submitted command bounds the full cache to 131072 tokens, sliding-window cache to 4096, prefill chunks to 2048 and Mamba state cache to 8 slots. Both graph batch limits are 1. BF16 cache and the model's default three loop passes are retained. DSpark uses the checkpoint's MTP head. One simultaneous request only; headroom is tight. No eager-mode flag is used.

Image support is enabled by the launch, but this 128K acceptance run did not include a separate image-quality test. Do not infer a vision benchmark score from these six gates.
