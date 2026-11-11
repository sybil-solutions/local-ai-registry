**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX 8-bit · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,326 tok/s | 102.1 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,217 tok/s | 147.7 tok/s (73.9/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,126 tok/s | 172.7 tok/s (43.2/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,103 tok/s | 85.7 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 986 tok/s | 118.0 tok/s (59.0/stream) (1 of 2 waves: streams ran one at a time) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 102.3 tok/s. Quality: KL to bf16 0.0037 nats (95% CI 0.0029–0.0045), top-token agreement 96.8%. Peak GPU memory in these rows 2.93 GiB.

**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX bf16 (unquantized) · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,397 tok/s | 62.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,250 tok/s | 41.5 tok/s (20.8/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,151 tok/s | 85.6 tok/s (21.4/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,184 tok/s | 56.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,076 tok/s | not run (streams ran one at a time: each answer ended before the other prompt was read) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 67.8 tok/s. Quality: the unquantized weights (the reference). Peak GPU memory in these rows 4.32 GiB.

**LFM3-1.7B · Apple M1 Max 32 GB (32-core GPU) · MLX 4-bit · mlx-vlm 0.7.12, 262,144 window, 128-token prefill step, up to 4 at once**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,498 tok/s | 130.9 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,325 tok/s | 188.1 tok/s (94.0/stream) | 2 | 262k (~3.0 GiB est. when full) | 1 |
| 8k | 1,200 tok/s | 207.1 tok/s (51.8/stream) | 4 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,217 tok/s | 112.8 tok/s | 1 | 262k (~3.0 GiB est. when full) | 1 |
| 32k | 1,039 tok/s | 133.4 tok/s (66.7/stream) (1 of 2 waves: streams ran one at a time) | 2 | 262k (~3.0 GiB est. when full) | 1 |

No context, 1 stream: 127.4 tok/s. Quality: KL to bf16 0.2571 nats (95% CI 0.2138–0.3044), top-token agreement 75.1%. Peak GPU memory in these rows 2.12 GiB.

