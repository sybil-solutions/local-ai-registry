"""Measure GPU memory bandwidth (device copy / reduction) and fp16 matmul throughput on Apple silicon with MLX."""
import json
import time

import mlx.core as mx


def timed(fn, reps):
    fn()
    mx.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


n_bytes = 2 * 2**30  # 2 GiB buffer, far above the SLC
a = mx.random.uniform(shape=(n_bytes // 4,), dtype=mx.float32)
mx.eval(a)

# copy: one read + one write of the buffer
t_copy = timed(lambda: mx.eval(a * 1.0), 20)
copy_gbs = 2 * n_bytes / t_copy / 1e9
# read-only: reduction reads the buffer once (closest to decode, which streams weights)
t_read = timed(lambda: mx.eval(a.sum()), 20)
read_gbs = n_bytes / t_read / 1e9

m = 8192
x = mx.random.normal((m, m), dtype=mx.float16)
y = mx.random.normal((m, m), dtype=mx.float16)
mx.eval(x, y)
t_mm = timed(lambda: mx.eval(x @ y), 10)
tflops = 2 * m**3 / t_mm / 1e12

print(json.dumps({"copy_gbs": round(copy_gbs, 1), "read_gbs": round(read_gbs, 1),
                  "fp16_matmul_tflops": round(tflops, 2), "mlx": mx.__version__,
                  "device": str(mx.default_device())}))
