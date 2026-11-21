# Measured GPU copy bandwidth and fp16 GEMM throughput on Metal via MLX (interleaved min-of-N).
import time, mlx.core as mx
def t(f, n=10):
    f(); best = 1e9
    for _ in range(n):
        s = time.perf_counter(); f(); best = min(best, time.perf_counter() - s)
    return best
x = mx.random.uniform(shape=(1024 * 1024 * 256,), dtype=mx.float32); mx.eval(x)  # 1 GiB
bw = 2 * x.nbytes / t(lambda: mx.eval(x + 1.0)) / 1e9
print(f"copy bandwidth (read+write 1 GiB): {bw:.0f} GB/s")
for n in (4096, 8192):
    a = mx.random.normal((n, n)).astype(mx.float16); b = mx.random.normal((n, n)).astype(mx.float16); mx.eval(a, b)
    print(f"fp16 GEMM {n}^3: {2 * n**3 / t(lambda: mx.eval(a @ b), 5) / 1e12:.2f} TFLOPS")
