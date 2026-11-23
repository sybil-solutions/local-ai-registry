"""Measure real Metal memory bandwidth (copy + read-reduce) and GEMM throughput with MLX."""
import time, json, mlx.core as mx
def t(f, n=20):
    f(); mx.synchronize(); best=1e9
    for _ in range(n):
        s=time.perf_counter(); f(); mx.synchronize(); best=min(best, time.perf_counter()-s)
    return best
res={}
a=mx.random.normal((1<<28,)).astype(mx.float16); mx.eval(a)  # 512 MiB
res["copy_gbs"]=round(2*a.nbytes/t(lambda: mx.eval(a*1))/1e9,1)
res["read_gbs"]=round(a.nbytes/t(lambda: mx.eval(a.sum()))/1e9,1)
w=mx.random.normal((8192,8192*4)).astype(mx.float16); x=mx.random.normal((1,8192)).astype(mx.float16); mx.eval(w,x)
res["gemv_fp16_gbs"]=round(w.nbytes/t(lambda: mx.eval(x@w))/1e9,1)
wq,s,b=mx.quantize(w,group_size=64,bits=4); mx.eval(wq,s,b)
qb=wq.nbytes+s.nbytes+b.nbytes
res["gemv_q4_gbs"]=round(qb/t(lambda: mx.eval(mx.quantized_matmul(x,wq,s,b,transpose=False,group_size=64,bits=4)))/1e9,1)
for n in (4096,8192):
    A=mx.random.normal((n,n)).astype(mx.bfloat16); B=mx.random.normal((n,n)).astype(mx.bfloat16); mx.eval(A,B)
    res[f"gemm_bf16_{n}_tflops"]=round(2*n**3/t(lambda: mx.eval(A@B),8)/1e12,2)
print(json.dumps(res))
