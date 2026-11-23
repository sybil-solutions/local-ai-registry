import json,time,statistics,subprocess
import torch

def state():
 return subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used','--format=csv,noheader,nounits'],text=True).strip()
torch.cuda.set_per_process_memory_fraction(.2)
torch.set_num_threads(1)
results={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'memory_total_bytes':torch.cuda.get_device_properties(0).total_memory,'before':state(),'copy':[],'gemm':{}}
source=torch.ones(1024**3,device='cuda',dtype=torch.uint8);dest=torch.empty_like(source)
matrices={n:(torch.randn(n,n,device='cuda',dtype=torch.bfloat16),torch.randn(n,n,device='cuda',dtype=torch.bfloat16),torch.empty(n,n,device='cuda',dtype=torch.bfloat16)) for n in (4096,8192)}

def elapsed(fn,reps=20):
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(reps):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/1000/reps
end=time.monotonic()+3
while time.monotonic()<end:
 for n,(a,b,c) in matrices.items():torch.mm(a,b,out=c)
 dest.copy_(source);torch.cuda.synchronize()
for i in range(10):
 t=elapsed(lambda:dest.copy_(source),50)
 results['copy'].append({'seconds':t,'buffer_bytes':source.numel(),'read_plus_write_gbs':2*source.numel()/t/1e9})
 for n,(a,b,c) in matrices.items():
  t=elapsed(lambda:torch.mm(a,b,out=c),20)
  results['gemm'].setdefault(str(n),[]).append({'seconds':t,'bf16_tflops':2*n**3/t/1e12})
results['after']=state();results['bandwidth_gbs_median']=statistics.median(x['read_plus_write_gbs'] for x in results['copy'])
results['bf16_tflops_median']={n:statistics.median(x['bf16_tflops'] for x in rs) for n,rs in results['gemm'].items()}
# Native FP8 is an additional measured bound; keep errors explicit instead of substituting a spec-sheet peak.
try:
 n=4096;a=torch.randn(n,n,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn);b=torch.randn(n,n,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).t();scale=torch.ones((),device='cuda')
 def fp8():return torch._scaled_mm(a,b,scale_a=scale,scale_b=scale,out_dtype=torch.bfloat16)
 for _ in range(10):fp8()
 torch.cuda.synchronize();ts=[elapsed(fp8,20) for _ in range(6)]
 results['fp8_4096_tflops']=[2*n**3/t/1e12 for t in ts]
except Exception as e:results['fp8_error']=repr(e)
print(json.dumps(results,indent=2))
