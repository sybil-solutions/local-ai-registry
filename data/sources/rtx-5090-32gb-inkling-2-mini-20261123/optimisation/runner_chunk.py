#!/usr/bin/env python3
"""Skill-guided, uncapped, serialized Inkling experiments. No power changes."""
import argparse,fcntl,hashlib,importlib.util,json,math,os,re,signal,statistics,subprocess as sp,threading,time,urllib.request
from pathlib import Path
import core
ROOT=Path(__file__).resolve().parent
CORE_HASH=hashlib.sha256((ROOT/'core.py').read_bytes()).hexdigest()
RUNNER_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NAME='inkling-hillclimb'; URL='http://127.0.0.1:8102'; VOLUME='inkling-jit-efac853b4681'
BASE=core.BASE
STOP_LOCK=threading.RLock()
# Fixed six-task panel from the requested skill, uncapped and with normal reasoning.
PANEL={
 'lru':'Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.',
 'tcp':'Explain, step by step, how TCP congestion control works, including slow start and fast recovery.',
 'story':'Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.',
 'database':'Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.',
 'backup':'Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.',
 'history':'Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.'
}
GREEDY={
 'exact':'Return exactly the following string and nothing else: cobalt lantern 7319.',
 'arithmetic':'What is 17 multiplied by 23? Give the answer briefly.',
 'sort':'Sort these numbers ascending, preserving duplicates: 9, -2, 0, 4, 9, 1. Return only the list.',
 'gcd':'Write a Python function gcd(a,b) implementing the Euclidean algorithm. Include assertions for gcd(54,24) and gcd(0,5).',
 'logic':'All ravens in this puzzle are blue. Mira is a raven. What colour is Mira? Give only the colour.',
 'json':'Return a JSON object with keys name and count, containing the string beacon and integer 3 respectively. No markdown.'
}
core.URL=URL;core.NAME=NAME;core.PROMPTS={**PANEL,**GREEDY}

def call(path,body=None,port=8102,timeout=180):
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.read()

def container_exists(name):return sp.run(['docker','inspect',name],stdout=sp.DEVNULL,stderr=sp.DEVNULL).returncode==0

def drain_stop(name,port,remove=True):
 with STOP_LOCK:
  return _drain_stop(name,port,remove)

def _drain_stop(name,port,remove):
 if not container_exists(name):return
 if sp.check_output(['docker','inspect','-f','{{.State.Running}}',name],text=True).strip()=='true':
  drained=False;drain_error=None
  try:
   call('/abort_request',{'abort_all':True},port)
   code,_=call('/flush_cache?timeout=120',{},port)
   assert code==200
   drained=True
  except Exception as e:drain_error=repr(e)
  core.append(ROOT/'drains.jsonl',{'unix_seconds':time.time(),'container':name,'drain_confirmed':drained,'drain_error':drain_error})
  # -1 disables Docker's escalation to SIGKILL. The client has a bounded wait;
  # a timeout terminates only the client, leaves the server signalled, and blocks the queue.
  sp.run(['docker','stop','--timeout','-1',name],check=True,timeout=60,stdout=sp.DEVNULL)
 if name==NAME and remove:sp.run(['docker','rm',name],check=True,stdout=sp.DEVNULL)

def environment(expected_container=None):
 profile=sp.check_output(['powerprofilesctl','get'],text=True).strip()
 governors=sorted(set(p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor')))
 power=sp.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip()
 allowed=set()
 if expected_container and container_exists(expected_container):
  out=sp.check_output(['docker','top',expected_container,'-eo','pid'],text=True)
  allowed={int(s.strip()) for s in out.splitlines()[1:] if s.strip().isdigit()}
 helpers=[];unknown=[]
 out=sp.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 for line in out.splitlines():
  pid,_,name=line.partition(',');name=name.strip()
  if int(pid.strip()) in allowed:continue
  basename=Path(name.split()[0]).name
  if '--type=gpu-process' in name and basename in ('ChatGPT','t3code'):helpers.append(basename)
  else:unknown.append(int(pid.strip()))
 online=[p.read_text().strip() for p in Path('/sys/class/power_supply').glob('*/online')]
 state={'profile':profile,'governors':governors,'power_limit_w':float(power),'display_helpers':sorted(helpers),'unknown_compute_pids':unknown,'ac':not online or '1' in online,'load_per_cpu':os.getloadavg()[0]/os.cpu_count()}
 if profile!='performance' or governors!=['performance'] or unknown or not state['ac']:raise RuntimeError('Environment guard: '+json.dumps(state))
 return state

def stable(a,b):
 return all(a[k]==b[k] for k in ('profile','governors','power_limit_w','display_helpers','ac')) and b['load_per_cpu']<.25

def gpu_sample():
 fields='memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu'
 vals=sp.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip().split(',')
 return dict(zip(fields.split(','),map(float,vals)))

class Watch:
 def __init__(self,path,cap_mib=32512):self.path=path;self.cap=cap_mib;self.stop=threading.Event();self.tripped=False;self.error=None;self.thread=threading.Thread(target=self.run,daemon=True)
 def run(self):
  while not self.stop.is_set():
   try:
    row={'unix_seconds':time.time(),**gpu_sample()};core.append(self.path,row)
    if row['memory.used']>self.cap:
     self.tripped=True;self.error='GPU memory cap exceeded';drain_stop(NAME,8102,remove=False);break
   except Exception as e:self.error=repr(e)
   self.stop.wait(2)
 def start(self):self.thread.start()
 def end(self):
  self.stop.set()
  # No next server may reuse NAME while the watchdog can still stop it.
  if self.thread.ident is not None:self.thread.join()

def looping(text,n=8):
 w=text.split();grams=[tuple(w[i:i+n]) for i in range(max(0,len(w)-n))]
 return 0 if len(grams)<=50 else 1-len(set(grams))/len(grams)

def sample(name,phase,kind,size,seed,watch,cold=True,position=None):
 before=environment(NAME)
 if before['load_per_cpu']>=.25:raise RuntimeError('Host load guard before sample')
 original_cold=core.cold
 if not cold:core.cold=lambda:None
 try:r=core.sample(name,phase,kind,size,seed,position)
 finally:core.cold=original_cold
 r['loop_fraction']=looping(r.get('content','')+' '+r.get('reasoning',''))
 r['visible_loop_fraction']=looping(r.get('content',''));r['reasoning_loop_fraction']=looping(r.get('reasoning',''))
 r['flags']=[]
 if r['loop_fraction']>.5 or r['visible_loop_fraction']>.5:r['flags'].append('LOOP')
 if r.get('finish')!='stop':r['flags'].append('NOT_NATURAL_STOP')
 if not r.get('done'):r['flags'].append('MISSING_DONE')
 if not r.get('passed'):r['flags'].append('OUTPUT_CHECK_FAILED')
 if 'error' in r:r['flags'].append('REQUEST_ERROR')
 if kind=='story' and not 200<=len(r.get('content','').split())<=1500:r['flags'].append('SCOPE')
 after=environment(NAME)
 if not stable(before,after):r['flags'].append('ENVIRONMENT_CHANGED')
 if watch.tripped:r['flags'].append('MEMORY_CAP')
 r['environment_before']=before;r['environment_after']=after
 r['output_sha256']=hashlib.sha256((r.get('reasoning','')+'\0'+r.get('content','')).encode()).hexdigest()
 usage=r.get('usage') or {};ct=usage.get('completion_tokens')
 r['usage_decode_tps']=(ct-1)/r['decode_seconds'] if ct and r.get('decode_seconds') else None
 if not ct or not usage.get('prompt_tokens') or not r['usage_decode_tps'] or not math.isfinite(r['usage_decode_tps']):r['flags'].append('USAGE_MISSING')
 if phase=='greedy':
  text=r.get('content','').strip();ok=bool(text)
  if kind=='exact':ok=text=='cobalt lantern 7319.'
  elif kind=='arithmetic':ok=bool(re.search(r'\b391\b',text))
  elif kind=='logic':ok=text.lower().strip('. ' )=='blue'
  elif kind=='sort':
   try:ok=json.loads(re.search(r'\[[^\]]*\]',text).group())==[-2,0,1,4,9,9]
   except Exception:ok=False
  elif kind=='json':
   try:ok=json.loads(re.search(r'\{[^}]*\}',text).group())=={'name':'beacon','count':3}
   except Exception:ok=False
  if not ok:r['flags'].append('GREEDY_SANITY_FAILED')
 core.append(core.ROOT/'ledger.jsonl',r)
 if watch.tripped or 'error' in r:raise RuntimeError('Infrastructure failure during request')
 return r

def launch(cfg,path):
 import shlex
 p=json.loads(json.dumps(BASE));p['args']+=cfg.get('extra_args',[])
 for k,v in cfg.get('replace_args',{}).items():p['args'][p['args'].index(k)+1]=str(v)
 p['env'].update(cfg.get('env',{}))
 a=['docker','run','-d','--name',NAME,*sum([shlex.split(f) for f in p['flags']],[]),'-p','127.0.0.1:8102:30000']
 if cfg.get('cache'):
  a+=['--mount',f'type=volume,src={VOLUME},dst=/root/.cache']
  p['env'].update(TRITON_CACHE_DIR='/root/.cache/triton',CUDA_CACHE_PATH='/root/.cache/nvidia')
 a+=['-v',str(Path.home()/'.cache/huggingface/hub')+':/root/.cache/huggingface/hub:ro']
 for k,v in p['env'].items():a+=['-e',k+'='+str(v)]
 a+=['--entrypoint','python3',p['image'],*p['args']]
 (path/'launch.json').write_text(json.dumps(a,indent=2));(path/'profile.json').write_text(json.dumps(p,indent=2))
 t=time.monotonic();sp.run(a,check=True,stdout=sp.DEVNULL);core.ready(8102,NAME);startup=time.monotonic()-t
 return startup

def run(cfg):
 name=cfg['name'];path=ROOT/'runs'/name;path.mkdir(parents=True,exist_ok=False);core.ROOT=path
 out={'name':name,'config':cfg,'core_sha256':CORE_HASH,'runner_sha256':RUNNER_HASH,'started_unix':time.time()};watch=Watch(path/'telemetry.jsonl');rows=[]
 try:
  out['environment_before']=environment()
  if out['environment_before']['load_per_cpu']>=.25:raise RuntimeError('Host too busy')
  watch.start();out['startup_seconds']=launch(cfg,path)
  # Exclude the first complete request and first post-JIT prompt from measurements.
  warmup_start=time.monotonic();warmup=sample(name,'warmup','lru',8192,900,watch)
  out['warmup_seconds']=time.monotonic()-warmup_start;out['warmup_flags']=warmup['flags']
  out['startup_plus_warmup_seconds']=out['startup_seconds']+out['warmup_seconds']
  for i,kind in enumerate(PANEL):rows.append(sample(name,'panel',kind,8192,4100+i,watch))
  for i in range(3):rows.append(sample(name,'prefill','prefill',32768,5100+i,watch))
  core.SAMPLING_TEMP=0
  for i,kind in enumerate(GREEDY):rows.append(sample(name,'greedy',kind,2048,6100+i,watch))
  core.SAMPLING_TEMP=.8
  out['environment_after']=environment(NAME)
  out['environment_valid']=stable(out['environment_before'],out['environment_after'])
  panel=[r for r in rows if r['phase']=='panel']
  out['panel_flags']=[{'kind':r['kind'],'flags':r['flags']} for r in panel if r['flags']]
  out['all_measured_flags']=[{'phase':r['phase'],'kind':r['kind'],'flags':r['flags']} for r in rows if r['flags']]
  out['decode_median']=statistics.median(r['usage_decode_tps'] for r in panel if r['usage_decode_tps'])
  out['answer_seconds_median']=statistics.median(r['seconds'] for r in panel)
  out['prefill_ttft_median']=statistics.median(r['ttft'] for r in rows if r['phase']=='prefill')
  out['qualified_screen']=len(panel)==6 and not out['all_measured_flags'] and out['environment_valid'] and not watch.tripped and not watch.error
  out['completed']=True
 except BaseException as e:out['error']=repr(e)
 finally:
  core.SAMPLING_TEMP=.8;watch.end();out['memory_cap_tripped']=watch.tripped;out['watch_error']=watch.error
  if watch.tripped or watch.error:out['qualified_screen']=False
  if container_exists(NAME):
   logs=sp.run(['docker','logs',NAME],capture_output=True,text=True);(path/'server.log').write_text(logs.stdout+logs.stderr)
  try:drain_stop(NAME,8102)
  except Exception as e:out['error']=out.get('error') or 'Cleanup failed: '+repr(e)
  (path/'result.json').write_text(json.dumps(out,indent=2));core.append(ROOT/'ledger.jsonl',out)
  print(json.dumps(out),flush=True)
 return out

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('configs',nargs='+');args=ap.parse_args()
 lock=open('/tmp/inkling-gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt)
 status={}
 try:
  core.ready(8101,'inkling-mini');drain_stop('inkling-mini',8101)
  for filename in args.configs:
   cfg=json.loads(Path(filename).read_text());previous=ROOT/'runs'/cfg['name']/'result.json'
   if previous.exists():
    prior=json.loads(previous.read_text())
    if prior.get('completed') and prior.get('config')==cfg and prior.get('core_sha256')==CORE_HASH and prior.get('runner_sha256')==RUNNER_HASH:
     print('RESUME: retaining completed run '+cfg['name'],flush=True);continue
    raise RuntimeError('Existing incomplete or incompatible run; preserve it and choose a new run name: '+cfg['name'])
   r=run(cfg)
   if 'error' in r:raise RuntimeError('Run failed: '+cfg['name'])
  status['completed']=True
 except BaseException as e:status['error']=repr(e)
 finally:
  try:drain_stop(NAME,8102)
  except Exception as e:status['cleanup_error']=repr(e)
  finally:
   try:
    if container_exists(NAME) and sp.check_output(['docker','inspect','-f','{{.State.Running}}',NAME],text=True).strip()=='true':
     raise RuntimeError('Trial did not stop; refusing to start a second GPU server')
    sp.run(['docker','start','inkling-mini'],check=True,stdout=sp.DEVNULL)
    core.ready(8101,'inkling-mini');status['restored']=True
   except Exception as e:status['restoration_error']=repr(e)
  (ROOT/'chunk-queue-status.json').write_text(json.dumps(status,indent=2));print(status,flush=True)
