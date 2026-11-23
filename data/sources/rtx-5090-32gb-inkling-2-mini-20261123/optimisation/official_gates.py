"""Repeat official gates on the exact current registry launch, safely serialized."""
import fcntl,json,os,shlex,shutil,signal,subprocess as sp,sys,time
from pathlib import Path
import runner,core
HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('INKLING_REGISTRY_ROOT','/tmp/inkling-registry-review'))
out=HERE/'official-qualification'
out.mkdir(exist_ok=False)
lock=open('/tmp/inkling-gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
runner.ROOT=out;core.ROOT=out
watch=runner.Watch(out/'telemetry.jsonl')
status={'waves':[]}

def launch_exact():
 p=json.loads(json.dumps(runner.BASE))
 a=['docker','run','-d','--name',runner.NAME,*sum([shlex.split(f) for f in p['flags']],[]),'-p','127.0.0.1:8102:30000']
 if p.get('shm'):a+=['--shm-size',p['shm']]
 a+=['-v',str(Path.home()/'.cache/huggingface/hub')+':/root/.cache/huggingface/hub:ro']
 for k,v in p['env'].items():a+=['-e',k+'='+str(v)]
 a+=['--entrypoint','python3',p['image'],*p['args']]
 (out/'launch.json').write_text(json.dumps(a,indent=2)+'\n');(out/'profile.json').write_text(json.dumps(p,indent=2)+'\n')
 t=time.monotonic();sp.run(a,check=True,stdout=sp.DEVNULL);core.ready(8102,runner.NAME);return time.monotonic()-t

def interrupt(*a):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt)
try:
 core.ready(8101,'inkling-mini');runner.drain_stop('inkling-mini',8101)
 status['environment_before']=runner.environment()
 watch.start();status['startup_seconds']=launch_exact()
 for wave in (1,2):
  core.cold()  # Same gate prompt repeats; reset prefix state so prefill stays cold.
  before=set((REPO/'lab/runs').glob('*.json'))
  cmd=[sys.executable,'lab/lab.py','try','thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7','--model','inkling-2-mini','--engine','sglang-inkling-2-mini-rtx5090-128k','--card','rtx-5090-32gb','--on','endpoint','--endpoint','http://127.0.0.1:8102','--gpu','NVIDIA GeForce RTX 5090']
  with (out/f'wave-{wave}.log').open('w') as log:
   result=sp.run(cmd,cwd=REPO,stdout=log,stderr=sp.STDOUT,timeout=1800)
  files=set((REPO/'lab/runs').glob('*.json'))-before
  assert len(files)==1,files
  record=next(iter(files));shutil.copyfile(record,out/f'wave-{wave}.json')
  data=json.loads(record.read_text());status['waves'].append({'wave':wave,'passed':data['passed'],'exit_code':result.returncode,'proof':data['proof']})
  if result.returncode or watch.tripped or watch.error:raise RuntimeError('Official gate wave failed')
 status['environment_after']=runner.environment(runner.NAME)
 status['environment_valid']=runner.stable(status['environment_before'],status['environment_after'])
 if not status['environment_valid']:raise RuntimeError('Environment changed during official gates')
 status['completed']=True
except BaseException as e:status['error']=repr(e)
finally:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 watch.end();status['memory_cap_tripped']=watch.tripped;status['watch_error']=watch.error
 try:
  if runner.container_exists(runner.NAME):
   logs=sp.run(['docker','logs',runner.NAME],capture_output=True,text=True,timeout=30);(out/'server.log').write_text(logs.stdout+logs.stderr)
 except Exception as e:status['log_capture_error']=repr(e)
 try:runner.drain_stop(runner.NAME,8102)
 except Exception as e:status['cleanup_error']=repr(e)
 finally:
  try:
   if runner.container_exists(runner.NAME) and sp.check_output(['docker','inspect','-f','{{.State.Running}}',runner.NAME],text=True).strip()=='true':raise RuntimeError('Trial still running')
   sp.run(['docker','start','inkling-mini'],check=True,stdout=sp.DEVNULL);core.ready(8101,'inkling-mini');status['restored']=True
  except Exception as e:status['restoration_error']=repr(e)
 status['qualified']=bool(status.get('completed') and status.get('environment_valid') and status.get('restored') and not any(status.get(k) for k in ('error','cleanup_error','restoration_error','log_capture_error','watch_error','memory_cap_tripped')))
 (out/'result.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
