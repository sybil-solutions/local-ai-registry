"""Select only a preregistered passing candidate, then qualify it after the queue."""
import json,subprocess as sp,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
statusfile=ROOT/'chunk-queue-status.json'
while not statusfile.exists():time.sleep(10)
status=json.loads(statusfile.read_text())
if not status.get('restored') or status.get('cleanup_error') or status.get('restoration_error'):
 raise RuntimeError('Queue did not restore safely; qualification not started')
sp.run([sys.executable,str(ROOT/'assess_chunk.py')],check=True)
rows=json.loads((ROOT/'chunk-assessment.json').read_text())
by={r['candidate']:r for r in rows}
cache=False  # Stochastic repeats failed; deterministic control was degenerate.
chunk=by.get('chunk4096-retry',{}).get('retained',False)
# Never combine independently tested changes without measuring the combination.
selected='chunk4096-retry' if chunk else 'chunk-control-a'
cfg=json.loads((ROOT/'configs'/f'{selected}.json').read_text())
cfg['name']='retained-profile'
(ROOT/'configs/retained-profile.json').write_text(json.dumps(cfg,indent=2)+'\n')
selection={'selected_screen':selected,'configuration':cfg,'cache_passed_both':cache,'chunk_passed':chunk,'status':'selected for qualification; not yet adopted','queue_completed':status.get('completed',False)}
(ROOT/'best-config.json').write_text(json.dumps(selection,indent=2)+'\n')
with (ROOT/'PLAN.md').open('a') as f:f.write('\nAutomatic preregistered selection: '+json.dumps(selection)+'. Qualification starts next; no deployment from screening alone.\n')
cmd=[sys.executable,str(ROOT/'qualification.py'),'--config',str(ROOT/'configs/retained-profile.json'),'--name','retained-profile','--output',str(ROOT/'qualification/retained-profile')]
with (ROOT/'qualification.log').open('w') as log:r=sp.run(cmd,stdout=log,stderr=sp.STDOUT)
(ROOT/'qualification-controller-status.json').write_text(json.dumps({'exit_code':r.returncode,'selection':selection},indent=2)+'\n')
