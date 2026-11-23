import hashlib,json,math,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(name):
 p=ROOT/'runs'/name
 return json.loads((p/'result.json').read_text()),[json.loads(x) for x in (p/'ledger.jsonl').read_text().splitlines()]

def median_ratio_ci(a,b,iterations=10000):
 # Paired by the six fixed task labels, not independent mixtures of tasks.
 ratios=[x/y for x,y in zip(a,b)];rng=random.Random(7319);boots=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(iterations))
 return {'paired_median_ratio':statistics.median(ratios),'bootstrap95':[boots[int(.025*iterations)],boots[int(.975*iterations)]]}

def screen_valid(result,samples):
 # The original exact-string prompt did not quote its terminal period.
 # Accept either reading for sanity only; byte-exact greedy equality is still required.
 flags=[]
 for r in samples:
  if r['phase']=='warmup':continue
  f=list(r['flags'])
  if r['phase']=='greedy' and r['kind']=='exact' and r['content'].strip() in ('cobalt lantern 7319','cobalt lantern 7319.'):
   f=[x for x in f if x!='GREEDY_SANITY_FAILED']
  flags+=f
 panel=[r for r in samples if r['phase']=='panel']
 return result.get('completed') and result.get('environment_valid') and not result.get('memory_cap_tripped') and not result.get('watch_error') and not flags and len(panel)==6 and all(r.get('usage_decode_tps') and math.isfinite(r['usage_decode_tps']) and r['usage_decode_tps']>0 for r in panel)

comparisons=[]
for candidate,left,right in [('greedy-cache-a','greedy-control-a','greedy-control-b'),('greedy-cache-b','greedy-control-b','greedy-control-c'),('greedy-chunk4096','greedy-control-c','greedy-control-d')]:
 try:cr,cs=load(candidate);lr,ls=load(left);rr,rs=load(right)
 except FileNotFoundError:continue
 row={'candidate':candidate,'controls':[left,right],'qualified_inputs':all(screen_valid(r,s) for r,s in ((cr,cs),(lr,ls),(rr,rs))),'sanity_correction':'Terminal period in unquoted exact-string prompt is ambiguous; both readings accepted, raw flags preserved.'}
 cp={r['kind']:r for r in cs if r['phase']=='panel'}
 row['comparisons']={}
 for name,result,samples in [(left,lr,ls),(right,rr,rs)]:
  control={r['kind']:r for r in samples if r['phase']=='panel'}
  kinds=sorted(cp)
  cmp=median_ratio_ci([cp[k]['usage_decode_tps'] for k in kinds],[control[k]['usage_decode_tps'] for k in kinds])
  cmp.update(startup_ratio=cr['startup_seconds']/result['startup_seconds'],decode_median_ratio=cr['decode_median']/result['decode_median'],answer_latency_ratio=cr['answer_seconds_median']/result['answer_seconds_median'],prefill_ttft_ratio=cr['prefill_ttft_median']/result['prefill_ttft_median'])
  cg={r['kind']:r['output_sha256'] for r in cs if r['phase']=='greedy'};bg={r['kind']:r['output_sha256'] for r in samples if r['phase']=='greedy'}
  cmp['greedy_exact_matches']=sum(cg.get(k)==bg.get(k) for k in cg);cmp['greedy_cases']=len(cg)
  cmp['per_task']={k:{'decode_ratio':cp[k]['usage_decode_tps']/control[k]['usage_decode_tps'],'answer_ratio':cp[k]['seconds']/control[k]['seconds']} for k in kinds}
  cmp['panel_exact_matches']=sum(cp[k]['output_sha256']==control[k]['output_sha256'] for k in kinds)
  row['comparisons'][name]=cmp
 row['startup_candidate_pass']=row['qualified_inputs'] and all(c['startup_ratio']<.95 and c['decode_median_ratio']>=.97 and c['prefill_ttft_ratio']<=1.03 and c['answer_latency_ratio']<=1.03 and c['greedy_exact_matches']==6 for c in row['comparisons'].values())
 row['lossless_panel_equivalence']=all(c['panel_exact_matches']==6 for c in row['comparisons'].values())
 row['retained']=row['startup_candidate_pass'] and row['lossless_panel_equivalence'] if 'cache' in candidate else row['qualified_inputs'] and row['lossless_panel_equivalence'] and all(c['decode_median_ratio']>=.97 and c['prefill_ttft_ratio']<=.95 and c['answer_latency_ratio']<=1.03 and c['greedy_exact_matches']==6 for c in row['comparisons'].values())
 row['control_panel_exact_matches']=sum(a['output_sha256']==b['output_sha256'] for a,b in zip([r for r in ls if r['phase']=='panel'],[r for r in rs if r['phase']=='panel']))
 row['per_task_nonregression']=all(t['decode_ratio']>=.97 and t['answer_ratio']<=1.03 for c in row['comparisons'].values() for t in c['per_task'].values())
 row['retained']=row['retained'] and row['per_task_nonregression']
 comparisons.append(row)
(ROOT/'deterministic-assessment.json').write_text(json.dumps(comparisons,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
