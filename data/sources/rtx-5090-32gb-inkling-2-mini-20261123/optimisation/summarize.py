"""Summarize measured evidence without promoting failed or incomplete runs."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text()) if p.exists() else None

def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def med(values):return statistics.median(values) if values else None

controls=[read(ROOT/'runs'/n/'result.json') for n in ['control-a','control-b','control-c']]
candidates=[read(ROOT/'runs'/n/'result.json') for n in ['cache-warm-a','cache-warm-b']]
q=ROOT/'qualification/retained-profile';rows=lines(q/'ledger.jsonl');recall=lines(q/'recall-grades.jsonl')
report={'decision':'Retain original serving settings; no new configuration passed every preregistered gate.',
 'scope':'128K c1 experimental profile; daily48K service and4096 output budget unchanged.',
 'cache_startup':{'control_seconds':[r['startup_seconds'] for r in controls],'control_median_seconds':med([r['startup_seconds'] for r in controls]),'candidate_seconds':[r['startup_seconds'] for r in candidates],'adopted':False,'reason':'Both sampled candidates failed combined gates; full temperature-zero follow-up invalidated by degenerate control.'},
 'stochastic_panel':{'method':'Median of six fixed uncapped tasks per run, temperature0.8, approximately8K input, cold prefix; not equivalent to official speed gate.',
  'control_decode_tps':[r['decode_median'] for r in controls],'candidate_decode_tps':[r['decode_median'] for r in candidates],
  'control_32k_ttft_seconds':[r['prefill_ttft_median'] for r in controls],'candidate_32k_ttft_seconds':[r['prefill_ttft_median'] for r in candidates]},
 'qualification':read(q/'result.json'),'semantic_panel_review':read(q/'panel-reviewed.json'),
 'daily_review':read(q/'daily-reviewed.json'),'cache_reuse':lines(q/'cache-evidence.jsonl'),
 'soak':read(q/'soak-summary.json'),'official':read(ROOT/'official-qualification/result.json'),
 'retrieval':{'all_cases':len(recall),'passed':sum(r['passed'] for r in recall),'initial_cases':[{'size':r['target_input'],'seed':r['seed'],'label':r['label'],'passed':r['passed'],'seconds':r['seconds'],'finish':r['finish']} for r in recall if r['phase']=='recall']}}
report['qualification_panels']=[]
for size in [32768,114688]:
 rr=[r for r in rows if r['phase']=='panel' and r['target_input']==size]
 report['qualification_panels'].append({'target_input':size,'n':len(rr),'median_decode_tps':med([r['usage_decode_tps'] for r in rr]),'median_ttft_seconds':med([r['ttft'] for r in rr]),'median_answer_seconds':med([r['seconds'] for r in rr]),'flags':[r['flags'] for r in rr]})
(ROOT/'final-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['qualification','semantic_panel_review','daily_review','cache_reuse']},indent=2))
