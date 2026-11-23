#!/usr/bin/env python3
"""Reproducible single-GPU Inkling campaign; no output caps or clock changes."""
import argparse, hashlib, json, random, signal, statistics, subprocess as sp, time, urllib.request, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=json.loads((ROOT/'configs/base-launch.json').read_text())
URL='http://127.0.0.1:8102'
SAMPLING_TEMP=0.8
NAME='inkling-extended-trial'
PROMPTS={
 'prose':'Write a complete, original story of about 2500 words about a lighthouse keeper and a cartographer discovering that their maps disagree. Include dialogue, a genuine disagreement resolved through evidence, and a definite ending. Write the story itself, not an outline.',
 'code':'Write a complete Python standard-library-only module implementing a thread-safe bounded LRU cache with per-entry TTL, an injectable monotonic clock, get/set/delete/clear, hit/miss/eviction statistics and a get_or_compute method. Explain the concurrency contract. Include a complete unittest suite covering expiration, eviction, replacement, missing keys, falsey values, errors in the factory and concurrent access. Provide the full implementation and tests, without placeholders.',
 'reasoning':'Solve all the following problems, showing intermediate reasoning and a clearly numbered final answer for each. For each integer n from 11 through 30 inclusive, a bag contains n red balls and n+3 blue balls. Draw four balls uniformly without replacement. Compute the exact probability of exactly two red balls as a reduced fraction. Derive the formula once, then show the integer combinations, numerator, denominator, reduction and answer for every n. Do not skip any case.'
}
VARIANTS={'baseline':{},'gamma3':{'--speculative-dspark-block-size':'3'},'gamma11':{'--speculative-dspark-block-size':'11'},'chunk1024':{'--chunked-prefill-size':'1024'},'chunk4096':{'--chunked-prefill-size':'4096'}}

VARIANTS.update({'compact':{'env:SGLANG_RAGGED_VERIFY_MODE':'compact'},'draft-mxfp8':{'--speculative-dspark-lm-head-dtype':'mxfp8'}})

def shell(*a):return sp.check_output(a,text=True).strip()
def request(path,body=None,timeout=600):
 req=urllib.request.Request(URL+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
 except urllib.error.HTTPError as e:
  raise RuntimeError(str(e)+': '+e.read().decode()[:500]) from e
def count(text):
 d=request('/v1/tokenize',{'model':'Inkling-2-Mini','prompt':text}); n=d.get('count') or d.get('length') or len(d.get('tokens',[]))
 if not n and text:raise RuntimeError('Tokenizer count unavailable')
 return n

def ready(port,name):
 end=time.monotonic()+600
 while time.monotonic()<end:
  if shell('docker','inspect','-f','{{.State.Running}}',name)!='true':raise RuntimeError(name+' exited')
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as r:model_info=json.load(r)
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health_generate',timeout=240) as r:
    assert r.status==200
   return model_info
  except Exception:time.sleep(3)
 raise TimeoutError('Readiness exceeded 600 seconds')

def telemetry():
 try:return shell('nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits')
 except Exception:return None

def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row)+'\n')

def cold():
 req=urllib.request.Request(URL+'/flush_cache?timeout=120',data=b'{}',headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:
  if r.status!=200:raise RuntimeError('Cache flush failed')

def prompt(kind,size,seed,position=None):
 rng=random.Random(seed)
 words='amber birch cedar delta elm fern granite harbor iris juniper kelp lantern meadow north ocean pine quartz river stone timber upland violet willow yellow zephyr'.split()
 paragraphs=[' '.join(rng.choices(words,k=100)) for _ in range(max(1,size//100+20))]
 text='\n'.join(paragraphs)
 # Binary search character boundary, with tokenizer as authority. Include unique prefix to avoid cross-request reuse.
 prefix=f'Synthetic reference document {seed}. The document is background material.\n'
 lo,hi=0,len(text)
 while lo<hi:
  mid=(lo+hi+1)//2
  if count(prefix+text[:mid])<=size-350:lo=mid
  else:hi=mid-1
 text=text[:lo]
 if kind=='recall':
  needle=f'beacon-{seed}-cobalt';idx=int(len(text)*position)
  text=text[:idx]+'\nThe secret lookup code is '+needle+'.\n'+text[idx:]
  task='Return only the exact secret lookup code from the reference document.'
 elif kind=='prefill':task='Return only the word READY.';needle='READY'
 elif kind=='probe':task='Explain how a write-ahead log supports recovery after a crash, in about 400 words, using one concrete example.';needle=None
 else:task=PROMPTS[kind];needle=None
 return prefix+text+'\nEND OF REFERENCE DOCUMENT.\nTask: '+task,needle

def sample(profile,phase,kind,size,seed,position=None):
 p,needle=prompt(kind,size,seed,position)
 cold()
 payload={'model':'Inkling-2-Mini','messages':[{'role':'user','content':p}],'temperature':0 if needle else SAMPLING_TEMP,'seed':seed,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 row={'unix_start':time.time(),'profile':profile,'phase':phase,'kind':kind,'target_input':size,'seed':seed,'position':position,'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),'gpu_before':telemetry()}
 start=time.monotonic();first=None;visible_first=None;pieces=[];content=[];reasoning=[];window=None;usage=None;finish=None;done=False
 try:
  with urllib.request.urlopen(req,timeout=600) as r:
   for line in r:
    if not line.startswith(b'data:'):continue
    raw=line[5:].strip()
    if raw==b'[DONE]':done=True;break
    d=json.loads(raw);usage=d.get('usage') or usage
    for choice in d.get('choices',[]):
     finish=choice.get('finish_reason') or finish;delta=choice.get('delta',{})
     thought=delta.get('reasoning_content') or delta.get('reasoning') or '';visible=delta.get('content') or '';piece=thought+visible
     now=time.monotonic()
     if piece:
      if first is None:first=now
      pieces.append(piece);content.append(visible);reasoning.append(thought)
      if visible and visible_first is None:visible_first=now
      if window is None and now-first>=30:window=(''.join(pieces),now-first)
  end=time.monotonic();whole=''.join(pieces);span=end-(first or start);window_text,window_span=window or (whole,span)
  n=count(whole);wn=count(window_text)
  row.update(seconds=end-start,ttft=None if first is None else first-start,visible_ttft=None if visible_first is None else visible_first-start,decode_seconds=span,accepted_text_tokens=n,accepted_text_tps=n/span if span else None,window_tokens=wn,window_seconds=window_span,window_tps=wn/window_span if window_span else None,usage=usage,finish=finish,done=done,content=''.join(content),reasoning=''.join(reasoning),gpu_after=telemetry())
  row['passed']=done and finish=='stop' and (needle is None or needle==row['content'].strip())
  if needle:row['expected']=needle
  if kind=='prose':
   row['visible_words']=len(row['content'].split())
   row['scope_check']=1000<=row['visible_words']<=5000
   row['passed']=row['passed'] and row['scope_check']
 except Exception as e:row.update(error=repr(e),seconds=time.monotonic()-start,passed=False,gpu_after=telemetry())
 append(ROOT/'samples.jsonl',row)
 print(json.dumps({k:row.get(k) for k in ('profile','phase','kind','target_input','seed','seconds','ttft','window_tps','finish','passed','error')}),flush=True)
 return row

