#!/usr/bin/env python3
"""CPU-only checkpoint inventory and explicitly conditional traffic ceilings.

Reads safetensors headers, never weight payloads or GPU APIs. Hardware inputs must
come from hwbench.py; there are no spec-sheet substitutes. Re-run after hardware.json changes.
"""
import argparse,collections,hashlib,json,math,re,statistics,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEFAULT=Path.home()/'.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7'

def header(path):
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
  if len(raw)!=n:raise ValueError('Truncated safetensors header')
  data=json.loads(raw)
 return {k:v for k,v in data.items() if k!='__metadata__'},hashlib.sha256(raw).hexdigest()

def size(v):return v['data_offsets'][1]-v['data_offsets'][0]
def elems(v):return math.prod(v['shape'])
def gb(n):return n/1e9

def build(checkpoint):
 config=json.loads((checkpoint/'config.json').read_text());t=config['text_config'];m=config['mtp_config'];loop=t['loop_config']
 visits={i:loop['num_passes'] if loop['core_start']<=i<loop['core_end'] else 1 for i in range(t['num_hidden_layers'])}
 main={};draft={};sources={};dtype=collections.Counter()
 for p in sorted(checkpoint.glob('*.safetensors')):
  h,digest=header(p);sources[p.name]={'header_sha256':digest,'file_bytes':p.stat().st_size,'tensor_payload_bytes':sum(map(size,h.values())),'tensor_count':len(h)}
  for v in h.values():dtype[v['dtype']]+=size(v)
  (draft if p.name=='mtp.safetensors' else main).update(h)
 groups=collections.Counter();effective=collections.Counter();per_layer={i:collections.Counter() for i in visits}
 linear={'dense_layer_params_effective':0,'expert_active_params_effective':0,'expert_active_params_effective_by_dtype':{}}
 expert_dtypes=collections.Counter()
 for k,v in main.items():
  n=size(v);match=re.match(r'model\.llm\.layers\.(\d+)\.',k)
  if match:
   i=int(match.group(1));is_expert='.mlp.experts.' in k
   group='routed_experts' if is_expert else 'dense_layers'
   groups[group]+=n;effective[group]+=n*visits[i];per_layer[i][group]+=n
   if is_expert:
    expert_dtypes[v['dtype']]+=n
    if k.endswith('_weight') and v['dtype'] in ('U8','F8_E4M3','BF16','F16'):
     unpack=2 if v['dtype']=='U8' else 1
     active=elems(v)*unpack*visits[i]*t['num_experts_per_tok']/t['n_routed_experts']
     linear['expert_active_params_effective']+=active
     by_dtype=linear['expert_active_params_effective_by_dtype'];by_dtype[v['dtype']]=by_dtype.get(v['dtype'],0)+int(active)
   elif len(v['shape'])==2 and v['dtype'] in ('BF16','F16','F32') and k.endswith('.weight'):
    linear['dense_layer_params_effective']+=elems(v)*visits[i]
  elif k=='model.llm.embed.weight':groups['input_embedding']+=n
  elif k=='model.llm.unembed.weight':groups['output_head']+=n
  elif k.startswith('model.llm.'):groups['text_global_other']+=n
  elif k.startswith('model.visual.'):groups['vision']+=n
  else:groups['other']+=n
 assert sum(groups.values())==sum(size(v) for v in main.values())
 local=set(t['local_layer_ids']);full=[i for i in visits if i not in local]
 nfull=sum(visits[i] for i in full);nlocal=sum(visits[i] for i in local)
 draft_full=m['num_hidden_layers']-len(m['local_layer_ids']);draft_local=len(m['local_layer_ids'])
 target_kv_per_layer=2*t['num_key_value_heads']*t['head_dim']*2
 local_kv_per_layer=2*t['swa_num_key_value_heads']*t['swa_head_dim']*2
 draft_kv_per_layer=2*m['num_key_value_heads']*m['head_dim']*2
 physical={'main':dict(groups),'draft_payload_bytes':sum(size(v) for v in draft.values()),'all_dtype_bytes':dict(dtype),'expert_dtype_bytes':dict(expert_dtypes)}
 kv={'target_full_layers':nfull,'target_local_layers':nlocal,'draft_full_layers':draft_full,'draft_local_layers':draft_local,'full_bytes_per_token':nfull*target_kv_per_layer+draft_full*draft_kv_per_layer,'target_full_bytes_per_token':nfull*target_kv_per_layer,'draft_full_bytes_per_token':draft_full*draft_kv_per_layer,'local_bytes_per_cached_token':nlocal*local_kv_per_layer+draft_local*draft_kv_per_layer,'actual_local_window':t['sliding_window_size'],'allocated_swa_pool_tokens':4096,'context_tokens':131072}
 kv['full_pool_bytes']=kv['full_bytes_per_token']*kv['context_tokens'];kv['allocated_swa_pool_bytes']=kv['local_bytes_per_cached_token']*kv['allocated_swa_pool_tokens']
 kv['decode_local_read_bytes_per_step']=kv['local_bytes_per_cached_token']*kv['actual_local_window']
 # Draft projection is shared with the target output head; it is not another checkpoint tensor.
 # MTP weights are scanned once per block in this scenario. Markov recurrence/reloads need profiling.
 weight_traffic={'dense_layer_visit_bytes':effective['dense_layers'],'expert_all_visit_bytes':effective['routed_experts'],'global_other_bytes':groups['text_global_other'],'output_head_bytes':groups['output_head'],'input_embedding_row_bytes':t['hidden_size']*2,'draft_payload_scan_once_bytes':physical['draft_payload_bytes'],'draft_output_head_bytes':groups['output_head']}
 # Only count quantized *weight* values, not scale tensors, as expert matrix parameters.
 linear['expert_active_params_effective']=int(linear['expert_active_params_effective'])
 linear['output_head_params']=elems(main['model.llm.unembed.weight'])
 linear['prefill_linear_flops_per_token_excluding_final_head']=2*(linear['dense_layer_params_effective']+linear['expert_active_params_effective'])
 result={'checkpoint_revision':checkpoint.name,'config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),'source_headers':sources,'physical':physical,'layer_visits':visits,'physical_full_layer_ids':full,'effective_layer_visits':sum(visits.values()),'per_layer_physical_bytes':{i:dict(v) for i,v in per_layer.items()},'kv_bf16':kv,'weight_traffic_inputs':weight_traffic,'linear_operation_inputs':linear,'expert_routing':{'top_k':t['num_experts_per_tok'],'experts':t['n_routed_experts']},'assumptions':['Core layer weights are revisited for each of three passes; no whole-core L2 retention is assumed.','Target input embedding is a row lookup, not a scan of the complete embedding table.','Input embedding and output head have separate checkpoint storage; the headers alone do not prove numerical equality or runtime parameter aliasing.','Draft projection reuses the target output head in the baseline runtime; it is read once per block in this model, not seven serial times.','The complete 635 MB MTP payload is included once per block as a scenario; lookup locality can lower traffic, while Markov recurrence and reloads can raise it.','Expert overlap and independent routing are scenarios, not guaranteed lower/upper bounds on observed DRAM traffic.','Full KV is read once per verification block and once by draft attention; kernels may reload it per tile, increasing traffic.','State read/write, activations, allocation, sampling, dequantization and runtime overhead are excluded from traffic estimates; state is small but nonzero.','Vision weights are resident but not scanned during text-only decode.','BF16 KV formula is corroborated by the launch log: target full K/V 2 GiB each, draft 0.25 GiB each; state pools remain additional.']}
 hardware_path=ROOT/'hardware.json'
 if hardware_path.exists():
  hw=json.loads(hardware_path.read_text());bandwidth=hw['bandwidth_gbs_median'];result['measured_hardware']={'hardware_sha256':hashlib.sha256(hardware_path.read_bytes()).hexdigest(),'bandwidth_gbs_read_plus_write_median':bandwidth,'bf16_tflops_median':hw['bf16_tflops_median'],'fp8_4096_tflops_median':statistics.median(hw['fp8_4096_tflops']) if hw.get('fp8_4096_tflops') else None,'note':'Copy bandwidth counts source read plus destination write. GEMM numbers are measured dense kernels, not NVFP4 grouped-MoE throughput.'}
  rows=[];ratio=t['num_experts_per_tok']/t['n_routed_experts'];verify_rows=m['block_size']
  base=effective['dense_layers']+groups['text_global_other']+groups['output_head']+t['hidden_size']*2
  draft_scan=physical['draft_payload_bytes']+groups['output_head']
  for ctx in (8192,32768,114688,131072):
   target_kv=nfull*target_kv_per_layer*ctx+nlocal*local_kv_per_layer*t['sliding_window_size']
   draft_kv=draft_full*draft_kv_per_layer*ctx+draft_local*draft_kv_per_layer*m['sliding_window_size']
   no_spec_bytes=base+effective['routed_experts']*ratio+target_kv
   rows.append({'context_tokens':ctx,'mode':'no_spec','accepted_per_step':1,'bytes_per_step':no_spec_bytes,'traffic_scenario_tps':bandwidth*1e9/no_spec_bytes})
   for scenario,union in [('perfect_expert_overlap',ratio),('independent_expert_routing',1-(1-ratio)**verify_rows)]:
    step=base+effective['routed_experts']*union+target_kv+draft_scan+draft_kv
    for accepted in (2,3,4,6):rows.append({'context_tokens':ctx,'mode':scenario,'verify_rows':verify_rows,'accepted_per_step':accepted,'expert_union_fraction':union,'bytes_per_step':step,'traffic_scenario_tps':bandwidth*1e9*accepted/step})
  result['decode_traffic_scenarios']=rows
  result['prefill_measured_ceiling']=None
  result['prefill_ceiling_reason']='No measured NVFP4 grouped-MoE throughput or stage profile; applying BF16 or FP8 dense peak to all 4-bit expert FLOPs would manufacture a ceiling.'
 return result

def markdown(r):
 p=r['physical'];g=p['main'];w=r['weight_traffic_inputs'];kv=r['kv_bf16'];lin=r['linear_operation_inputs'];hw=r.get('measured_hardware')
 lines=['# Inkling 2 Mini: measured hardware and conditional ceilings','','Generated by `python3 theory.py` using only checkpoint headers/config and `hardware.json`. No GPU work is performed by this script.','',f'Weights revision: `{r["checkpoint_revision"]}`. All bytes below are decimal GB unless marked GiB.','', '## Model accounting','',f'The checkpoint has 24 physical text layers. Core layers 6–17 run three times, giving {r["effective_layer_visits"]} layer visits per token: {kv["target_full_layers"]} full-attention and {kv["target_local_layers"]} local-attention visits. The three-layer draft has one full-attention layer and two local layers.','', '| Component | Physical GB | Visit-weighted GB per target forward |','|---|---:|---:|',f'| Routed experts, including FP4/FP8 payload and scales | {gb(g["routed_experts"]):.3f} | {gb(w["expert_all_visit_bytes"]):.3f} before routing |',f'| Other layer weights | {gb(g["dense_layers"]):.3f} | {gb(w["dense_layer_visit_bytes"]):.3f} |',f'| Output head | {gb(g["output_head"]):.3f} | {gb(g["output_head"]):.3f} |',f'| Input embedding | {gb(g["input_embedding"]):.3f} | Row lookup: {w["input_embedding_row_bytes"]} bytes |',f'| Vision | {gb(g["vision"]):.3f} | Excluded for text decode |',f'| Draft checkpoint | {gb(p["draft_payload_bytes"]):.3f} | Once per block, as a traffic scenario |','', 'The input embedding and output head have distinct checkpoint offsets. This proves separate stored tensors, not whether the runtime aliases parameters or their numerical values happen to match. The baseline draft uses the target output projection. Its BF16 head scan adds about 0.926 GB per draft block, but not another0.926GB of unique checkpoint weights. Some MTP tensors are lookup tables or recurrent Markov projections. The one-scan estimate needs kernel profiling: locality can lower traffic, whereas recurrence and reloads can raise it.','',f'Expert payload and scale bytes are counted from safetensors offsets, not parameter count×bits. Physical layers 0–19 use packed FP4 weights; layers 20–23 use FP8 weights (F8_E4M3 in the checkpoint). Effective active matrix parameters, excluding final head, are {lin["dense_layer_params_effective"]/1e9:.3f}B dense plus {lin["expert_active_params_effective"]/1e9:.3f}B routed. This is {lin["prefill_linear_flops_per_token_excluding_final_head"]/1e9:.3f}GFLOP/token before attention and other operations.','', '## KV and memory','',f'BF16 full KV: {kv["full_bytes_per_token"]:,}B/token total ({kv["target_full_bytes_per_token"]:,} target + {kv["draft_full_bytes_per_token"]:,} draft). At 131072 tokens this is {kv["full_pool_bytes"]/2**30:.2f}GiB. The allocated 4096-token SWA pool adds {kv["allocated_swa_pool_bytes"]/2**30:.3f}GiB. The actual sliding window is 1024 tokens, so each step needs only that window logically. State caches, allocator metadata, graph buffers, temporary activations and driver memory are additional.','', 'The server log independently reports target full K/V 2.00 GiB each and draft full K/V 0.25 GiB each, agreeing with this accounting. It also reports approximately 0.02 GiB conv-state and 0.04 GiB intermediate conv-window storage for eight state slots. Checkpoint size alone is therefore not a memory-fit estimate.','', '## Hardware measurements','']
 if hw:
  lines += [f'- Device copy: **{hw["bandwidth_gbs_read_plus_write_median"]:.0f}GB/s**, median of ten rounds, each copying a 1 GiB buffer 50 times; read+write bytes counted.',f'- BF16 dense GEMM: **{hw["bf16_tflops_median"]["4096"]:.1f}TFLOP/s** at 4096³ and **{hw["bf16_tflops_median"]["8192"]:.1f}TFLOP/s** at 8192³ (ten rounds each).',f'- FP8 dense GEMM: **{hw["fp8_4096_tflops_median"]:.1f}TFLOP/s** at 4096³, median of six rounds.' if hw['fp8_4096_tflops_median'] else '- FP8 measurement unavailable; see hardware.json.', '', '## Decode bandwidth scenarios','', 'These are optimistic bandwidth-only scenario estimates, not measured performance limits or an achievable performance promise. They assume the full attention KV is read once per block, good query-row reuse, one draft-block forward, and no whole-core weight retention across loop passes.','', '`target_step = dense_visits + target_head + expert_visits × expert_union + target_KV`','', '`draft_step = MTP_payload + shared_head_read + draft_KV`','', '`tokens/s = measured_bandwidth × accepted_tokens_per_step / (target_step + draft_step)`','', 'DSpark proposes a block in parallel. Multiplying the entire draft transformer/head by seven serial draft passes would substantially overstate its traffic. Eight verification rows may select the same experts or different experts; the scenarios use perfect overlap (top 8/128) and independent routing [1−(1−8/128)^8]. Actual routing, reloads, dequantization and cache reuse must be profiled.','', '| Context | No spec, scenario tok/s | DSpark a=3, independent routing | DSpark a=3, perfect expert overlap |','|---|---:|---:|---:|']
  for ctx in (8192,32768,114688,131072):
   def value(mode,a):return next(x['traffic_scenario_tps'] for x in r['decode_traffic_scenarios'] if x['context_tokens']==ctx and x['mode']==mode and x['accepted_per_step']==a)
   lines.append(f'| {ctx:,} | {value("no_spec",1):.0f} | {value("independent_expert_routing",3):.0f} | {value("perfect_expert_overlap",3):.0f} |')
  lines += ['', 'The JSON also includes acceptance 2, 4, 6 scenarios. Compare a clean non-looping measured panel only with the scenario matching its measured acceptance. Do not divide a looping-output token rate by this table or call the result hardware efficiency. No percentage of a single practical ceiling is justified until routing/acceptance and per-step traffic are profiled.']
 else:lines += ['Hardware measurements pending. Re-run `python3 theory.py` after hardware.json is available.']
 lines += ['', '## Prefill, TTFT and startup limitations','', 'A defensible mixed-precision prefill ceiling is **not yet measured**. BF16/FP8 dense GEMM cannot stand in for the mixed NVFP4/FP8 grouped-MoE kernels. The expert path has different precision, small per-expert row counts, routing and dequantization overhead. Applying a blanket 50% BF16 peak to the whole model would give a misleading number. Report prefill as measured prompt tokens/TTFT, and profile stages or measure the actual MoE path before assigning a ceiling.','', 'Short TTFT includes tokenization, scheduling and a target forward; the bandwidth scenarios are only a weight/KV floor, not an end-to-end TTFT prediction. A startup floor additionally needs measured checkpoint read speed and allocation/JIT time, which this script does not measure. Persistent caches are a separately testable startup lever.','', '## Remaining uncertainty','']
 lines += ['- '+a for a in r['assumptions']]
 return '\n'.join(lines)+'\n'

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=DEFAULT);args=ap.parse_args()
 r=build(args.checkpoint);(ROOT/'model-components.json').write_text(json.dumps(r,indent=2)+'\n');(ROOT/'THEORY.md').write_text(markdown(r))
 print(json.dumps({'physical_payload_gb':sum(s['tensor_payload_bytes'] for s in r['source_headers'].values())/1e9,'effective_layer_visits':r['effective_layer_visits'],'hardware_present':'measured_hardware' in r,'files':['model-components.json','THEORY.md']}))
