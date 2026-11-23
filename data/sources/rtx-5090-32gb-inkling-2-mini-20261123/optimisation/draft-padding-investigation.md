# Draft MXFP8 allocation and SWA source investigation

Read-only inspection of the running `inkling-hillclimb` container on the RTX 5090 host, 2026-11-23. No imports, GPU work, configuration changes, or service operations were performed.

## MXFP8 currently falls through

Source root: `/sgl-workspace/sglang/python/sglang/srt/`.

- `models/dspark.py:124` (`project_through_lm_head`) requires CUDA hidden states, floating weights, hidden width divisible by32, and **weight rows divisible by128** for `mxfp8`. Otherwise execution reaches ordinary `torch.matmul` without warning.
- `models/inkling2.py:305` constructs `ParallelLMHead(config.padded_vocab_size, config.hidden_size)` with no padding override. `layers/vocab_parallel_embedding.py:43,234` defaults to64-row allocation padding.
- Current checkpoint/config has padded rows201024 and hidden width2304. `201024 % 128 = 64`; this explains a successful but unchanged MXFP8 pilot. `models/inkling2_mtp.py:201` does use the shared projection helper, so the flag is reachable after correcting allocation.

## A JSON override alone does not work

`hf_transformers/config.py:418` merges dictionary overrides into existing nested `PretrainedConfig` objects with `current.update(value)`. `configs/inkling.py:128` has already normalized raw vocab201024/unpadded200058 into internal logical vocab200058 and padded vocab201024.

Therefore the minimal candidate is:

```json
{"text_config":{"padded_vocab_size":201088}}
```

It preserves normalized logical vocab and untouched loop settings, **but it fails checkpoint loading**. `VocabParallelEmbedding.__init__` (`layers/vocab_parallel_embedding.py:264`) defaults `org_vocab_size` to the supplied number of embeddings. The override sets that to201088. Its ordinary weight loader (`:390–445`) asserts loaded rows equal `org_vocab_size`; checkpoint rows remain201024. Inkling's loader (`models/inkling2.py:577–647`) delegates unchanged to this parameter loader. No supported vocab-padding CLI was found in arg groups/server args. Increasing configured padded vocab can also resize embedding-related modules, so this is not allocation-only.

Minimal source change to investigate, **not applied**:

```python
self.lm_head = ParallelLMHead(
    config.padded_vocab_size,
    config.hidden_size,
    padding_size=128,
    prefix=add_prefix("lm_head", prefix),
)
```

This retains original/checkpoint row count201024 and allocates201088 rows. The existing weight loader copies checkpoint rows and explicitly zeros trailing allocation rows. No input-embedding padding change is necessary. Target logits are cropped to logical vocab by `layers/logits_processor.py` (`:515`, `:1338`, `:1613`). Draft projection and folded Markov bias both use `gather_and_crop_vocab` (`models/dspark.py:117`, `models/inkling2_mtp.py:90,195`), cropping to original201024 rows; the64 added allocation rows therefore do not become draft choices. Fold buffers derive their physical shape from head weight shape, so they follow padding.

This allocation change preserves checkpoint values and loop count; enabling MXFP8 afterward changes draft arithmetic. It still needs runtime correctness, acceptance, latency and memory validation. Padding alone adds only64*2304*2=294912 bytes to the BF16 head, but the newly enabled MXFP8 cached copy is approximately0.463GB before scales/workspaces, and the original BF16 target head remains live.

## SWA4096 ->3584

`model_executor/pool_configurator.py:1051` derives hybrid SWA capacity as floor(full_tokens*ratio/page)*page. For full capacity131072, ratio0.02734375 gives3584 tokens exactly (28 pages of128).

There **is** a source-enforced startup minimum: `validate_swa_pool_size` (`:281`) rejects `swa_tokens <= sliding_window_size + page_size`. For window1024 and page128 the floor is1152, so3584 passes this particular guard. That guard does not prove sufficient steady-state headroom.

`compute_swa_request_cap` (`:1095`) budgets a more conservative working set:

- trailing slots = window + eviction_interval*draft_tokens + page;
- decode allocation, page-aligned for nonoverlap speculation;
- one prefill chunk without overlap, two with overlap;
- another page, multiplied appropriately for concurrent requests.

`environ.py:808` defaults eviction_interval to128. The actual resolved baseline in `resolved-baseline.json` has speculative_num_draft_tokens8 (gamma7), steps1, topk1, max_running_requests1, chunk2048 and disable_overlap_schedule=false. `mem_cache/allocation_sizing.py:39` returns8 for one decode allocation (topk1); overlap doubles it to16. Thus the generic conservative cap is:

```
trailing = 1024 + 128*8 + 128 = 2176
cap = (2176 + 2*8)*1 + 2*2048 + 128 = 6416
page-aligned cap = 6528
```

This is a **conservative sizing calculation, not a hard allocation requirement**. Explicit ratio sizing does not necessarily enforce it; `pool_configurator.py:1540` explicitly skips cap mode when an operator ratio is present. The working baseline4096 is already below this generic cap and has passed126K recall and reached full131K capacity in prior exploratory tests, according to the parent agent. Consequently the formula alone cannot establish that3584 fails or that4096 is insufficient. Its double-chunk and eviction reserves describe a worst case rather than demonstrated simultaneous occupancy on the DSpark path.

**Recommendation:** retain tested4096 unless a separate controlled experiment demonstrates3584 remains stable and useful.3584 passes the hard startup guard (>1152), but no source-only evidence here proves enough headroom under every prefill/decode/reuse schedule. In particular test long chunked prefill, decode across eviction boundaries, repeated/reused prefixes and admission progress before accepting it. The logical attention window remains1024; reduced allocated capacity can still affect admission/retraction and throughput. Do not reduce the attention window to finance MXFP8.
