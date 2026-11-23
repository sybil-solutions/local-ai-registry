# Inkling 2 Mini / RTX 5090: hardware and tuning evidence

This bundle records a completed tuning campaign. **No additional serving change was adopted.** The original command remains; repeated official gates pass, while broader tests expose correctness and context-budget limitations. No universal reliability or speed-gain claim is made.

## Reproduction pins and environment

- Weights: [`thinkingmachines/Inkling-2-Mini@ed40e7d`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7).
- Serving image: `lmsysorg/sglang:inkling-2-mini@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de`; reported SGLang version `0.0.0.dev1+inkling2`.
- Methodology: [inference-optimisation skill at fda968e](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0).
- One RTX 5090, 32607 MiB reported, driver 615.78.08; Ryzen 9 9950X3D and 128 GB system RAM. Performance profile and 575 W power limit were left unchanged.
- This is a desktop measurement. The same ChatGPT/T3 display GPU helpers remained resident; unrelated compute jobs were excluded. It is not a headless-server claim or a multi-user capacity result.

Exact identifiers are in [pins.json](pins.json). [hardware-status.json](hardware-status.json) records the hardware run's environment and restoration; [host-runtime.json](host-runtime.json) records the host software. No private desktop command lines or credential paths are included.

## Hardware and conditional ceilings

[hardware.json](hardware.json) is the raw output of [hwbench.py](hwbench.py). Median measured bandwidth was **1529 GB/s**, counting both reads and writes for a 1 GiB device copy. Dense GEMM measured **219.8 TFLOP/s BF16 at 4096³**, **230.8 TFLOP/s BF16 at 8192³**, and **542.7 TFLOP/s FP8 at 4096³**. These are microbenchmarks, not model inference rates.

[theory.py](theory.py) reads only safetensors headers/config and the hardware result. Reproduce the inventory with:

```sh
python3 theory.py --checkpoint /path/to/the/pinned/snapshot
```

[model-components.json](model-components.json) and [THEORY.md](THEORY.md) account for three visits through the loop core, FP4 experts in layers 0–19, FP8 experts in layers 20–23, dense/head weights, and full versus local KV. The 128K BF16 full KV allocation is 4.5 GiB including the draft; the 4096-token SWA pool adds 0.656 GiB. State, graph, allocator and workspace memory are additional.

The decode calculations are **conditional bandwidth scenarios**, not a single measured or achievable ceiling. They depend on accepted tokens per step, overlap between routed experts, query-row reuse and the draft Markov path. The draft transformer is block-parallel; multiplying its entire weight traffic by seven serial draft passes would be incorrect. No percentage-of-ceiling claim is justified without profiling actual acceptance/routing and traffic. A mixed NVFP4/FP8 grouped-MoE prefill ceiling is not measured: dense BF16/FP8 GEMM cannot substitute for it.

## Exploratory findings retained in full

[discarded-and-inconclusive-settings.json](discarded-and-inconclusive-settings.json) records the decisions and their evidence:

- **Gamma 3:** startup rejected the change because Inkling MTP requires the checkpoint block size. Gamma 11 was not completed; no measured result is claimed for it.
- **1024-token prefill chunks:** a completed response emitted 48,284 visible words for a request for about 2,500, with extensive repetition. Its higher token rate was not a successful-answer speed gain.
- **Compact verification:** the first prose request exhausted its available context after 123,007 reasoning tokens and produced no final answer. Rejected.
- **MXFP8 draft projection pilot:** source inspection found that 201024 head rows fail the helper's 128-row divisibility guard. The option silently falls through to ordinary matmul. The pilot does not demonstrate an MXFP8 gain.
- **Initial 4096-token prefill attempt:** a cache-flush timeout occurred during warmup before any timed sample. Inconclusive; stronger generation-readiness checks and a controlled retry were preregistered.

[exploratory-prose-summary.json](exploratory-prose-summary.json) includes timings, actual usage, finish reasons, repetition measures and output hashes. [exploratory-prose.jsonl.gz](exploratory-prose.jsonl.gz) retains all recorded prose responses, including rejected outputs and both baseline repeats. [exploratory-method.json](exploratory-method.json) describes the prompt, sampling and counting; [gamma3-startup-excerpt.txt](gamma3-startup-excerpt.txt) preserves the relevant exception.

These exploratory settings were not each bracketed by neighbouring controls and cannot establish a gain. Their token-rate fields remain raw evidence, not headline performance claims. Natural completion alone is not semantic correctness; scope, repetition, exact retrieval and task grading are separate checks. The pinned model card itself documents looping as a model limitation, so these examples do not establish a unique causal link to any one runtime setting.

## Source investigation and qualification boundary

[draft-padding-investigation.md](draft-padding-investigation.md) explains the MXFP8 guard and a possible allocation-padding source change. **That source change was not applied or benchmarked here.** It would require a new pinned build plus correctness, memory and performance qualification. The same note explains why reducing allocated SWA capacity cannot be justified solely from its startup minimum; the tested 4096-token allocation remains the reference. [resolved-baseline.json](resolved-baseline.json) preserves the relevant resolved settings.

The later skill-guided protocol uses the fixed six-task panel, adjacent control/candidate/control runs, output sanity and greedy-output checks, and independent startup/decode/latency/TTFT gates. The skill's example 512-token caps are omitted for registry performance runs; actual prompt sizes come from the tokenizer. Official registry acceptance rates, exploratory stress-test rates and controlled-panel medians are different measurements and must remain labelled separately. Final results and qualification are supplied by the accompanying report.

## Controlled tuning decision

No additional serving setting was adopted. The frozen 128K profile retains its 2048-token prefill chunks, 4096-token sliding-window pool, 8 state slots, BF16 KV and default model loops. The daily 48K service and 4096 output budget remain unchanged.

Persistent compiler caches reduced warm startup from 126.7 seconds (median of 3 adjacent controls) to 49.5 and 37.5 seconds. Both candidate runs failed the predeclared combined performance gates. Their sampled outputs varied in length, so this is a measured startup improvement with an inconclusive broader performance result, not proof that caching causes a regression. Six short greedy reasoning+content hashes matched across all controls and cache candidates. The temperature-zero full-panel follow-up was stopped because its baseline story generated 91805 reasoning tokens without an answer before explicit abort. That invalid baseline cannot support a useful-answer speed comparison. The first candidate was canceled during startup; remaining deterministic entries were not run.

The 4096-token prefill retry passed generation readiness but its first 8K warmup hit CUDA OOM while allocating 138 MiB. No candidate speed sample was collected. Earlier gamma changes were incompatible; compact verification and 1024-token chunks produced unacceptable outputs. The MXFP8 draft-head flag fell through to BF16 because its 201024 rows fail 128-row alignment. An allocation-padding source fix was investigated but not built or benchmarked. It would need a separately pinned engine and memory/quality qualification.

## Quality and context interpretation

The six automatic panel checks detect termination, loops, scope and transport failures; they do not prove that an answer is correct. Independent review of the twelve 32K/112K answers found 7 passes with limitations and 5 substantive failures. These included non-runnable LRU tests, incorrect backup retention, a TCP condition error and a historical chronology error. This is a small retrospective review, not a standardized quality score or evidence that long context caused the errors.

All eight initial three-code retrievals passed at approximately 64K and 126K input, across two seeds and cold/warm requests. One cold 64K request required 59660 output tokens and 334 seconds despite answering correctly. Prefix reuse was confirmed by server counters in all four warm requests, but greedy reasoning hashes differed between cold and warm. Thus neither whole-answer latency nor byte-equivalent generation should be inferred from a cache-hit counter alone.

The small synthetic vision, tool-roundtrip, streaming and exact-marker checks passed. A blue-square image is a smoke test, not a vision-quality benchmark. The uncapped Daily replay uses frozen original prompts/rubrics with only the request output cap removed; it is separate from the public 4096-budget ranking.

## Reproduction and remaining work

`runs-data.tar.gz` retains complete per-request outputs and failed/canceled run records; `telemetry-logs.tar.gz` retains sampled GPU state and server logs; `qualification-data.tar.gz` retains context tests, grading and frozen fixtures. `package-provenance.json` records exact original and published hashes. Private home paths are redacted, and portable source adaptations are explicit. The source image, engine commit, package inventory, model revision and methodology commit are pinned.

The hardware measured 1529 GB/s copy bandwidth and 230.8 TFLOP/s BF16 dense GEMM. DSpark traffic estimates are conditional on acceptance, expert overlap and cache reuse; they are not a single practical speed ceiling. No mixed NVFP4/FP8 grouped-MoE prefill ceiling was measured. Remaining performance work would require kernel profiling and a new engine build, especially the draft-head alignment issue. No physical-maximum performance claim is made.

## Final validation results

The unchanged exact published launch passed all six official gates twice, with prefix caches flushed before each wave. Wave 1 measured 310.0 tok/s over 9069 tokens/29.3 s; wave 2 measured 163.8 tok/s over 1392 tokens/8.5 s. Both retrieved the code from 91890 prompt tokens. All records are retained; the latest proof is used, not the fastest. The unmodified lab stores token counts rather than full story text, so these are narrow gate results, not semantically reviewed story scores or a causal optimisation gain.

The uncapped frozen Daily suite passed 12/12 on each of two passes (16 manually reviewed nonrepair responses plus 8 sandbox-graded repair attempts). The broader 12 answer panel had 5 substantive errors, as described above. No public Daily ranking was changed.

The c 1 soak ran 903.29 seconds wall time, including 765.71 seconds in generation requests, across 34 cycles/68 requests. Prompt construction/tokenizer checks account for the remainder; this was not 15 minutes of uninterrupted GPU saturation. Near 126K retrieval passed 31/34;3 cases exhausted the context before producing an answer. All 34 fixed probes produced byte-identical reasoning+content, with first/last 11-sample median usage-decode rates 241.154→241.147 tok/s (−0.0028%). No environment change, watchdog error, memory-cap trip or server crash was recorded. Thus transport/performance stability and the failed extended correctness gate are separate results.

Daily 48K serving and OMP were restored. Power profile and 575 W limit were unchanged. The unused compiler-cache volume and local experiment artifacts are retained for reproduction; no modified engine build was deployed.
