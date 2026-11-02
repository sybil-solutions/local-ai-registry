# Getting more data

A recipe is only as good as the run behind it. There are three ways a recipe gets here, from strongest to weakest proof, and one rule: **no GGUF**. New recipes use EXL3 (ExLlamaV3, or SGLang with EXL3 kernels), or NVFP4, FP8 and AWQ on vLLM or SGLang.

## GGUF

- **NVIDIA and Intel: none in production.** Every NVIDIA card runs EXL3 (TabbyAPI or SGLang-EXL3) or NVFP4/FP8 (vLLM, SGLang), and the Arc Pro B70 runs EXL3 through vLLM on XPU. The GGUF recipes they had are kept in `data/archive/gguf-recipes/`.
- **AMD: GGUF only until a ROCm run passes.** Nobody rents Radeon cards (neither Vast nor RunPod lists any), so these need an owner: vLLM or SGLang on ROCm with FP8/AWQ weights, through `lab.py try --on endpoint`. The MI300X is the one AMD card RunPod rents.
- **Ranking:** a GGUF recipe never outranks another recipe on the same card (`catalog.py`).

## Where the cards are

Vast has most consumer and pro NVIDIA cards, some only on unverified hosts (`--any-host`; the gates are the same). RunPod adds the RTX PRO 4000/4500/6000 and the MI300X; a pod that never starts on one RunPod cloud is retried on the other. Neither has AMD Radeon, Intel Arc, laptops, the RTX 2000 Ada or the DGX Spark: those take `--on endpoint` from an owner, or a `proxy` from a sibling card (`lab.py proxy`).

## 1. The lab rents the card (tested)

`lab/lab.py try` rents the card on Vast, runs the six gates and writes the recipe only if all pass. It takes any template engine profile:

| Profile | Engine | Weights | Cards |
|---|---|---|---|
| `tabbyapi-exl3` | TabbyAPI on ExLlamaV3 1.5.1 | EXL3 | every NVIDIA card from 8 GB up (CUDA 13.2 hosts) |
| `sglang-exl3` | SGLang with EXL3 kernels (`ghcr.io/0xsero/sglang-exl3`) | EXL3 | Ampere and Ada, 24 GB up |
| `vllm` | vLLM 0.30.0, upstream image | NVFP4 (Blackwell), FP8 (Ada, Blackwell), AWQ/GPTQ, BF16 | any NVIDIA card |
| `sglang` | SGLang 0.5.20 (CUDA 13), upstream runtime image | NVFP4, FP8, AWQ/GPTQ, BF16 | any NVIDIA card |

```
python3 lab/lab.py try nvidia/Qwen3.8-27B-NVFP4@482ca0f3… --model qwen3.8-27b --engine vllm \
    --card rtx-5090-32gb --set ctx=65536 --set mem=0.92
python3 lab/campaign.py lab/plans/vllm-nvfp4.txt --engine vllm --jobs 5
```

Settings a recipe may change: `ctx`, `seqs`, `mem` (GPU memory fraction), `kv` (KV cache dtype), `tools` and `think` (parsers), `extra` (any further server flags, as one string). The plan files in `lab/plans/` are the campaigns: one line per card and build.

What fits where:
- **Blackwell (RTX 50, PRO 4000/4500/6000):** NVFP4 on vLLM or SGLang first (native FP4), EXL3 as the second pick.
- **Ada (RTX 40, 4000/6000 Ada):** EXL3 on TabbyAPI or SGLang-EXL3; FP8 on vLLM or SGLang for 48 GB cards.
- **Ampere (RTX 30, A6000):** EXL3; AWQ on vLLM.
- **8-16 GB cards:** EXL3 at 2-4 bpw is the only way to fit 27B-class models; NVFP4 fits 9-12B models on 16 GB Blackwell.

Cost: a run is 10-40 minutes on a $0.3-2.5/h machine.

## 2. An owner runs the lab on their own machine (tested)

Some cards are never for rent: the DGX Spark, AMD and Intel workstation cards, NPUs. Anyone who owns one starts the server themselves and points the lab at it:

```
python3 lab/lab.py try <repo>@<commit> --model <id> --engine <profile> --card dgx-spark-gb10-128gb \
    --on endpoint --endpoint http://<host>:8000 --gpu "DGX Spark"
```

The same six gates run; the proof says `on: owner`. This is how the Spark recipes published by MiaAI-Lab become tested ones: MiaAI-Lab (or anyone with Sparks) runs `try --on endpoint` against their running server and opens a pull request with the recipe and `lab/runs/` evidence.

## 3. Someone publishes a launch (reported)

People publish working launches with measured numbers. `lab/import_reported.py` reads them from `data/reported/<source>/<repo>.json` (the launch copied from the source repository, pinned to its commit) and writes a frozen profile and a recipe marked `reported`: the site shows it as reported, links the source, and lists it below the tested picks. Only a lab run makes it a tested recipe.

The first source is [MiaAI-Lab](https://github.com/MiaAI-Lab): 36 repositories, 78 launches, 63 recipes after keeping the fastest variant per file name (43 on the DGX Spark, the rest on the RTX 5090, PRO 6000, 6000 Ada and the 16 GB cards). Left out: GGUF launches, an abliterated build, and one launch with no stated card. `data/reported/miaai-lab/ENGINES.md` describes their engines (an ExLlamaV3 fork with an aarch64 build for GB10 and DSpark/DFlash2 drafts, sparkring, tool-eval-bench).

## Swapping engines (ExLlamaV3 to SGLang-EXL3)

A card's pick changes engine only after a controlled head-to-head on that card (`lab/compare.py`); until then a
lab-template SGLang-EXL3 recipe ranks below the ExLlamaV3 one for the same model.

- **Controls.** Both arms passed the six gates, with the same weights (repo@revision), context window and number of
  requests served at once (at least 8, so no level queues). They use the same sampling and seed and the same prompts
  in the same order. Both run on the same rented machine: A runs and is destroyed, then B is rented on that machine.
  compare.py refuses a pair that differs in any of these.
- **Workload.**
  - Prefill: 5 waves per level of cold prompts with a unique random prefix.
  - Decode: 3 waves per level of your own agent sessions (`~/tuning-kit/omp_replay_corpus.jsonl`), replayed turn by
    turn with tools and thinking and no output cap.
  - Levels: C = 1, 2, 4, 8.
  - Tokens are counted with the model's tokenizer on the client.
  - The corpus stays private; it goes only to the rented host.
- **Decision.** A bootstrap 95% confidence interval of median(B)/median(A) for prefill, per-stream decode and
  aggregate decode at each level. B replaces A only if every interval is above 1.0, with no failed request.
  - The verdict and intervals go to `registry/prefer.json`; `catalog.py` reads it.
  - Every sample goes to `lab/runs/`.
- **Why.** An unmatched first run on the 3090 Ti (different bpw; SGLang serving one request at a time, TabbyAPI two)
  looked like a clear SGLang decode win. It was a queueing artifact. Two runs of the same engine on one GPU also
  differed by up to 80% in aggregate decode, so single numbers decide nothing.

## Next

- Run the MiaAI-Lab Spark launches through `try --on endpoint`, starting with the three picks.
- A `exllamav3` template profile: the upstream ExLlamaV3 server (or MiaAI-Lab's fork for GB10) in an image built by `images/`, so EXL3 runs without TabbyAPI.
- tool-eval-bench as a seventh, scored check once a pinned version and fixed flags are chosen.
