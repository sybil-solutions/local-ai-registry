# Inkling 2 Mini on an RTX 5090 32 GB: Siphon, 128K

[Siphon](https://github.com/hxyulin/siphon-engine) is a Rust and CUDA inference engine for one model, Inkling 2 Mini, on one RTX 5090 (`sm_120a`). This recipe runs the image `ghcr.io/hxyulin/siphon-engine@sha256:3831a21e…`, built from commit `cc362139e69267bb82c9c7e876d784b5fe18f7c7` by the repo's [image workflow](https://github.com/hxyulin/siphon-engine/blob/cc362139e69267bb82c9c7e876d784b5fe18f7c7/.github/workflows/image.yml). The workflow attests the build's provenance: `gh attestation verify oci://ghcr.io/hxyulin/siphon-engine@sha256:3831a21ed89f4870539d66462c93377e317f8e39fbf29571dfc1e9641886c8ad --repo hxyulin/siphon-engine`.

## Setup

- Weights: `thinkingmachines/Inkling-2-Mini@7a67ae23`, the official checkpoint unchanged (NVFP4 and MXFP8 routed experts). Siphon makes FP8 copies of the dense weights at load (`--weights fp8`) and keeps the global KV cache in FP8 (`--kv fp8`).
- Context: 131,072 positions, one stream. The server plans GPU memory before loading: 21.32 GiB planned, 21.25 GiB in use after warmup.
- Decoding: speculation from the checkpoint's bundled DSpark draft. Siphon speculates by default, so no flag names it and the lab's `spec` field is null.
- Sampling: temperature, top-k, top-p and min-p run on the GPU in vLLM's order. Fields a request leaves out come from the checkpoint's `generation_config.json` (temperature 1). Speculation keeps a drafted token only when it equals the token the model draws for that request's seed and position, so sampled output has the model's filtered distribution exactly. At `temperature: 0`, speculation returns exactly the tokens of plain greedy decode.
- Prefix cache: the server keeps the last prompt's state at its last multiple of 2,048 tokens, so a follow-up turn prefills only its new tokens and gets the same tokens as without the cache.

## Lab run

`lab.py try --on endpoint` against the container, pulled from GitHub Container Registry, on the owner's RTX 5090 (driver 620.73.04, 600 W limit), with only the desktop compositor on the GPU (35 MiB).

| Gate | Result |
| --- | --- |
| load | `/v1/models` lists `inkling-2-mini` |
| chat | "red, yellow, blue" |
| reasoning | "391", 106 characters of thinking |
| tools | `get_weather {"city":"Paris"}`, then the reply quoting the tool's code |
| context | 91,891-token prompt, answer "58213", 6.9 s for the whole request |
| speed | 511.0 tok/s, counted with the server's `/tokenize` |

The lab samples the speed gate's story at temperature 0.8, so its tok/s depends on the story the model writes. This run's story was 3,070 tokens at 2.87 tokens per verify round. On an earlier image (commit `ca19600`), three runs of the same gate gave 520, 335 and 332 tok/s, and a greedy run gives 667 tok/s.

## Files

- `evidence/lab-try.log`: the lab's output.
- `evidence/server.log`: the container's log for the run: memory plan, verify timings by width, and one line per request.
- `evidence/rendered-launch.json`: `lab.py render` of the recipe.
