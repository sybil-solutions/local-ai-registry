# Recipe kit submissions

Measured runs made with the [recipe kit](https://github.com/sybil-solutions/local-ai-recipe-kit): one JSON per run
(`schema: local-ai-recipe-kit/submission/v1`), built by the kit's `tools/make_submission.py` and checked by its
`tools/validate.py`. Each file holds the probe of the machine, the exact launch (image pinned by digest, argv, env),
the reference-panel quality result, the fixed-protocol speed sweep and, where `memwatch.sh` ran, the measured memory
needs (`needs_measured`).

These are supporting evidence. A file here is not a recipe and has not passed the registry's six lab gates; a recipe
only exists once `lab/lab.py try` passes on the card.

## Layout

`<card id>/<YYYY-MM-DD>-<handle>[-<config>].json`. Add new files in the same pull request as the recipe they support,
or on their own for a measured run that produced no recipe. `validate.py` needs the panel and the speed sweep, so a run
that never got that far is reported on the bounty issue (or a new issue) with its `probe.py` output instead.

## Files

Moved unchanged from the kit's `submissions/` directory (kit commit `ef9782c`); the kit no longer keeps submissions.

| file | target | config | top-1 | mean KL | C1 decode tok/s | prefill 32k tok/s | added in kit |
|---|---|---|---|---|---|---|---|
| `rtx-3090-24gb/2026-09-29-0xSero.json` | qwen3.8-flash-next-offload | sglang-exl3-flashnext, 1x RTX 3090 | 0.9913 | 0.00092 | 63.0 | 2837 | `6de59dd` (reference submission) |
| `2x-r9700-32gb/2026-09-29-benstadin-bprime.json` | qwen3.8-flash-next-offload | vLLM MXFP4, TP 2, profile B', BF16 KV | 0.9700 | 0.01470 | 115.0 | 3983 | `ef883d2` (kit #1) |
| `2x-r9700-32gb/2026-09-29-benstadin-bprime-fp8.json` | qwen3.8-flash-next-offload | as above, FP8 KV | 0.9687 | 0.01502 | 123.4 | 4397 | `ef883d2` (kit #1) |
| `2x-r9700-32gb/2026-09-29-benstadin-m2.json` | qwen3.8-flash-next-offload | vLLM MXFP4, TP 2, profile M2, BF16 KV | 0.9700 | 0.01470 | 104.0 | 3826 | `ef883d2` (kit #1) |
| `2x-r9700-32gb/2026-09-29-benstadin-m2-fp8.json` | qwen3.8-flash-next-offload | as above, FP8 KV | 0.9687 | 0.01502 | 121.0 | 4180 | `ef883d2` (kit #1) |

Numbers are copied from the files (sweep medians). The 3090 file has no `needs_measured`. It is independent history,
not the proof of the current `rtx-3090-24gb/qwen3.8-flash-next.sglang.200k` recipe: it ran image
`sglang-exl3-flashnext@sha256:95b1a85d…`, the recipe's launch pins `@sha256:ce47df25…`, and the recipe's proof is its
own six-gate run. The registry has no 2x R9700 card or recipe yet.
