# DeepSeek-V4.1-Flash-1017 on 4x RTX PRO 6000 Blackwell — the served configuration, lab-accepted

Evidence for `registry/recipes/nvidia/rtx-pro-6000-blackwell-96gb/deepseek-v4.1-flash.vllm.512k.4x.moet-r21.json`
(engine `vllm-moet-deepseek-v4.1-flash-1017-512k-tp4-r21@21d8fcb5deb6`, weights
`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`). One session on the owner's machine,
2026-11-12, 18:46–19:40 UTC: the launch was started exactly as the registry renders it, by the Local AI app on that
machine; `lab/lab.py try --on endpoint` ran its six gates against it and wrote the recipe; then the standard speed
table was measured on the machine itself.

## What this launch is

The configuration the owner has served on this machine since 2026-10-27, now expressed as a registry launch. Against
the r21 launch that was in the registry as *reported* (sybil-solutions/local-ai-registry#428, from #367) it changes:

| | reported r21 launch | this launch |
|---|---|---|
| DSpark draft tokens | 7 | **5** |
| `--max-num-seqs` | 8 | **16** |
| `--max-cudagraph-capture-size` | 64 | **128** |
| `--max-logprobs` | default | **128** |
| served name | `deepseek-v4.1-flash` | **`deepseek-v4.1-flash-1017`** |
| engram_disk library | v1 (the image's default) | **v3** (`VLLM_MOET_ENGRAM_DISK_LIB`) |
| image | `…@sha256:ef1e28b71902…` | `…@sha256:21d8fcb5deb6…` (the same r21 build plus `libengram_disk_v3.so`) |

The image was built by `release-image.yml` of sybil-solutions/local-ai-images at commit `36c3d5b22627`
(run 43132033722); `image-attestation.json` is `gh attestation verify` on its digest. The v3 library in the image is
byte-identical to the one the owner's own service loads (sha256 `16fb4d1b729e…`), and `engine.log.gz` and
`/proc/<worker>/maps` showed it mapped in the tensor-parallel workers of this run.

One thing a registry launch cannot carry over from the owner's service: that service keeps FlashInfer's autotune
results, the DeepGEMM JIT cache and the KV disk tier in host directories between starts. A registry launch names no
host paths, so this one redoes that work at every start (8 min 23 s here) and its KV disk tier starts empty.

## Machine

`host-and-container.json`: AMD Ryzen Threadripper PRO 9965WX (24 cores, 48 threads), 125 GiB RAM, 4x NVIDIA RTX PRO
6000 Blackwell Workstation Edition 96 GB on PCIe 5.0, **300 W power limit per GPU**, driver 595.84, Pop!_OS 22.04,
kernel 6.12.10, Docker 29.1.3. The same file holds the container as Docker ran it: image by digest, arguments,
environment, 48 GiB `/dev/shm`, `memlock` unlimited, `CAP_IPC_LOCK`, bridge network, the weights mounted read-only.

The machine was otherwise in use: 13 logged-in users' sessions, a 1-minute load average of 4 to 10 on 48 threads,
and the shell of a stopped model server holding 7.3 GiB of RAM. Nothing else was on the GPUs.

## Results

| | |
|---|---|
| start to ready | 8 min 23 s (18:46:35Z to 18:54:58Z), nothing cached |
| lab gates | load chat reasoning tools context speed — **6/6** (`lab-try.log`, run record `lab/runs/rtx-pro-6000-blackwell-96gb.deepseek-v4.1-flash.vllm-moet-deepseek-v4.1-flash-1017-512k-tp4-r21.512k.20261112T185616.json`) |
| lab decode, 1 stream | **407.7 tok/s** (3,624 tokens in the 8.9 s after the first token, counted with the served tokenizer; a 600-word story at temperature 0.8) |
| lab prefill | **13,533 tok/s** (351,853 prompt tokens, cold and unique, 26.0 s for the whole request; the planted code came back) |
| GPU KV cache | 9,139,310 tokens (17.43x a 524,288-token request) |
| host memory | MemAvailable fell from 88.41 GiB to 38.23 GiB at its lowest (19:11:34Z): **50.2 GiB (53.9 GB)**, inside the launch's `needs.host_ram_gb` of 66 |
| GPU memory | 97,168 MiB at the highest on one card (of 97,887) |
| GPU power | 296 W per card on average during the speed table |
| the app's own probe at ready | 392.4 tok/s on 1 stream, 661.6 total on 4, 936.9 total on 8 (short answers) |

The lab ran on the owner's laptop and reached the machine over the internet (200 to 800 ms round trips), so the lab's
prefill figure includes that delay and the upload of the prompt.

### Speed table, measured on the machine

`sweep/sweep.py` on the machine against `127.0.0.1`. Every request has a context of its own that the engine has not
seen (a random nonce first; no row is flagged for a cache hit), asks for an essay of about 3,000 words at temperature
0.7 and sets no output limit; answers were 2,921 to 6,549 tokens and ran 13 to 40 s. One warm-up wave, then the median
of three waves. Prefill is prompt tokens over the time to the first token (at 2 and 4 streams: all prompts' tokens
over the slowest first token). Decode at 2 and 4 streams is the total inside the window in which all streams were
decoding, with the per-stream median in brackets.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| none | — | 285.7 tok/s | 1 | FP4 KV record (`nvfp4_ds_mla`), 32 GiB host offload + disk tier | 4 |
| 8k | 15,280 tok/s | 254.0 tok/s | 1 | same | 4 |
| 8k | 14,878 tok/s | 324.5 tok/s (164.5) | 2 | same | 4 |
| 8k | 13,791 tok/s | 480.0 tok/s (120.4) | 4 | same | 4 |
| 32k | 14,920 tok/s | 242.4 tok/s | 1 | same | 4 |
| 32k | 14,819 tok/s | 311.6 tok/s (152.0) | 2 | same | 4 |

Rows 8k at 4 streams and both 32k rows are from `sweep/on-host.json` (19:21:46Z to 19:32:23Z). During the first
three rows of that run another client was sending requests to the same engine (the access log shows 44 of them
between 19:21:40Z and 19:26:10Z), which lowered those rows; they were measured again with new prompts in
`sweep/on-host-first-rows-again.json` (19:35:24Z to 19:39:41Z, one outside request in that window), and the table
uses the second measurement for them. The same client returned at 19:32:07Z, during the last wave of 32k at 2
streams (300.3 tok/s against 311.6 and 316.7 in the other two).

Why one stream is 254 to 286 tok/s here and 407.7 in the lab's gate: decode speed with DSpark depends on how many
drafted tokens each step accepts, and that depends on the text. `engine.log.gz` reports a mean accepted length of
3.2 to 4.0 tokens per step during the lab's gates (a story the model first drafts in its thinking and then writes
out) and 2.0 to 2.5 during these essays.

## Files

- `lab-try.log` — the lab run's output.
- `host-and-container.json` — the machine and the container as it ran.
- `image-attestation.json` — the image's build attestation.
- `engine.log.gz` — the engine's log from start to the end of the measurements, without the per-request access lines.
- `memory-load-and-lab.log`, `memory-sweeps.log` — MemAvailable, GPU memory and GPU power every 2 s (`memwatch.sh`);
  the second also covers a sweep run from the laptop and a run whose prompts repeated it, neither of which is
  reported here.
- `sweep/` — the script, both result files with every wave, and their logs.
