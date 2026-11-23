# LFM3-1.7B on an Apple M1 8 GB: owner-machine evidence

Evidence for `registry/recipes/apple/apple-m1-8gb/lfm3-1.7b.mlx-vlm.128k.json` (engine
`mlx-vlm-lfm3-1.7b-8bit-128k@22218be9e29b`, weights
`LiquidAI/LFM3-1.7B-MLX-8bit@4d6d94c9c2c2f249f1f740f4f08e071b0dbfe099`). One session on one machine, 2026-11-11 to
2026-11-12 UTC: a 30-second speed probe, then the `lab/lab.py try --on endpoint` run that wrote the recipe
(23:17:58 to 23:29:21 UTC), then two needle requests near the full window, with the server's memory logged every 10 s.

## Machine

Mac mini (M1, 2020), `Macmini9,1`, Apple M1, 8-core CPU (4P + 4E), **8-core GPU**, 8 GB unified memory, 68.25 GB/s.
macOS 26.4.1 (25E253), Python 3.12.15, mlx-vlm 0.7.12, mlx 0.32.4, transformers 5.22.2. AC power.
`iogpu.wired_limit_mb` left at 0. Two small background servers ran throughout (about 0.4 GB together); no other
model server was running. Details in `host.json`, the whole environment in `pip-freeze.txt`.

## The launch: the 256K launch with a smaller window

`registry/launches/mlx-vlm-lfm3-1.7b-8bit-128k.json` is `mlx-vlm-lfm3-1.7b-8bit-256k.json` with two fields changed:
`id` and `ctx` (262,144 to 131,072), plus its `about` text. The command, packages, environment, weights and launcher
(`mlx-vlm-lfm3-serve.py`, sha256 `291fa7d3743b07ee46a89345640afb3eb81f5a40caf4255890e731fb0074e935`) are the same bytes,
so this adds no runtime: it is the pinned mlx-vlm runtime that #427 established, run on a smaller Mac.

The window is smaller because 8 GB cannot hold the full one. On the M1 Max the 255K probe peaked at 6.73 GiB of GPU
memory and an 8.72 GB process footprint. Here the whole machine has 8 GB, and by default macOS lets the GPU wire only about
two-thirds of it. A frozen host launch fixes `ctx`, so a smaller window needs its own launch file.

## Results

| | lab run (recipe proof) |
|---|---|
| gates | load chat reasoning tools context speed: 6/6 |
| speed gate | **28.5 tok/s** (855 tokens in the first 30 s, counted by the served tokenizer; the answer ran to 5,792 and ended by itself) |
| context gate | 91,882 prompt tokens (server-reported; 70.1% of 131,072), code recalled in a 120-token answer |
| prefill | 244 tok/s over the whole request (376.4 s); 249.8 tok/s as the server times it |
| peak GPU memory (MLX, context request) | 3.58 GiB |
| peak process footprint during the run (`memwatch-1.log`) | 4,235 MB, no swap used |

The speed gate clears the 15 tok/s floor by 90%. A 30-second probe before the run, with the lab's own `stream_rate`
and the same prompt, gave the same 28.5 tok/s.

## Near the full window: the window fits, but the answer did not end

After the lab run, two needle requests filled 97% of the window (127,068 and 127,108 tokens, sized with the served
tokenizer). Both read the whole prompt and stayed in memory. Neither answer ended. This section says exactly what
happened, because it bears on how the recipe behaves at the top of its window.

| | probe 1: `probe_full.py` from #427, unchanged | probe 2: `probe_capped.py` |
|---|---|---|
| sampling | temperature 0.6 (as in #427's probe) | the model's own `generation_config` (temperature 0.1), as the lab's gates use |
| answer cap | none | 12,000 tokens |
| prompt tokens | 127,068 | 127,108 |
| prefill | 612.5 s, 207.4 tok/s | 621.0 s, 204.7 tok/s |
| peak GPU memory (MLX) | not reported (the request never returned) | 4.26 GiB |
| peak process footprint | 5,427 MB (`memwatch-1.log`) | 5,375 MB (`memwatch-2.log`) |
| decode | 9 to 12 tok/s | 12.9 tok/s |
| outcome | still thinking after 28,860 tokens (38.9 min); stopped by hand | `finish_reason: length` at 12,000 tokens, no answer |

Probe 2 saved its reasoning (`probe-capped-127500-default.json`). The model found the code in its first sentence
(`We need to find "The access code for the vault is 90748." from line 1380?`), then began copying the filler lines
back into its reasoning to locate the line and was still copying when the cap ended it. The weights, launcher and
kernels are the same as on the M1 Max, and memory never ran out, so this is the model's thinking on a near-full
filler prompt rather than a limit of this Mac. On the M1 Max, #427's probe at 255,365 tokens ended in 3,054 tokens;
the lab's own context gate here, at 91,882 tokens, ended in 120. Two requests are not enough to say how often it
happens, and this run did not test prompts between 92K and 127K.

Practical upshot for a reader: the declared 131,072-token window loads and is read on an 8 GB M1, but an uncapped
request near the top of it can think for a long time. A client that sets `max_tokens` keeps such a request bounded.

When probe 1's client was stopped, the server kept decoding the abandoned request, so the server was restarted for
probe 2; `run-server-1.log` and `memwatch-1.log` cover the speed probe, the lab run and probe 1, and
`run-server-2.log` and `memwatch-2.log` cover probe 2. Memory was tight during both probes: free memory fell below
100 MB at times and swap rose to a few hundred MB, while nothing failed. The Mac dropped off the tailnet for a few
minutes around 00:15 UTC during probe 1 and came back by itself without a restart; the server and the request ran
through it.

## Files

| file | role | sha256 |
|---|---|---|
| `../../../../lab/runs/apple-m1-8gb.lfm3-1.7b.mlx-vlm-lfm3-1.7b-8bit-128k.128k.20261111T232921.json` | the run record | `31f21c522ab347d07130135711b1ceea864fa7b3cadcb84345d85f3a018cb30d` |
| `lab-try.log` | the lab's console output | `61de34c9626f9a12ed4a08cdf96d3c8f29ffa5a77c36b847a8309d22ef513973` |
| `run-server.sh` | the measured launch, paths in `$HOME` form | `6f7110bf7f425980108bb8c9f40f6bfea95fcae7338e107f6c50ea336be0314d` |
| `run-server-1.log` | server log: speed probe, lab run, probe 1 | `6874f5c815d5f120befc14326abfb4da4fd02cfb847a9815a90d358e2172f0c7` |
| `run-server-2.log` | server log after the restart: probe 2 | `7e25a54bdbe470f9e93efd9b537d6f939ab56f3db010b0394eb6a3bebd14f5c7` |
| `memwatch-1.log`, `memwatch-2.log`, `memwatch.sh` | the server process's physical footprint every 10 s (it includes GPU memory), free memory, swap and load | `e367b13498c5010e9bdd888530ed876f0a96ecb78b852331b583f14c7f486749`, `8ed5e2eb12ebe4bbf10ef12ebb59131343ed48dd6743dc25e122abebac8002fb` |
| `lab-start.timestamp`, `probe-start.timestamp` | when the lab run and probe 1 began | |
| `probe_full.py` | not copied: probe 1 ran `data/sources/apple-m1-max-32gb/lfm3-1.7b/probe_full.py` unchanged; it never wrote a result | |
| `probe_capped.py`, `probe-capped-127500-default.json` | probe 2 and its result, with the head and tail of its reasoning | `661206582787f532bf1d2f8453054bfec3dd4888d1c244271f903c89bcdc5299`, `d7cf99384ca9c9b6ba3b16202507acd457339e0ec700e586b88b120a21b4c8e0` |
| `host.json`, `pip-freeze.txt` | machine facts (identifiers omitted) and the environment | |
