# LFM3-1.7B on an Apple M5 32 GB: owner-machine evidence

Evidence for `registry/recipes/apple/apple-m5-32gb/lfm3-1.7b.mlx-vlm.256k.json` (engine
`mlx-vlm-lfm3-1.7b-8bit-256k@122c66fdb191`, weights
`LiquidAI/LFM3-1.7B-MLX-8bit@4d6d94c9c2c2f249f1f740f4f08e071b0dbfe099`). One session on one machine, 2026-11-12 UTC:
a 30-second speed probe, then the `lab/lab.py try --on endpoint` run that wrote the recipe (02:16:05 to 02:27:48 UTC),
with the server's memory logged every 10 s.

## The launch: #427's, unchanged

This recipe runs `registry/launches/mlx-vlm-lfm3-1.7b-8bit-256k.json` exactly as the M1 Max recipe does, at the full
262,144 window. The run record's `launch_sha256` (`d451e1a6...dfedd`) is the same as the M1 Max proof's, so the command,
packages, environment, weights and launcher are byte-identical. No file in `registry/launches/` changes.

## Machine

MacBook Pro (`Mac17,2`), Apple M5, 10-core CPU (4 super, 6 efficiency), **10-core GPU**, 32 GB unified memory,
153 GB/s. macOS 27.2 (26B5091g), Python 3.12.13, mlx-vlm 0.7.12, mlx 0.32.4, transformers 5.22.2. AC power.
`iogpu.wired_limit_mb` left at 0. No other model server was running. Details in `host.json`, the whole environment in
`pip-freeze.txt`.

This is the owner's working laptop and it was busy throughout: a browser, an editor and several coding agents were
running, and other agents compiled Rust and TypeScript during the run. The 1-minute load average ran from 7 to 118
and 5.3 to 7.9 GB of swap was in use (`memwatch.log`). The numbers below were measured under that load.

## Results

| | lab run (recipe proof) |
|---|---|
| gates | load chat reasoning tools context speed: 6/6 |
| speed gate | **37.8 tok/s** (1,136 tokens in the first 30 s, counted by the served tokenizer; the answer ran to 10,862 and ended by itself) |
| context gate | 184,722 prompt tokens (server-reported; 70.5% of 262,144), code recalled in a 371-token answer |
| prefill | 485 tok/s over the whole request (381.2 s); 518.9 tok/s as the server times it |
| peak GPU memory (MLX, context request) | 5.36 GiB |
| peak process footprint (`memwatch.log`) | 6,954 MB |

The speed gate clears the 15 tok/s floor by 2.5 times. A 30-second probe before the run, with the lab's own
`stream_rate` and the same prompt, gave 42.4 tok/s. The window meets the 131,072-token KV floor for cards above 8 GB.

For comparison, the October attempts on this card (a field note in the owner's records) found every then-listed
128K-plus route below the speed floor on a base M5: Qwen3.8-27B decoded at 5.0 to 7.5 tok/s. A 1.7B model is what makes
the floor reachable at 153 GB/s.

Not run here: #427's extra needle probe near the full window. The same weights on an 8 GB M1 (#429) read a 127K-token
needle prompt but did not finish their thinking within 12,000 tokens; that is described in
`data/sources/apple-m1-8gb/lfm3-1.7b/README.md` on that PR.

## Files

| file | role | sha256 |
|---|---|---|
| `../../../../lab/runs/apple-m5-32gb.lfm3-1.7b.mlx-vlm-lfm3-1.7b-8bit-256k.256k.20261112T022748.json` | the run record | `abbf9d56eb5291d6b307790efeb9fd110b39da633c7cb6134f9972e5f791ab68` |
| `lab-try.log` | the lab's console output | `f60b98c8097f3a93a65dbd7bebaf8577fd68f34480cadde62ed10cad95bbc341` |
| `run-server.sh` | the measured launch, paths in `$HOME` form | `263386c4d27bb6b87ac96ff32258e1047f2886b7cfb4300227fded8a8cce4467` |
| `run-server.log` | server log: speed probe and lab run | `eed1bc446ffa0a0404a896c13723e1e10ed313dfec1e443ee8ec5a8f6f19ec67` |
| `memwatch.log`, `memwatch.sh` | the server process's physical footprint every 10 s (it includes GPU memory), free memory, swap and load | `784dc9598936304967ea165e98ab8b8f8089d184b8894abeb544eccde452a1fd` |
| `lab-start.timestamp` | when the lab run began | |
| `host.json`, `pip-freeze.txt` | machine facts (identifiers omitted) and the environment | |
