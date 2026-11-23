# radeon-890m-48gb: supporting evidence (2026-11-23)

Measured on a GMKtec EVO-X1 Pro (Ryzen AI 9 HX 470, Radeon 890M, 64 GB LPDDR5X-6400, Windows 11 Pro
26300), the machine that owns this card. Nothing here is a recipe: the recipe is
`registry/recipes/amd/radeon-890m-48gb/ornith-1.5-35b-a3b.llama.cpp.128k.json` and its proof is its own
six-gate `lab.py try` run.

- `2026-11-23-radeon-890m-probe.json`: the Windows port of `tools/probe.py`, same fields, WMI plus the
  Vulkan device for the GPU. Adrenalin 32.0.23027.2005, 48792 MiB UMA aperture, 46353 MiB free at capture,
  four LPDDR5X-6400 modules at 6400 MT/s.
- `2026-11-23-radeon-890m-memwatch.csv`: the Windows port of `tools/memwatch.sh`, the same four columns,
  one sample per line while the server prefilled 32687 tokens and decoded. GPU in use 30757 to 30793 MiB,
  host memory available 15.26 to 15.78 GB, pagefile 3.52 to 3.72 GB: flat, nothing swapped out.
- `2026-11-23-radeon-890m-gates.json`: the `lab.py try` record behind the recipe's second proof
  (16.5 tok/s, prefill 153, LF-normalised log `sha256:c65086f6de59dcdf`).
- `2026-11-23-radeon-890m-gates-rerun.json`: the record behind the recipe's first proof (17.8 tok/s,
  prefill 145, `sha256:3f5c8f6bdc8645dc`). The first run's client was killed by its harness at a 600 s
  timeout between writing its record and writing the recipe, so a second identical run was already in
  flight. It finished at 14:49, passed the same six gates on the same launch hash, and wrote the recipe
  with its own proof first and the earlier passing proof kept behind it. Both runs used the same command,
  build and weights. The difference between their numbers is run-to-run variance.

No `validate.py` submission file is included. It needs the reference panel and the speed sweep, the sweep
posts to an SGLang/vLLM `/generate` endpoint that llama.cpp does not serve, and no reference panel exists
for a card nobody has run. Per this directory's README the run is reported with its probe output instead.
The measured speed is the registry's own speed gate: 16.5 tok/s over the first 30 s of a stream, 153 tok/s
prefill at 106252 prompt tokens, and a separate 32k run at 255.7 tok/s prefill with the needle answered.
