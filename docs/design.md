# Design

The registry answers one question for a program: on this GPU, which model should I run, and exactly how? It holds only what has been run and checked. The data behind it lives in [local-ai-data](https://github.com/0xSero/local-ai-data).

## Four kinds of file

| File | What it is | Written by |
|---|---|---|
| `cards/<vendor>/<card>.json` | A GPU: name, vendor, backend, memory, bandwidth, and the `match` block programs use to recognise it | Hand, rarely |
| `engines/<profile>.json` | An engine template (`tabbyapi-exl3`, `vllm`, `sglang`, `sglang-exl3`): pinned image, entrypoint, arguments, environment, port, config file, and `defaults` a recipe can change | Hand, reviewed |
| `launches/<name>.json` | A launch frozen exactly as it was validated before the lab, or as its publisher runs it; one recipe points at it until a lab run replaces it | `lab/import_data.py`, `lab/import_reported.py` |
| `recipes/<vendor>/<card>/<model>.<engine>.<ctx>k[.<n>x].json` | A recipe: weights at a commit, a profile pinned to its image digest, settings that differ from the profile's defaults, the card, and the proof | `lab/lab.py` only |
| `dist/catalog.json`, `plugin/v2/recipes.json` | Everything above, rendered, with at most 3 picks per card setup | `make` |

`lab/models.json` names each model (family, release date) and each build's size and format.

## A recipe is output

`lab.py try` runs weights on a card and writes the recipe only if all six gates pass:

| Gate | Passes when |
|---|---|
| load | The server lists the model within an hour |
| chat | A plain question gets an answer that ends on its own |
| reasoning | The thinking comes back separately and a two-step train question is answered 130 |
| tools | A weather tool is called with the right city, and its result is used in the reply |
| context | A code planted in a prompt filling ~85% of the window is recalled |
| speed | Decode over the first 30 seconds of an answer is at least 15 tok/s |

No answer is ever capped; each runs to its natural end. The full evidence of every run goes to `lab/runs/` and then to `lab-runs/` in the data repo; the recipe keeps a hash of it.

A proof may carry:
- `proxy: <card>`: run on a sibling card, because nobody rents this one.
- `legacy: true`: the recipe passed the older acceptance (load and chat only). `lab.py convert` re-runs it through all six gates.
- `reported: true`: someone else published the launch and its numbers (`src` names their repo at a commit; `claims` what they say it does). `lab/import_reported.py` brings these in from `data/reported/<source>/`. Our gates have not run; a lab run replaces it.

A recipe for several computers ends in `.<n>x` and its profile says `machines: n`. A profile may also name `flags` (host networking or IPC), a `build` (an image built from a source's Dockerfile at a commit), `setup` (what the source needs outside the container), `needs` (what the host needs besides the card, for launches that offload weights to it: `host_ram_gb` of free system RAM, `disk_gb` for the weights, `fast_storage: "nvme"` when they must be on a local NVMe SSD; validated by `lab.py check`, exported to the plugin) and `source`. A profile may also name a `companion`. That is a second container on a card of another kind in the same machine (`hw`, `cards`, a digest-pinned `image`, optional `entrypoint`, `environment`, `arguments`, typed `resources` and `shm`). It starts before the engine, gets the engine's read-only weights and prepared-pack mounts, and shares a tmpfs dir with the engine at `/run/local-ai/shared`; the recipe never names host paths. `lab.py check` validates it. No plugin runs companions yet, so `export_plugin.py` keeps such launches out of the plugin catalog. They are candidates, for example GLM-5.3-Flash on an RTX 3090 with a B70 expert server.

A new proof from another host on the same recipe confirms it.

## What gets picked

For each card, `catalog.py` keeps at most three recipes, one per model, ranked by: lab proof over legacy over reported, then family order (Qwen, Gemma, DeepSeek, GLM, Step, Kimi, MiniMax), newest model, and decode speed. A model older than 240 days is dropped. The first pick is the recommended one. Every other recipe for the card is listed after the picks (`more`), best first. This happens once per setup (`setups`: cards per machine × machines, smallest first); the card's own `picks` and `more` are its one-card setup, so a one-card client never gets a recipe that needs several. See [data.md](data.md) for where recipes come from.

## Running a recipe

`lab.py render <recipe>` gives the full launch. Every program:
1. downloads the weights at the pinned commit;
2. writes the config file;
3. starts the image on a bridge network, with no host IPC and no added capabilities, passing the card through;
4. puts the Local AI gateway in front.

A host profile runs on the machine itself instead: install its pinned pip packages, download the weights to the
profile's paths, write the launcher, raise `iogpu.wired_limit_mb` to the card's memory less the profile's reserve
(it resets on reboot), and start the launcher. `lab.py try --on endpoint` tests it where it runs.

## Rules CI enforces

- Every recipe is at its path, with weights pinned to a commit and a profile pinned to an image digest (or, for a host profile, the digest of its launch).
- Every recipe's latest proof passed all six gates, or is marked `legacy`, or is `reported` with its source.
- `dist/catalog.json` and `plugin/v2/recipes.json` are current.
