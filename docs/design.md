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
| reasoning | The thinking comes back separately and 17 × 23 is answered 391 |
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

### A pinned host runtime

A host profile whose runtime is not on the machine can pin it with an optional `runtime` (never together with `pip`).
The SDKs render it as steps in place of the free-text `install`. Each setup step is one `( set -e ... )` subshell, so it
stops at its first failure and exits non-zero, and the one-script `command()` of such a launch starts with `set -e`, so a
failed step stops every later step and the server, however the script is run. Nothing is written through a symlink: each
step first refuses a symlink at its destination or at any directory above it, and checks again right before it renames
into place. A file (`refs/<ref>`, the launcher) is written to an exclusive `mktemp` file in its checked directory and
renamed over the destination only if that is neither a link nor a directory.

Every download in these steps is `curl -q ...` (`-q` first, so no `.curlrc` is read).

- `archive: {url, sha256, into}`: an `https` `.tar.gz` or `.tgz` (the file name is the URL's last segment, percent-decoded,
  so `%2B` is `+`). It is downloaded into a fresh directory inside `into`, checked with `shasum -a 256 -c` and extracted
  there. Before anything live is touched, no member name may be absolute or contain `..`; every link in the extracted tree
  must be relative, not dangling, and resolve inside it (safe internal links such as `python3 -> python3.12` are kept);
  every hard link must stay inside it; it may hold no device, fifo, socket or set-id file; every top-level entry must have
  a plain name, must not be a link in `into`, and must not hold another step's destination, compared ignoring case (an
  entry `SITE` holds `rt/site`, and so does `site` when the target is `RT/site`); and the staged interpreter
  (so resolved inside the archive) must exist and run. Only then does each entry replace the
  one in `into` by rename. If the publication does not finish (a rename fails, or the step is stopped by TERM, INT or
  HUP), every entry already moved is put back before anything is cleaned up (a further TERM, INT or HUP during the put-back is
  ignored, so it always finishes), so the previous runtime stays whole and
  usable; if putting it back fails too, the previous copy is kept (not deleted), the step says where and fails. The same
  holds for the packages target and wheel dir and for a hub model directory. A KILL cannot be caught: it leaves the
  staging and backup directories in place, never deletes them. Other files in
  `into` (packages, weights, the launcher) are kept.
- `python`: the interpreter, a relative path inside `into`, run as `./<python>`, never looked up on `PATH`.
- `packages: {dir, target, wheels: [{name, version, url, sha256}]}` (needs `python`; `dir` and `target` do not overlap):
  every wheel is downloaded into a fresh directory beside `dir` and checked against its hash, then installed offline into a
  fresh directory beside `target` with
  `./<python> -I -m pip --isolated install --no-index --find-links <wheels> --only-binary :all: --require-hashes -r requirements.lock.txt --target <fresh> --no-cache-dir`,
  with `PIP_CONFIG_FILE=/dev/null` (which turns off every pip config file; `--isolated` alone does not) and `TMPDIR`, `TMP`,
  `TEMP`, `HOME`, `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` inside the staging directory, which is removed afterwards.
  The lock lists exactly these wheels, so a missing or extra dependency fails the install. Only after the install succeeds
  do the fresh directories replace `target` and `dir` by rename, so `target` holds exactly the pinned wheels (nothing left
  from an earlier install); a failure, including a failed rename, leaves the previous `target` and `dir` as they were.

A host weights entry with `layout: "hub"` is a Hugging Face cache at `at`; one launch's host weights are all hub or all
directories. Entries may share a cache only when their `at` is the same path; distinct caches never nest. A repo appears
at most once per cache (a second entry, for another revision or other files, is rejected), because each publish replaces
the repo's whole model directory. Paths are compared ignoring case, as macOS file systems do.

A host launch with a runtime or with hub weights is confined, so its launcher (`config.at`) must be a plain relative path
that overlaps no other destination, and is written by the same guarded writer (an exclusive temp file renamed into
place, never through a link). A host launch with directory weights and no runtime renders its launcher as before. With a pinned runtime, `runtime.packages` must include `huggingface_hub`, and the download runs on the pinned
interpreter with `target` on its path (`snapshot_download(..., token=False)`), so the machine needs no `hf`. Without one
it is `hf download`. Either way the download goes into a fresh staging cache beside `at`, with every `HF_*` and
`HUGGING_FACE_*` variable and `NETRC` removed and `HOME`, `HF_HOME`, `HF_XET_CACHE` and `TMPDIR` inside that staging cache, so no
ambient token, config, cache or `.netrc` is used, and nothing already in `at` (a planted `blobs` or `refs` link) is written
through. In the staged cache `models--<org>--<name>/snapshots/<revision>` must exist and every link must stay inside the
model directory (the same check as the archive); with `ref` (for example `main`) a `refs/<ref>` file is written to name
the revision, so a server that loads the repo id offline finds it. Only then does the staged model directory replace the
one in `at` by rename, put back if that fails. A rerun downloads the revision again.

`lab.py check` rejects hashes that are not 64 hex characters, URLs that are not plain `https` paths, absolute paths or `..`,
an interpreter outside the archive, wheel files whose name or version differ from the entry, duplicate wheels, unknown
keys, `ref` without `layout: "hub"`, runtime weights without hub layout or a pinned `huggingface_hub`, and destinations
that collide: `packages.target`, `packages.dir`, each hub cache (weights may share one) and the launcher never equal or
contain one another, none is or contains `into`, and none lies in the archive entry that holds the interpreter. Both SDKs
refuse to render any of them. These keys are in a host profile's digest only when present, so profiles without them keep their
digest, and adding them to a measured profile means rerunning its recipe.

## Rules CI enforces

- Every recipe is at its path, with weights pinned to a commit and a profile pinned to an image digest (or, for a host profile, the digest of its launch).
- Every recipe's latest proof passed all six gates, or is marked `legacy`, or is `reported` with its source.
- `dist/catalog.json` and `plugin/v2/recipes.json` are current.
