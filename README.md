# Local AI registry

**https://local.sybilsolutions.ai**

The model to run on your GPU, and exactly how to run it. Every recipe here is the output of a run that passed six checks on that card: it loads, chats, reasons, calls tools, holds its context window, and decodes at 15 tok/s or more.

```
registry/                      the source of truth, tiny
  cards/<vendor>/<card>.json          a GPU: name, memory, how programs detect it
  engines/<profile>.json              an engine template: pinned image, arguments, config, defaults
  launches/<name>.json                a launch frozen exactly as it was validated (one recipe each, until a lab run replaces it)
  recipes/<vendor>/<card>/<model>.<engine>.<context>k[.<n>x].json   (n = cards in the setup)
                                      one recipe (~400 bytes): weights, engine, settings, proof
  models.json                         each model: family, release date, what it is for
lab/            run, check and publish recipes (lab.py try | convert | render | check); digest.py: the Monday report
dist/           catalog.json: everything above, rendered, at most 3 picks per GPU setup
site/           local.sybilsolutions.ai and its API (static, on Cloudflare Pages)
sdk/            js/ and python/: pick a recipe for a GPU, print the command
app/            local-ai: run a recipe on any Linux, from a terminal or a status bar
images/         the container images we build (gateway, tabbyapi-exl3, ...)
plugin/         what the Omarchy Local AI plugin reads (paths are fixed)
data/           everything the registry held before, as it was; nothing reads it
docs/           design.md
```

## A recipe

`registry/recipes/nvidia/rtx-3070-ti-8gb/qwen3.5-9b.tabbyapi.64k.json`:

```json
{"model":"qwen3.5-9b","weights":"TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938",
 "engine":"tabbyapi-exl3@0f83e6198dc3","set":{"ctx":65536,"draft":"mtp"},"card":"rtx-3070-ti-8gb",
 "proof":[{"at":"2026-09-25","on":"vast","gpu":"RTX 3070 Ti","gates":"load chat reasoning tools context speed",
           "tps":105.3,"prefill":1485,"served":"Qwen3.5-9B-exl3-22ef1303","log":"sha256:6fc6fe2e6f2cead9"}]}
```

- `weights`: a Hugging Face repo at a full commit.
- `engine`: a profile in `registry/engines/`, pinned to its image digest.

A launch with a reproducible failure can be withdrawn by prepending a proof with
`withdrawn: true`, its source URL and reason. Catalogs and plugin exports omit it;
historical proofs and recipe IDs stay in source. Requalify the exact launch with
a fresh passing lab proof before offering it again.
- `set`: only the settings that differ from the profile's defaults.
- `proof`: the latest passing runs. A proof marked `proxy` was run on the sibling card it names; one marked `legacy` passed the older acceptance (load and chat) and is waiting for a full run.

`python3 lab/lab.py render <recipe>` prints the full launch: image, arguments, config file, weights location, port.

A launch that keeps weights in system RAM or reads them from disk while serving also states what the host needs besides the card, and the plugin offers it only where the machine has it:

```json
"needs": {"host_ram_gb": 75, "disk_gb": 85.14, "fast_storage": "nvme"}
```

`host_ram_gb` is the free system RAM the server takes: how far MemAvailable falls from before the start to its lowest point while the lab gates run (not the weight file sizes; the plugin compares it with MemAvailable), `disk_gb` the space for the weights, and `fast_storage: "nvme"` means the weights directory must be on a local NVMe SSD. `lab.py check` validates it; `plugin/v2/recipes.json` carries it.

## Prepared weights and container resources

The v3 plugin contract carries `prepare: {at, args, verifyArgs, gpu, sizeGb}` and
`resources: {memoryBytes, memorySwapBytes, memlockUnlimited, ipcLock}`. Preparation
uses the serving image, environment and resource limits. Raw inputs remain read-only;
verification mounts every input and generated output read-only and receives no GPU.
`memorySwapBytes` is Docker's total memory plus swap limit; equal memory and swap
values disable swap. Arbitrary Docker flags remain excluded from plugin exports.

The plugin stages generated packs privately, verifies them with the pinned image,
and publishes them atomically under a hash of the complete preparation contract.
It verifies every reuse. Stop retains completed packs; Forget removes only managed,
unused downloads. Disk eligibility counts missing generated output separately from
verified raw inputs. The by-hand SDK steps carry these same resource and verification
arguments; they do not implement the plugin's managed lifecycle.

Typed launches require a passing proof bound to the full rendered `launch_sha256`.
Changing the image, inputs, environment, resources or preparation requires new
acceptance evidence. v2 excludes any recipe requiring preparation or typed resources
rather than silently dropping those requirements. This contract adds no new qualified
hardware recipes by itself; fidelity, exact-image lab gates and promotion remain separate.

## Running a recipe

Every program runs one the same way:

1. Download the weights at the pinned commit.
2. Write the config file.
3. Start the image on a bridge network with the card passed through (`--gpus` on NVIDIA, the render node on AMD and Intel).
4. Put the [gateway](https://github.com/sybil-solutions/local-ai-images) in front, which serves the OpenAI, Anthropic and Responses APIs.

## Browsing models and hardware

[Models](https://local.sybilsolutions.ai/models/) and
[hardware](https://local.sybilsolutions.ai/hardware/) are full-text searchable;
column headings sort the directories, while the Sort control orders configuration
tables by hardware, model, memory, recipe, context, decode, prefill or evidence.
Each model has a `/models/<id>/` page with its rendered Hugging Face model card
and every recorded hardware configuration. Hardware pages at `/gpu/<id>/`
show original-source photos, specifications and the same table by model.
Photo captions link the original manufacturer page and explicitly label
representative family or chassis images.

Configuration rows distinguish card count, machine count and documented
host RAM, disk/NVMe and logical-CPU requirements. Selecting an exact recipe
changes its decode, prefill, context, evidence and copyable command together.
Unknown measurements stay unknown and sort after recorded measurements.
The compact table shows Hardware, Recipe, Speed and Run (plus Model on hardware
pages). Only setups with multiple recipes have dropdowns. Command expands
the pinned launch, complete setup requirements and evidence.
The command is one Bash invocation containing the existing SDK's pinned steps;
engine prerequisites and source-specific setup still apply. Multi-machine
recipes run on each machine with its documented rank and network settings.

Model cards are public Hugging Face README snapshots pinned to their source
commit, cached in `site/lib/model-cards.data.json` and sanitized during the
static build. Missing catalog model links are resolved from verified upstream
metadata or matching public repositories, with provenance shown on the page.
`npm run build` fetches newly used cards; to refresh existing snapshots, run
`node scripts/model-cards.mjs --refresh` from `site/`, then build.
Hardware image provenance lives in `site/lib/hardware-photos.json`; add the
original-source image and its attribution there when adding hardware.

## Adding or updating a recipe

Recipes are never written by hand. Run one:

```bash
python3 lab/lab.py try <repo>@<commit> --model <id> --engine tabbyapi-exl3 --card <card> --set ctx=65536 --set draft=mtp
```

`try` rents the card on Vast (or RunPod, or `--on endpoint --endpoint <url>` for your own machine), runs the six checks, writes the recipe only if all pass, and destroys the machine. `lab/campaign.py <plan>` runs many at once. `lab.py convert <card>` re-runs a `legacy` recipe the same way. Then `make` and open a pull request; CI runs `make check`.

### Bounties

Cards and models without a tested recipe carry a **$50 bounty**: [open bounties](https://github.com/sybil-solutions/local-ai-registry/issues?q=is%3Aissue+is%3Aopen+label%3Abounty).
Claim one with `/attempt`, run it with the [recipe kit](https://github.com/sybil-solutions/local-ai-recipe-kit), and open one PR with the recipe `lab.py try --on endpoint` wrote.
The rules are in [BOUNTIES.md](https://github.com/sybil-solutions/.github/blob/main/BOUNTIES.md); every PR follows [CONTRIBUTING](https://github.com/sybil-solutions/.github/blob/main/CONTRIBUTING.md).

## Using it

- **Omarchy Local AI 6.10+** reads `plugin/v3/recipes.json`; older versions retain the independent v2 export and cache.
- **Anything else** reads `dist/catalog.json` (or https://local.sybilsolutions.ai/api/v2/catalog.json): `cards.<card>.picks` lists recipe keys for one card, first one recommended; `cards.<card>.setups` does the same for each setup (one card, several cards in one machine, several machines); and `recipes.<key>.launch` is the rendered launch. The SDKs in `sdk/` do the matching for you.
