# Local AI for Linux

Run a model validated for your GPU, or one across several cards of a kind, and open a coding agent on it — on any Linux desktop. This is the distro-neutral core of [Local AI for Omarchy](https://github.com/0xSero/omarchy-local-ai): the same backend, recipes and view model, without Omarchy's shell or helper commands.

- **Validated models.** `core/recipes.json` holds every recipe in [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) for 36 cards (NVIDIA RTX 30/40/50, RTX Ada and Pro, Intel Arc Pro B70, AMD ROCm), the recommended one first. Each was accepted on its exact card, or cards: download, load, a correctness check and speed at several context lengths.
- **Checked weights.** Downloaded as you from a pinned revision; every file's size and sha256 is checked against the Hub.
- **Containers.** The engine runs on a private network with `no-new-privileges`; a keyed gateway on `127.0.0.1` speaks the OpenAI, Anthropic and Responses APIs and logs one line per answer.
- **Agents.** pi, Claude Code, Codex, OpenCode, omp, Crush, Grok, Copilot and Hermes open in your terminal, in the folder you pick, pointed at the gateway. Nothing in their own config is touched.

## Install

```bash
git clone https://github.com/sybil-solutions/local-ai-registry && cd local-ai-registry/app
sudo make install          # /usr/local/bin/local-ai, /usr/local/share/local-ai, the polkit action
local-ai doctor            # what this machine has and lacks, with the command that adds each piece
```

It needs `jq`, `curl`, `iproute2` (`ss`), `pciutils` (`lspci`) and a container runtime. `doctor` prints the install command for your distro (pacman, apt, dnf or zypper).

### Container runtime

`local-ai` picks, per model, the first of:

1. **Docker you can reach** (you are in the `docker` group, or Docker is rootless): no password.
2. **Rootless Podman**: no password, no daemon. NVIDIA cards reach it through the container toolkit's CDI spec.
3. **Docker through pkexec**: each start and stop asks for your password once, with a plain message from `org.local-ai.policy`. The command refuses to run as root unless it and every directory above it are root-owned, which `make install` gives it.

### NVIDIA cards

Install `nvidia-container-toolkit` (Arch: `pacman -S nvidia-container-toolkit`; Debian, Ubuntu, Fedora and openSUSE: from NVIDIA's repository, see the [toolkit's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)), then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker   # Docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml                          # Podman
```

Intel Arc and AMD cards need no toolkit: the engine gets the card's render node (and `/dev/kfd` on AMD). AMD cards are found through ROCm's `amd-smi`.

## Use

```bash
local-ai snapshot                          # what a panel draws, as JSON
local-ai run <recipe> <gpu>[,<gpu>]        # e.g. local-ai run qwen38-27b-exl3-3bpw-rtx3090-sglang-tp1 nvidia:0
local-ai stop <recipe>
local-ai open <recipe>                     # the chosen agent on it, in your terminal
local-ai set agent|folder <value> [recipe]
local-ai share <recipe> [off]              # tailscale serve, tailnet only, still keyed
local-ai log
local-ai doctor
local-ai waybar                            # one line of JSON for a bar module
local-ai remove                            # models, containers, engine images, weights and settings
```

`local-ai snapshot | jq '.kinds[] | {hw, models: [.models[].id]}'` lists the models each of your cards can run. Each running model answers on `http://127.0.0.1:<port>/v1` (ports from 12434); the key is in `~/.local/state/local-ai/gateway.key`. Agents are opened with `xdg-terminal-exec`, else `$TERMINAL -e`, else `x-terminal-emulator -e`.

### Waybar (Sway, Hyprland, niri, river)

```jsonc
"custom/local-ai": {
  "exec": "local-ai waybar",
  "return-type": "json",
  "interval": 5
}
```

It shows `AI`, `AI 1` when a model is ready, `AI …` while one starts and `AI !` when one failed, with the class `idle`, `ready`, `busy` or `failed` for your CSS and the running models in its tooltip.

## Layout

| Path | What |
|---|---|
| `core/local-ai` | The backend: GPUs, downloads, containers, agents, usage, the snapshot |
| `core/Model.js` | The view model every panel draws: plain functions, snapshot and UI state in, rows and actions out |
| `core/recipes.json` | The vendored recipes, one card kind per line (`make sync` copies them from `../plugin/v2/recipes.json`) |
| `core/org.local-ai.policy.in` | The polkit actions for the password prompts (installed with the command's path) |
| `test/` | `core-test.sh` runs the backend end to end with shims for Docker, Podman, the GPU tools and pkexec; `model-test.sh` checks the view model; `make test` runs both |

State lives in `~/.local/state/local-ai`, weights in `~/.cache/local-ai/models`; containers are named `local-ai-<recipe>-engine` and `-gateway` and labelled `io.local-ai`, so this and the Omarchy build can live on one machine.

## Status

- **Tested:** the backend end to end with shims (Docker and rootless Podman paths, groups, a second copy of a model, downloads and their hash check, agents, stop, the Waybar line, `doctor`), the view model (72 assertions), and `doctor`, `snapshot` and `waybar` on a machine with 2× RTX 3090 and 2× Arc Pro B70 on Arch.
- **Not yet run on real hardware:** a model started by this build, and the Podman path (the test machine has no Podman).
- **Panels to come:** a Quickshell panel for Wayland compositors (a port of the Omarchy panel), a KDE Plasma widget and a GNOME Shell extension, all drawing `core/Model.js`.
- Two builds on one machine do not see each other's models: an Intel or AMD card one of them runs looks free to the other (NVIDIA cards show as in use by another program).

## License

MIT
