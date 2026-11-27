# Inkling-2-Mini on the Intel Arc Pro B70: owner run

This folder holds the evidence for two recipes in `registry/recipes/intel/intel-arc-pro-b70-32gb/`:

- `inkling-2-mini.llama.cpp.128k.json`, a 131,072-token window;
- `inkling-2-mini.llama.cpp.256k.json`, a 262,144-token window.

The two launches differ only in `--ctx-size`. The registry cannot rent this card, so the owner ran the lab on their
own machine.

## Machine

| Part | Value |
|---|---|
| Card | Intel Arc Pro B70 32 GB, PCI id `8086:e223` (`lspci` shows `Battlemage G31 [Arc Pro B70]`) |
| Host | Ryzen 9 5950X, 32 GB RAM, CachyOS, Linux 7.3.0 with the xe driver |
| Container runtime | podman 6.2.0 |
| GPU runtime in the image | compute-runtime 26.40.40196.8, IGC 2.42.6, oneAPI DPC++ 2026.0.0, oneDNN 3.11.2 |

Nothing else used the card during the runs.

## Engine and weights

The engine is llama-server from [Thotheris/llama.cpp](https://github.com/Thotheris/llama.cpp/tree/inkling2-b70),
branch `inkling2-b70`, at commit `20fa6fdf5f98dfe830db17acb2296699fd7bfb63`. The branch is upstream `074f1f3` plus
SYCL and Vulkan kernels for Inkling 2. The branch's
[SYCL-Inkling2-B70.md](https://github.com/Thotheris/llama.cpp/blob/20fa6fdf5f98dfe830db17acb2296699fd7bfb63/docs/backend/SYCL-Inkling2-B70.md)
lists the changes, the settings, and the llama-bench and KL-divergence numbers.

The image was built from that commit with this command:

```bash
podman build -f .devops/intel-b70.Dockerfile --target server -t llama-inkling-b70:20fa6fd .
```

The result was image `bba88be999fa` (9.69 GB). `evidence/image-build.txt` has the end of the build log. The build
compiled the commit again and produced the same image as the build of the parent commit `02b5228`, because
`20fa6fd` changes only documentation.

The weights are `Inkling-2-Mini-Q4_K_M.gguf` from `lmstudio-community/Inkling-2-Mini-GGUF` at revision
`23c938b028945d4b0fa9f71fb603b5cd1a683e71` (17,478,094,048 bytes). The local file's sha256 is
`dac3702abcb5684cbcb1fbd148d94b2d531283be974630ac61964d194db483de`, the same as the LFS oid on Hugging Face.

## Lab runs

`tools/run-lab.sh` starts the container from a launch JSON and runs `lab.py try --on endpoint` against it. The
container gets `/dev/dri`, the launch's environment, arguments and shared memory size, and the GGUF mounted read-only
at `/models`. `evidence/<ctx>/rendered-launch.txt` has the exact command, with the model directory replaced by
`<gguf-dir>`.

The two runs ran one after the other on 2026-11-27 and finished at 06:05 and 06:10 UTC. All six gates passed in
each. The run records are in `data/lab-runs/`. The first 16 hex digits of each record's sha256 match the `log` field
of its recipe.

| Gate | 128K | 256K |
|---|---|---|
| load | healthy after 58 s | healthy after 97 s |
| chat | `Red, yellow, blue` | `Red, yellow, blue` |
| reasoning | 123 thinking characters, answer `391` | 102 thinking characters, answer `391` |
| tools | called `get_weather` with `{"city":"Paris"}`, and the reply quoted code `0c291cf9` | called `get_weather` with `{"city":"Paris"}`, and the reply quoted code `a5eb8291` |
| context | found `58213` in a 91,890-token prompt, 27.2 s for the whole request | found `58213` in a 184,730-token prompt, 69.6 s for the whole request |
| speed | 1,578 tokens in 19.2 s, so 82.3 tok/s | 1,265 tokens in 14.8 s, so 85.7 tok/s |
| prefill in the recipe | 3,378 tok/s | 2,654 tok/s |
| peak device memory | 25,246 MiB | 30,316 MiB |
| run record | `…128k.20261127T060516.json`, `log` `6528deb19cf984ae` | `…256k.20261127T061035.json`, `log` `f41d3e6910af4054` |

- The speed windows are shorter than 30 s because the answers ended sooner.
- At 256K, the server leaves about 2.3 GB of the card free. Run it with nothing else on the card, and no display.
- The load time includes compiling the GPU kernels, because no kernel cache was mounted. With a volume at
  `/root/.cache`, a restart takes about 7 s.
- The recipe's prefill is the context gate's figure. It divides the prompt by the time of the whole request, so it
  counts the answer too. It is not the engine's prefill rate.
- `tools/run-lab.sh` reads the peak device memory once a second from the `drm-total-vram0` and `drm-resident-vram0`
  lines of the server's `/proc/<pid>/fdinfo` files, and keeps the largest value. The card has 32,656 MiB.

## Settings that were not submitted

- **A larger window.** At 393,216 tokens the compute buffer does not fit, and at 524,288 the 16 GiB KV cache does not
  fit. 262,144 is the largest power-of-two window that loads with f16 KV.
- **A q8_0 KV cache.** Inkling 2's attention reads a relative-position bias, and on this branch only the f16 SYCL
  flash-attention kernels support it. With `-ctk q8_0 -ctv q8_0`, attention runs on the CPU. `llama-bench` on the
  host build measured 72 tok/s prefill at `-p 2048` against 6,487 with f16, and 51 tok/s decode against 88.

## Files

| File | Contents |
|---|---|
| `evidence/128k/` and `evidence/256k/` | for each run, `lab-try.log` (the output of `lab.py try`), `rendered-launch.txt` (the `podman run` command) and `server.log` (the llama-server log) |
| `evidence/image-build.txt` | the end of the image build log, and the image id |
| `tools/run-lab.sh` | the script that starts the container, runs the lab, records the load time and the peak memory, and stops the container |
