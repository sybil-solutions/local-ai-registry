# RTX 3090 host launch: Inkling-2-Mini UD-IQ4_XS, vision, 336K context

Inkling-2-Mini GGUF UD-IQ4_XS (unsloth/Inkling-2-Mini-GGUF @ae9327c5) with the F16 mmproj (image input) on one
RTX 3090 24GB. The context is 344,064 tokens with a q8_0 KV cache and one slot. The engine is unsloth's llama.cpp
release `b12588-mix-c0c3de8` (source tarball, sha256 `92ee8439…573607`) plus `inkling2-hc19.patch` in this
directory, built for sm_86. Launch: `registry/launches/llama.cpp-inkling-2-mini-ud-iq4-xs-q8kv-336k-rtx-3090-24gb.json`.

## Why a patched build

Stock llama.cpp cannot hold long contexts with this model on a 24 GB card. Inkling's global layers use a relative-position
attention bias. Upstream (unsloth's port) runs it as banded flash attention, but only for f16/bf16/f32 K/V. A quantized
KV cache falls back to a dense bias path whose compute buffer grows with n_kv × tokens × heads. With f16 KV the card tops
out near 96K tokens with full-resolution images.

The patch (`git diff` over the release source, 9 files, CUDA only):

1. Banded flash attention reads q8_0 K/V tiles and dequantizes them in the kernel (Turing+ MMA path). All banded layers
   share the whole input mask, so the scheduler makes one input copy per graph instead of a strided copy per layer.
2. The CUDA buffer type no longer reserves the f16 K/V conversion scratch behind `FLASH_ATTN_EXT_BANDED` outputs. That
   space is n_kv × 4 KiB per layer, and the banded kernel never uses it. The q8 compute buffer at 128K drops from
   815 to 321 MiB.
3. Fused CUDA kernels for Inkling's per-layer glue: clip, loop controller, conv taps, depth-mix, SiTU, relmask. Each
   fusion checks for allocator overlap and runs into pool scratch when the graph allocator reused a live input's memory.
4. Evicts cached CUDA graphs and retries when `cudaGraphInstantiate` runs out of memory near the cap.

Quality: KLD over 24×1024 tokens against the unfused build's logits is 0.0038 for f16 KV, against a noise floor of
0.0037 from changing the ubatch size alone. For q8_0 KV it is 0.0041, with top-1 agreement at 97.98%. UD-IQ4_XS against
UD-Q4_K_XL is 0.0043, with top-1 at 97.86%; this was not measured against BF16.

## Measured (owner's machine, Ryzen 5 5600, PCIe 3.0 x16, ~1.25 GB of VRAM held by a desktop)

| | |
|---|---|
| lab gates | load chat reasoning tools context speed (`lab.py try --on endpoint`) |
| decode, lab speed gate | 71.7 tok/s (registry harness, temperature 0.8, thinking counted) |
| decode, own harness | 81.7 tok/s on a short prompt (temp 1, top-p 1), 67.9 tok/s at a 32K prompt |
| prefill | lab context gate 1052 tok/s (242,750-token prompt); 1920 tok/s at 4K, ~1800 at 32K; ~900 tok/s averaged over a 330K prompt |
| long-context retrieval | 3/3 needles in a 330K-token prompt; peak device memory 24.11 GB incl. desktop |
| vision | 6/6 image Q&A at full resolution, including a 4103-token image, at this exact context |
| host RAM | MemAvailable fell by 5.8 GiB during the gates (needs.host_ram_gb 7 with headroom) |

Limits:
- **One slot only.** With `--parallel > 1` and the q8_0 cache, several long prompts prefilled at once use the dense path.
  That path converts the whole cache to f16 per layer and runs out of memory at runtime (4 × 32K prompts in a 192K pool).
  For concurrency use f16 KV: `--parallel 4 --kv-unified -c 147456` measured 4 × 39 tok/s and 4 × 32K sessions.
- 352K does not start on this card with a desktop attached.

## Build

```sh
docs/rtx3090-inkling-2-mini-llamacpp-hc19/build.sh llama.cpp-inkling2-hc19
```

The script downloads the pinned tarball, checks its sha256, applies the patch, and builds `llama-server` with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. Measured with CUDA 13.4 and driver 615.78. The patched tree was checked to
be byte-identical to the measured source.
