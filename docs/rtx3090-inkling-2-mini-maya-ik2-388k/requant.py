# requant.py IN OUT REGEX TYPE: copy a GGUF, re-quantizing the tensors whose name matches REGEX to TYPE through ggml
# (NVFP4: with a per-expert .scale = amax/(448*6), as maya-ik2 requires), every other tensor byte for byte.
# env: GGUF_PY (llama.cpp gguf-py dir), LIBGGML_BASE (a libggml-base.so with NVFP4).
# Inkling-2-Mini: requant.py Inkling-2-Mini-UD-Q4_K_XL.gguf Inkling-2-Mini-UD-Q4_K_XL-nvfp4tail.gguf 'blk\.2[0-3]\.ffn_(gate_up|down)_exps\.weight' NVFP4
import sys, os, re, ctypes, numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", "third_party/llama.cpp-b12590/gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q, quants
src, dst, rx, tname = sys.argv[1:5]; T = Q[tname]
lib = ctypes.CDLL(os.environ["LIBGGML_BASE"])  # a libggml-base.so with NVFP4 (llama.cpp b12588+)
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
r = GGUFReader(src)
arch = r.fields["general.architecture"]; arch = bytes(arch.parts[arch.data[0]]).decode()
w = GGUFWriter(dst, arch, use_temp_file=False)
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture": continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        vals = [bytes(f.parts[i]).decode() if sub == gguf.GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data]
        w.add_key_value(f.name, vals, vt, sub_type=sub)
    else:
        v = bytes(f.parts[f.data[0]]).decode() if vt == gguf.GGUFValueType.STRING else f.parts[f.data[0]].tolist()[0]
        w.add_key_value(f.name, v, vt)
bs, ts = gguf.GGML_QUANT_SIZES[T]
plan = []
def bshape(shape, ty):
    b, s_ = gguf.GGML_QUANT_SIZES[ty]; rev = list(reversed(shape)); rev[-1] = shape[0] // b * s_; return rev
for t in r.tensors:
    shape = [int(x) for x in t.shape]            # ne0 first
    if re.search(rx, t.name):
        n_per_row = shape[0]; nrows = int(np.prod(shape[1:]))
        nbytes = nrows * n_per_row // bs * ts
        plan.append((t, True, nrows, n_per_row))
        w.add_tensor_info(t.name, bshape(shape, T), np.dtype(np.uint8), nbytes, raw_dtype=T)
        if T == Q.NVFP4:
            ne = shape[-1]; plan.append((t, "scale", ne, 0))
            w.add_tensor_info(t.name.replace(".weight", ".scale"), [ne], np.dtype(np.float32), ne * 4)
    else:
        plan.append((t, False, 0, 0))
        w.add_tensor_info(t.name, bshape(shape, t.tensor_type), np.dtype(np.uint8), int(t.n_bytes), raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
scales = {}
for t, rq, nrows, npr in plan:
    if rq == "scale":
        w.write_tensor_data(scales.pop(t.name)); continue
    if not rq:
        w.write_tensor_data(np.asarray(t.data).view(np.uint8).reshape(-1)); continue
    f32 = np.ascontiguousarray(quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(nrows, npr).astype(np.float32))
    ref32 = f32
    if T == Q.NVFP4:
        ne = int(t.shape[-1]); per = f32.reshape(ne, -1)
        sc = (np.abs(per).max(1) / (448.0 * 6.0)).astype(np.float32); sc[sc == 0] = 1.0
        scales[t.name] = sc; f32 = np.ascontiguousarray((per / sc[:, None]).reshape(nrows, npr).astype(np.float32))
    out = np.empty(nrows * npr // bs * ts, dtype=np.uint8)
    step = 4096
    for s in range(0, nrows, step):
        n = min(step, nrows - s)
        got = lib.ggml_quantize_chunk(int(T), f32[s:s+n].ctypes.data, out[s*npr//bs*ts:].ctypes.data, 0, n, npr, None)
        assert got == n * npr // bs * ts, (got, n)
    deq = quants.dequantize(out.reshape(nrows, -1), T).reshape(nrows, npr)
    if T == Q.NVFP4: deq = (deq.reshape(ne, -1) * scales[t.name][:, None]).reshape(nrows, npr)
    err = np.abs(deq - ref32).mean() / (np.abs(ref32).mean() + 1e-12)
    print(f"{t.name}: {t.tensor_type.name} -> {T.name}, rel L1 err {err:.4f}", flush=True)
    w.write_tensor_data(out)
w.close(); print("done", dst)
