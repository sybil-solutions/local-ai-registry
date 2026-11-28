"""Print a GGUF file's metadata key/values and a tensor size summary (reads the header only)."""
import struct
import sys
from collections import defaultdict

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# bytes per block and elements per block for the ggml types that appear in these quants
TYPE_SIZE = {0: (4, 1), 1: (2, 1), 8: (34, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256),
             14: (210, 256), 16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
             21: (110, 256), 22: (82, 256), 23: (136, 256), 29: (56, 256), 30: (2, 1), 39: (17, 32), 40: (36, 64)}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_str(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def read_value(f, t):
    if t in SCALARS:
        return read(f, SCALARS[t])
    if t == 8:
        return read_str(f)
    if t == 9:
        et, n = read(f, "I"), read(f, "Q")
        vals = [read_value(f, et) for _ in range(n)]
        return vals if n <= 64 else f"<array of {n}>"
    raise ValueError(f"type {t}")


with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = read(f, "I"), read(f, "Q"), read(f, "Q")
    for _ in range(n_kv):
        key = read_str(f)
        val = read_value(f, read(f, "I"))
        if "tokenizer.ggml" not in key and key != "tokenizer.chat_template":
            print(f"{key} = {val}")
    groups = defaultdict(int)
    unknown = set()
    for _ in range(n_tensors):
        name = read_str(f)
        dims = [read(f, "Q") for _ in range(read(f, "I"))]
        ttype, _offset = read(f, "I"), read(f, "Q")
        n = 1
        for d in dims:
            n *= d
        if ttype not in TYPE_SIZE:
            unknown.add(ttype)
            continue
        bs, be = TYPE_SIZE[ttype]
        kind = "experts" if "_exps" in name else ("output" if name.startswith("output.") else
                                                  ("token_embd" if name.startswith("token_embd") else "dense"))
        groups[kind] += n // be * bs
    print("tensor bytes by group (GB):", {k: round(v / 1e9, 3) for k, v in groups.items()},
          "unknown types:", sorted(unknown))
