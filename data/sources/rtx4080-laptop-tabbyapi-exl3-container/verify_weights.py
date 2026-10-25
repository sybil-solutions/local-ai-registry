#!/usr/bin/env python3
"""Check every pinned weight file: size, and sha256 (LFS files) or git blob sha1 (small files). Exit 1 on any mismatch.

The directory must hold exactly the manifest's files. The one other entry allowed is a real `.cache`
directory (not a symlink or a file), which `huggingface-cli download --local-dir` leaves behind and the
engine never reads; any other extra entry fails."""
import hashlib, os, sys
exp, d = sys.argv[1], sys.argv[2]
bad = 0
for line in open(exp):
    kind, want, size, name = line.split(maxsplit=3)
    name = name.strip(); p = os.path.join(d, name)
    if not os.path.exists(p) or os.path.getsize(p) != int(size):
        print("MISSING/SIZE", name); bad += 1; continue
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % int(size))
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    ok = h.hexdigest() == want
    bad += not ok
    print("OK " if ok else "BAD", kind, h.hexdigest(), size, name)
extra = set(os.listdir(d)) - {l.split(maxsplit=3)[3].strip() for l in open(exp)}
cache = os.path.join(d, ".cache")
if ".cache" in extra and os.path.isdir(cache) and not os.path.islink(cache):
    extra.discard(".cache")
    print("IGNORED .cache/ (download cache directory)")
if extra: print("EXTRA", sorted(extra)); bad += 1
sys.exit(1 if bad else 0)
