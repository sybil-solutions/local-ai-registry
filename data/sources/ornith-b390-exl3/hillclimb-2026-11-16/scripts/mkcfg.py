#!/usr/bin/env python3
"""mkcfg.py BASE NEW [-e K=V]... [-s --flag=val]... [-a --flag[=val]]... [-r --flag] [-i image] [-m mount]"""
import json, sys, copy
base, new, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
c = copy.deepcopy(json.load(open(f"configs/{base}.json"))); c["name"] = new
a = c["args"]; i = 0
while i < len(rest):
    op, v = rest[i], rest[i + 1]; i += 2
    if op == "-e": k, val = v.split("=", 1); c["env"][k] = val
    elif op == "-s":
        k, val = v.split("=", 1); j = a.index(k); vals = val.split()
        n = 1
        while j + n < len(a) and not a[j + n].startswith("--"): n += 1
        a[j + 1:j + n] = vals
    elif op == "-a":
        k, _, val = v.partition("="); a += [k] + (val.split() if val else [])
    elif op == "-r":
        j = a.index(v); n = 1
        while j + n < len(a) and not a[j + n].startswith("--"): n += 1
        del a[j:j + n]
    elif op == "-i": c["image"] = v
    elif op == "-m": c.setdefault("mounts", []).append(v)
json.dump(c, open(f"configs/{new}.json", "w"), indent=1)
print(new, " ".join(a[-12:]))
