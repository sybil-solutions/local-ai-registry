#!/usr/bin/env python3
"""1 Hz memory sampler for a llama-server run on Apple silicon.
Writes one JSON line per second: the server pid's RSS and physical footprint, and system vm_stat
(free / wired / compressed / swapouts) plus iogpu.wired_limit_mb. Every number is read, none is derived.
usage: memwatch_llamacpp.py <pid> <out.jsonl>"""
import json, re, subprocess, sys, time

pid, out = int(sys.argv[1]), sys.argv[2]
PAGE = int(subprocess.run(["sysctl", "-n", "hw.pagesize"], capture_output=True, text=True).stdout)

def vmstat():
    txt = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", txt).group(1)) * PAGE / 2**30
    return {"free_gib": round(get("Pages free"), 2), "wired_gib": round(get("Pages wired down"), 2),
            "compressed_gib": round(get("Pages occupied by compressor"), 2),
            "swapouts": int(re.search(r"Swapouts:\s+(\d+)", txt).group(1))}

def proc():
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if not r:
        return None
    fp = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s+([\d.]+)\s*([KMG]B)", fp)
    foot = float(m.group(1)) * {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1}[m.group(2)] if m else None
    return {"rss_gib": round(int(r) / 2**20, 2), "footprint_gib": round(foot, 2) if foot else None}

limit = int(subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True).stdout)
with open(out, "a") as f:
    while True:
        p = proc()
        if p is None:
            break
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": pid, "wired_limit_mb": limit, **p, **vmstat()}) + "\n")
        f.flush()
        time.sleep(1)
