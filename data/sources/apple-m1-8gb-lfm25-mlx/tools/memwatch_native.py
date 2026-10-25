#!/usr/bin/env python3
"""Native memory watcher, written for this task because the kit's tools/memwatch.sh needs Linux `bc` (absent on the target host)
and /proc/meminfo (absent on macOS). NOT the kit tool; its CSV is labeled native. One line per second until killed.

  memwatch_native.py OUT.csv [--nvidia-smi PATH] [--pid PID]

Linux: mem_available_gb = MemAvailable, swap_used_gb = SwapTotal-SwapFree, gpu_used_mib = nvidia-smi memory.used (sum),
       gpu_util_pct, gpu_power_w, gpu_sm_mhz (read only).
macOS: mem_available_gb = (free+inactive+purgeable+speculative pages) x page size from vm_stat (an approximation macOS
       itself calls reclaimable, not MemAvailable), swap_used_gb from vm.swapusage, gpu_in_use_gb = AGXAccelerator
       PerformanceStatistics "In use system memory", gpu_util_pct = "Device Utilization %".
--pid adds rss_gb of that process (ps)."""
import re, subprocess, sys, time, platform

def sh(cmd, t=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout
    except Exception:
        return ""

def rss(pid):
    o = sh(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    return round(int(o) * 1024 / 1e9, 3) if o.isdigit() else ""

def linux(smi):
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    row = [round(m["MemAvailable"] / 1e9, 3), round((m["SwapTotal"] - m["SwapFree"]) / 1e9, 3)]
    o = sh([smi, "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"])
    vals = [x.strip() for x in o.splitlines()[0].split(",")] if o.strip() else ["", "", "", ""]
    return row + vals

def mac():
    o = sh(["vm_stat"])
    ps = int(re.search(r"page size of (\d+)", o).group(1))
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", o).group(1))
    avail = (g("Pages free") + g("Pages inactive") + g("Pages purgeable") + g("Pages speculative")) * ps
    sw = sh(["sysctl", "-n", "vm.swapusage"])
    used = re.search(r"used = ([\d.]+)M", sw)
    io = sh(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"])
    inuse = re.search(r'"In use system memory"=(\d+)', io)
    util = re.search(r'"Device Utilization %"=(\d+)', io)
    return [round(avail / 1e9, 3), round(float(used.group(1)) / 1000, 3) if used else "",
            round(int(inuse.group(1)) / 1e9, 3) if inuse else "", util.group(1) if util else ""]

def main():
    out = sys.argv[1]
    smi = sys.argv[sys.argv.index("--nvidia-smi") + 1] if "--nvidia-smi" in sys.argv else "nvidia-smi"
    pid = sys.argv[sys.argv.index("--pid") + 1] if "--pid" in sys.argv else None
    is_mac = platform.system() == "Darwin"
    head = (["t", "mem_available_gb_approx", "swap_used_gb", "gpu_in_use_gb", "gpu_util_pct"] if is_mac else
            ["t", "mem_available_gb", "swap_used_gb", "gpu_used_mib", "gpu_util_pct", "gpu_power_w", "gpu_sm_mhz"]) + ["rss_gb"]
    with open(out, "w") as f:
        f.write("# memwatch_native.py (task tool, not kit memwatch.sh) on " + platform.platform() + "\n" + ",".join(head) + "\n")
        while True:
            row = [int(time.time())] + (mac() if is_mac else linux(smi)) + [rss(pid) if pid else ""]
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            time.sleep(1)

if __name__ == "__main__":
    main()
