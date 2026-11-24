#!/usr/bin/env python3
"""prof.py CONFIG.json — launch with graphs disabled + torch profiler, profile ~N decode steps, aggregate kernels."""
import json, sys, time, glob, subprocess, requests
sys.path.insert(0, ".")
import hc
cfg = json.load(open(sys.argv[1])); cfg["name"] += "-prof"
a = cfg["args"]
for flag in ("--cuda-graph-backend-decode", "--cuda-graph-bs-decode"):
    if flag in a:
        j = a.index(flag); n = 1
        while j + n < len(a) and not a[j + n].startswith("--"): n += 1
        del a[j:j + n]
a.append("--disable-cuda-graph")
cfg["env"]["SGLANG_TORCH_PROFILER_DIR"] = "/traces"
cfg.setdefault("mounts", []).append(f"{hc.HC}/traces:/traces")
subprocess.run("rm -rf traces/*", shell=True)
watch, *_ = hc.launch(cfg)
hc.chat(hc.DECODE_PROMPTS[0], 64, ignore_eos=True)
hc.chat(hc.DECODE_PROMPTS[1], 32, ignore_eos=True)
MODE = sys.argv[2] if len(sys.argv) > 2 else "decode"
if MODE == "prefill":
    text, _ = hc.doc(16384, 3); hc.flush()
    r = requests.post(hc.BASE + "/start_profile", json={"num_steps": 3, "activities": ["CPU", "XPU"], "record_shapes": False}, timeout=60)
    print("start_profile", r.status_code); out = hc.chat(text, 1, temperature=0.0); print("ttft", out["ttft"], out["prompt_tokens"])
    time.sleep(30); watch.stop.set(); print(glob.glob("traces/**", recursive=True)[:10]); sys.exit(0)
if MODE == "c4":
    from concurrent.futures import ThreadPoolExecutor
    def go(i): return hc.chat(hc.DECODE_PROMPTS[i % 3] + f" (v{i})", 200, ignore_eos=True)
    with ThreadPoolExecutor(4) as ex:
        fs = [ex.submit(go, i) for i in range(4)]
        time.sleep(4)
        r = requests.post(hc.BASE + "/start_profile", json={"num_steps": 40, "activities": ["CPU", "XPU"], "record_shapes": False}, timeout=60)
        print("start_profile", r.status_code)
        print("per-stream", [round(f.result()["decode"], 1) for f in fs])
    time.sleep(20); watch.stop.set(); print(glob.glob("traces/**", recursive=True)[:10]); sys.exit(0)
r = requests.post(hc.BASE + "/start_profile", json={"num_steps": 40, "activities": ["CPU", "XPU"], "record_shapes": False}, timeout=60)
print("start_profile", r.status_code, r.text[:200])
out = hc.chat(hc.DECODE_PROMPTS[2], 160, ignore_eos=True)
print("decode", out["decode"])
time.sleep(20)
watch.stop.set()
print(glob.glob("traces/**", recursive=True)[:10])
