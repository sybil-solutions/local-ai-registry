#!/usr/bin/env python3
"""Hill-climb harness for Inkling-2-Mini on SGLang-XPU (trellis-serve) on Arc B390.

  hc.py run CONFIG.json [--long 131072|262000] [--skip-c4]
  hc.py show

A config: {"name", "image"?, "env": {...}, "args": [sglang args], "mounts"?: [...]}.
Each run: recreate container `inkling`, time startup, then measure TTFT, c1/c4 decode,
cold prefill (4K, 32K), optional long-context stability, KV pool and RAM footprint.
A watchdog aborts requests and stops the server if the stack footprint exceeds 46 GiB.
"""
import json, os, statistics, subprocess, sys, threading, time, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

HC = Path(__file__).resolve().parent
LEDGER = HC / "ledger-sgl.jsonl"
BASE = "http://127.0.0.1:8000"
MODEL = "inkling-2-mini"
NAME = "inkling"
RAM_CAP = 46 * 2**30
CACHE = Path.home() / ".cache/inkling-hc"   # persistent JIT/inductor/SYCL caches

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def sh(*a, check=True):
    return subprocess.run(a, check=check, capture_output=True, text=True).stdout.strip()

def mem_used():
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":"); m[k] = int(v.split()[0]) * 1024
    return m["MemTotal"] - m["MemAvailable"]

class Watch:
    """Samples host RAM; footprint = used - baseline. Trips at RAM_CAP."""
    def __init__(self, baseline):
        self.base, self.peak, self.tripped = baseline, 0, False
        self.stop = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while not self.stop.wait(1.5):
            try:  # GPU GTT residency (xe fdinfo) + container anon; MemAvailable delta misses GTT
                fp = max(int(sh(str(HC / "stackmem.sh"), NAME)), mem_used() - self.base)
            except Exception:
                fp = mem_used() - self.base
            self.peak = max(self.peak, fp)
            if fp > RAM_CAP and not self.tripped:
                self.tripped = True
                log(f"RAM CAP EXCEEDED: {fp/2**30:.1f} GiB; aborting")
                try:
                    requests.post(BASE + "/abort_request", json={"abort_all": True}, timeout=10)
                except Exception: pass

def dlogs():
    r = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
    return r.stdout + r.stderr

def drain_stop():
    try:
        requests.post(BASE + "/abort_request", json={"abort_all": True}, timeout=10)
        time.sleep(3)
    except Exception: pass
    subprocess.run(["docker", "stop", "-t", "60", NAME], capture_output=True)
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)

def launch(cfg):
    drain_stop()
    time.sleep(2)
    for d in ("root-cache", "sycl", "neo", "inductor", "triton"):
        (CACHE / d).mkdir(parents=True, exist_ok=True)
    env = {"SYCL_CACHE_PERSISTENT": "0",
           "NEO_CACHE_PERSISTENT": "1", "NEO_CACHE_DIR": "/hc-cache/neo",
           "TORCHINDUCTOR_CACHE_DIR": "/hc-cache/inductor", "TRITON_CACHE_DIR": "/hc-cache/triton",
           **cfg.get("env", {})}
    args = ["docker", "run", "-d", "--name", NAME, "--device", "/dev/dri/renderD128",
            "--shm-size", "4g", "-p", "127.0.0.1:8000:8000",
            "-v", "/home/sero/Models/Inkling-2-Mini:/models:ro",
            "-v", f"{CACHE}:/hc-cache", "-v", f"{CACHE}/root-cache:/root/.cache/hc"]
    for m in cfg.get("mounts", []): args += ["-v", m]
    for k, v in env.items(): args += ["-e", f"{k}={v}"]
    args += ["--entrypoint", "/opt/ornith-entrypoint.sh", cfg.get("image", "ornith-hc:dev"),
             "python3", "-m", "sglang.launch_server", "--model-path", "/models",
             "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", "8000", *cfg["args"]]
    base = mem_used()
    t0 = time.monotonic()
    sh(*args)
    watch = Watch(base)
    deadline = t0 + 1500
    while time.monotonic() < deadline:
        st = sh("docker", "inspect", "-f", "{{.State.Status}}", NAME, check=False)
        if st != "running":
            raise RuntimeError("container exited during startup:\n" + dlogs()[-4000:])
        if watch.tripped: raise RuntimeError("RAM cap exceeded during startup")
        try:
            if requests.get(BASE + "/health", timeout=3).status_code == 200: break
        except requests.RequestException: pass
        time.sleep(1)
    else:
        raise RuntimeError("startup timeout")
    t_health = time.monotonic() - t0
    chat("Say ready.", 4)  # first real generation completes startup
    t_ready = time.monotonic() - t0
    log(f"startup: health {t_health:.1f}s, first-token-ready {t_ready:.1f}s")
    return watch, base, t_health, t_ready

DEC_T = float(os.environ.get("HC_TEMP", "1.0"))

def chat(prompt, max_tokens, ignore_eos=False, stream=True, temperature=None, timeout=3600):
    if temperature is None: temperature = DEC_T
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0,
            "chat_template_kwargs": {"reasoning_effort": os.environ.get("HC_EFFORT", "none")}, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
    start = time.monotonic(); first = None; text = ""; usage = None
    r = requests.post(BASE + "/v1/chat/completions", json=body, stream=True, timeout=timeout)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("error"): raise RuntimeError(d["error"])
        if d.get("usage"): usage = d["usage"]
        for c in d.get("choices", []):
            p = (c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning_content") or "")
            if p:
                first = first or time.monotonic(); text += p
    end = time.monotonic()
    n = usage["completion_tokens"] if usage else 0
    return {"ttft": (first or end) - start, "wall": end - start, "n": n,
            "prompt_tokens": usage["prompt_tokens"] if usage else 0,
            "decode": (n - 1) / (end - first) if first and n > 1 and end > first else None,
            "text": text}

DECODE_PROMPTS = [
    "Write a complete Python implementation of an LRU cache with O(1) get and put, with docstrings, type hints and unit tests.",
    "Explain step by step how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery), with a worked example.",
    "Write a Rust command-line tool that counts word frequencies in a file and prints the top 20, with error handling and comments.",
]

def flush():
    try: requests.post(BASE + "/flush_cache?timeout=30", timeout=45)
    except Exception: pass

WORDS = ("river stone market signal garden copper lantern harbor meadow circuit planet violin "
         "thunder orchard canvas engine marble falcon silver window theory bridge compass ember").split()

def doc(tokens, seed):
    rnd = random.Random(seed)
    markers = [f"marker-{rnd.randint(1000,9999)}-{c}" for c in "abc"]
    def build(nwords):
        w = [rnd.choice(WORDS) for _ in range(nwords)]
        for i, m in sorted(zip((nwords//10, nwords//2, 9*nwords//10), markers), reverse=True):
            w.insert(i, f"\nSecret code {m}.\n")
        return " ".join(w)
    probe = build(2000)
    per_word = requests.post(BASE + "/tokenize", json={"prompt": probe}, timeout=120).json()["count"] / 2000
    text = build(int((tokens - 120) / per_word))
    return text + "\n\nList the three secret codes in order, then summarize the document in two sentences.", markers

def metric_sum(text, name):
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith("sglang:" + name + "{"))

def on_ac():
    import glob
    vals = [open(p).read().strip() for p in glob.glob("/sys/class/power_supply/*/online")]
    return any(v == "1" for v in vals)

def wait_ac():
    if on_ac(): return
    log("on battery: waiting for AC power before benchmarking")
    while not on_ac(): time.sleep(30)
    log("AC power back; settling 60 s"); time.sleep(60)

def bench(cfg, long_ctx=None, skip_c4=False):
    wait_ac()
    rec = {"name": cfg["name"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cfg": cfg}
    try:
        watch, base, rec["startup_health_s"], rec["startup_ready_s"] = launch(cfg)
        info = requests.get(BASE + "/server_info", timeout=30).json()
        rec["kv_pool_tokens"] = info.get("max_total_num_tokens") or (info.get("internal_states") or [{}])[0].get("max_total_num_tokens")
        logs = sh("docker", "logs", NAME, check=False) + subprocess.run(["docker","logs",NAME],capture_output=True,text=True).stderr if False else ""
        import re as _re
        mr = _re.findall(r"max_running_requests=(\d+)", logs)
        rec["max_running"] = int(mr[-1]) if mr else info.get("max_running_requests")
        rec["spec_steps"] = info.get("speculative_num_steps")
        # warmup
        chat("Write a haiku about rivers.", 64)
        chat(DECODE_PROMPTS[0], 128, ignore_eos=True)
        # TTFT short (cache flushed each time)
        tt = []
        for i in range(5):
            flush(); tt.append(chat(f"Reply with one word. Request {i}.", 1)["ttft"])
        rec["ttft_short_ms"] = statistics.median(tt) * 1000
        # c1 decode
        m0 = requests.get(BASE + "/metrics", timeout=15).text
        c1 = [chat(p, 512, ignore_eos=True) for p in DECODE_PROMPTS * int(os.environ.get("HC_C1_REPS", "2"))]
        m1 = requests.get(BASE + "/metrics", timeout=15).text
        rec["c1_decode"] = statistics.median(r["decode"] for r in c1)
        rec["c1_decode_all"] = [round(r["decode"], 2) for r in c1]
        rec["c1_sample"] = c1[0]["text"][:300]
        gen = metric_sum(m1, "generation_tokens_total") - metric_sum(m0, "generation_tokens_total")
        ver = metric_sum(m1, "spec_verify_calls_total") - metric_sum(m0, "spec_verify_calls_total")
        rec["accept_len"] = gen / ver if ver else None
        log(f"c1 {rec['c1_decode']:.1f} tok/s  accept {rec['accept_len']}  ttft {rec['ttft_short_ms']:.0f} ms")
        if not skip_c4 and (rec["max_running"] or 1) >= 4:
            ps = [DECODE_PROMPTS[i % 3] + f" (variant {i})" for i in range(4)]
            t0 = time.monotonic()
            with ThreadPoolExecutor(4) as ex:
                rs = list(ex.map(lambda p: chat(p, 512, ignore_eos=True), ps))
            wall = time.monotonic() - t0
            rec["c4_decode_sum"] = sum(r["n"] - 1 for r in rs) / (max(r["wall"] for r in rs) - min(r["ttft"] for r in rs))
            rec["c4_decode_per_stream"] = [round(r["decode"], 2) for r in rs]
            rec["c4_aggregate_wall"] = sum(r["n"] for r in rs) / wall
            log(f"c4 sum {rec['c4_decode_sum']:.1f} tok/s  wall-agg {rec['c4_aggregate_wall']:.1f}")
        # cold prefill
        for n in (4096, 32768):
            flush()
            text, mk = doc(n, n)
            r = chat(text, 1, temperature=0.0)
            rec[f"prefill_{n//1024}k"] = r["prompt_tokens"] / r["ttft"]
            rec[f"ttft_{n//1024}k_s"] = r["ttft"]
            log(f"prefill {n//1024}K: {rec[f'prefill_{n//1024}k']:.0f} tok/s ({r['ttft']:.2f}s)")
        if long_ctx:
            flush()
            text, mk = doc(long_ctx, 7)
            r = chat(text, 256, temperature=0.0)
            k = f"long_{long_ctx//1024}k"
            rec[k] = {"prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft"],
                      "prefill": r["prompt_tokens"] / r["ttft"], "decode": r["decode"],
                      "retrieval": sum(m in r["text"] for m in mk)}
            log(f"{k}: {rec[k]}")
        rec["ram_peak_gib"] = watch.peak / 2**30
        rec["ram_cap_ok"] = not watch.tripped
        rec["ok"] = not watch.tripped
        watch.stop.set()
    except Exception as e:
        rec["ok"] = False; rec["error"] = str(e)[-3000:]
        log("FAILED:", rec["error"][-800:])
    try: (HC / "logs" / f"{cfg['name']}.log").write_text(dlogs())
    except Exception: pass
    rec["on_ac_end"] = on_ac()
    if not rec["on_ac_end"]: rec["ok"] = False; rec["error"] = (rec.get("error") or "") + " [battery: invalid]"
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    summary = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("cfg",)}
    print(json.dumps(summary, indent=1))
    return rec

def show():
    keys = ["name", "ok", "startup_ready_s", "ttft_short_ms", "c1_decode", "accept_len", "c4_decode_sum",
            "prefill_4k", "prefill_32k", "kv_pool_tokens", "ram_peak_gib"]
    print(" | ".join(keys))
    for l in LEDGER.read_text().splitlines():
        r = json.loads(l)
        print(" | ".join(f"{r.get(k):.1f}" if isinstance(r.get(k), float) else str(r.get(k)) for k in keys))

if __name__ == "__main__":
    if sys.argv[1] == "show": show()
    elif sys.argv[1] == "run":
        cfg = json.loads(Path(sys.argv[2]).read_text())
        lc = int(sys.argv[sys.argv.index("--long") + 1]) if "--long" in sys.argv else None
        bench(cfg, lc, "--skip-c4" in sys.argv)
    elif sys.argv[1] == "stop": drain_stop()
