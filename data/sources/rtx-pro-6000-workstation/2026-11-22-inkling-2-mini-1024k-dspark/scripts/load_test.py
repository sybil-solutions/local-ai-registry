#!/usr/bin/env python3
"""Aggregate decode throughput at N concurrent users against a running vLLM server (supplementary to the registry's
gates, which measure one user). Same method as the Inkling hill-climb (eval/.../inkling-2-mini/hillclimb/scripts/
hc_run.py load_phase): N closed-loop clients send gate-like prompts (IFEval and MATH-500 alternating, fixed order),
reasoning effort medium, 2,000-token cap per answer by default (--max-tokens 0: none); after every client is running, vLLM's own generated-token counter
(/metrics) is read over a fixed window. Drafts are not counted, only tokens the server emitted.

    load_test.py <endpoint> <served model> <out.jsonl> N [N ...] [--window 60] [--max-tokens 0] [--prompts DIR]
"""
import argparse, json, random, re, statistics, subprocess, threading, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("endpoint")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("levels", type=int, nargs="+")
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--prompts", default="eval/general/src",
                help="folder with ifeval.jsonl (field prompt) and math500.jsonl (field problem)")
ap.add_argument("--max-tokens", type=int, default=2000, help="per-answer limit; 0 = none (the server's own)")
a = ap.parse_args()
URL = a.endpoint.rstrip("/")
EFF = {"chat_template_kwargs": {"reasoning_effort": "medium"}, "reasoning_effort": "medium"}


def metrics():
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode().splitlines():
        if line.startswith("vllm:"):
            name, _, val = line.rpartition(" ")
            try:
                out[name.split("{")[0]] = out.get(name.split("{")[0], 0.0) + float(val)
            except ValueError:
                pass
    return out


def chat(content, seed, stop):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}, **EFF}
    if a.max_tokens:
        body["max_tokens"] = a.max_tokens
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, t1 = time.monotonic(), None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if stop.is_set():
                break
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line != "data: [DONE]" and t1 is None:
                ev = json.loads(line[5:])
                if any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                       or (c.get("delta") or {}).get("reasoning") for c in ev.get("choices") or []):
                    t1 = time.monotonic()
    return {"t0": t0, "ttft": (t1 or time.monotonic()) - t0}


def pool():
    load = lambda n: [json.loads(l) for l in (Path(a.prompts) / f"{n}.jsonl").read_text().splitlines()]  # noqa: E731
    x = [r["prompt"] for r in load("ifeval")]
    y = [r["problem"] + "\n\nPut your final answer in \\boxed{}." for r in load("math500")]
    rng = random.Random(7); rng.shuffle(x); rng.shuffle(y)
    return [p for pair in zip(x, y) for p in pair]


def drain(timeout=180):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        m = metrics()
        if m.get("vllm:num_requests_running", 0) + m.get("vllm:num_requests_waiting", 0) == 0:
            return
        time.sleep(1)


def level(n, prompts):
    stop, lock, nxt, done, errors = threading.Event(), threading.Lock(), [0], [], []

    def worker():
        while not stop.is_set():
            with lock:
                i = nxt[0]; nxt[0] += 1
            try:
                done.append(chat(prompts[i % len(prompts)], 1000 + i, stop))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e)[:200]); time.sleep(1)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    t_r = time.monotonic()
    while time.monotonic() - t_r < 60 and metrics().get("vllm:num_requests_running", 0) < n:
        time.sleep(1)
    time.sleep(5)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used",
                            "--format=csv,noheader,nounits", "-lms", "1000"], stdout=subprocess.PIPE, text=True)
    m0, t0, samples = metrics(), time.monotonic(), []
    while time.monotonic() - t0 < a.window:
        m = metrics()
        samples.append((m.get("vllm:num_requests_running", 0), m.get("vllm:num_requests_waiting", 0),
                        m.get("vllm:kv_cache_usage_perc", 0)))
        time.sleep(1)
    m1, t1 = metrics(), time.monotonic()
    smi.terminate()
    util = [tuple(float(v) for v in l.split(",")) for l in smi.stdout.read().splitlines() if l.count(",") == 3]
    stop.set()
    for t in threads:
        t.join(timeout=120)
    drain()
    d = lambda k: m1.get(k, 0) - m0.get(k, 0)  # noqa: E731
    gen, drafts, acc = d("vllm:generation_tokens_total"), d("vllm:spec_decode_num_drafts_total"), d("vllm:spec_decode_num_accepted_tokens_total")
    running = statistics.mean(s[0] for s in samples)
    in_win = [r for r in done if t0 <= r["t0"] <= t1]
    return {"users": n, "agg_tps": round(gen / (t1 - t0), 1), "window_s": round(t1 - t0, 1),
            "running_mean": round(running, 1), "waiting_mean": round(statistics.mean(s[1] for s in samples), 1),
            "per_running_stream_tps": round(gen / (t1 - t0) / max(running, 1e-9), 1),
            "kv_peak": round(max(s[2] for s in samples), 3),
            "tok_per_step": round(1 + acc / drafts, 2) if drafts else None,
            "ttft_p50_s": round(statistics.median(r["ttft"] for r in in_win), 2) if in_win else None,
            "requests_started_in_window": len(in_win), "errors": len(errors), "error_sample": errors[:2],
            "gpu_util": round(statistics.mean(u[0] for u in util), 1) if util else None,
            "power_w": round(statistics.mean(u[1] for u in util)) if util else None,
            "sm_mhz": round(statistics.mean(u[2] for u in util)) if util else None,
            "gpu_mem_max_mib": int(max(u[3] for u in util)) if util else None}


prompts = pool()
for n in a.levels:
    r = level(n, prompts)
    print(json.dumps(r), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
