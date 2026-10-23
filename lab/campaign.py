#!/usr/bin/env python3
"""Run many lab.py tries at once: one line per job in a plan file, `<card> <weights> <model> <set>...`.
Skips a job whose recipe already exists; at most --jobs run together. Logs go to lab/runs/<card>.<model>.log."""
import argparse, json, shlex, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--engine", default="tabbyapi-exl3")
ap.add_argument("--max-price", default="2.0")
ap.add_argument("--min-cuda", default="13.0")
ap.add_argument("--min-inet", default="300")
ap.add_argument("--disk", default="90")
ap.add_argument("--any-host", action="store_true", help="vast: also unverified hosts")
a = ap.parse_args()
todo = []
for line in Path(a.plan).read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    card, weights, model, *sets = shlex.split(line)
    r = {"model": model, "weights": weights, "engine": f"{a.engine}@{lab.profile(a.engine)['image'].split('@sha256:')[1][:12]}", "card": card,
         "set": {k: v for k, v in (s.split("=", 1) for s in sets)}}
    path = lab.recipe_path(r, lab.render({**r, "set": {k: int(v) if v.isdigit() else v for k, v in r["set"].items()}}))
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["engine"].split("@")[0] == a.engine:  # another engine's recipe there is no reason to skip
        print(f"skip {card} {model}: recipe exists"); continue
    todo.append((card, model, ["python3", str(lab.ROOT / "lab" / "lab.py"), "try", weights, "--model", model, "--engine", a.engine, "--card", card,
                               "--max-price", a.max_price, "--min-cuda", a.min_cuda, "--min-inet", a.min_inet, "--disk", a.disk] + (["--any-host"] if a.any_host else []) + [x for s in sets for x in ("--set", s)]))
running = []
while todo or running:
    while todo and len(running) < a.jobs:
        card, model, argv = todo.pop(0)
        log = open(lab.RUNS / f"{card}.{model}.log", "w")
        running.append((card, model, subprocess.Popen(argv, stdout=log, stderr=log)))
        time.sleep(3)
    for job in running[:]:
        if job[2].poll() is not None:
            running.remove(job)
            print(f"{'PASS' if job[2].returncode == 0 else 'FAIL'} {job[0]} {job[1]}", flush=True)
    time.sleep(10)
