#!/usr/bin/env python3
"""Head-to-head of two engines serving the same weights on the same card: a controlled experiment.

    lab/compare.py <recipe A> <recipe B> [--replay ~/tuning-kit/omp_replay_corpus.jsonl] [--levels 1,2,4,8]

Hypothesis: B (e.g. SGLang-EXL3) is faster than A (e.g. TabbyAPI on ExLlamaV3) on this card.

Controlled: both recipes passed the six gates; same weights (repo@revision), same context window, same number of
requests served at once (>= the highest level), the same KV pool in tokens (`pool`), same sampling and per-request seed, same prompts
in the same order. Both run on the same rented machine, one after the other: A boots and is measured and destroyed,
then B is rented on the machine A ran on.

Measured at each concurrency level C (the first request after boot is a discarded warm-up):
  prefill:  PREFILL_WAVES waves of C concurrent cold prompts of N tokens (unique random prefix, so no cache can
            help); one sample per wave = C*N / slowest time to first token.
  decode:   DECODE_WAVES waves of C of your agent sessions replayed turn by turn (tools, thinking, growing context,
            no output cap); per-stream samples = (tokens - 1) / (last token - first token) per request, aggregate
            samples = output tokens / wall time per wave.
  Tokens are counted with the model's tokenizer on the client, identically for both engines.

Decision per metric and level: a bootstrap 95% confidence interval of median(B) / median(A). B is faster only if
the whole interval is above 1.0, A only if it is entirely below; otherwise no difference is shown. B replaces A on
the card only if B is faster on prefill, per-stream decode and aggregate decode at every level with no failed
request. The decision goes to registry/prefer.json; every sample, both launches and the host go to lab/runs/.
The replay corpus is private and never enters the repository.
"""
import argparse, asyncio, datetime as dt, json, random, statistics, string, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab
import aiohttp
from tokenizers import Tokenizer

PREFER = lab.REG / "prefer.json"
PREFILL_WAVES, DECODE_WAVES, TURNS, BOOT = 5, 3, 2, 2000


def tokenizer(weights):
    repo, rev = weights.split("@")
    return Tokenizer.from_str(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{rev}/tokenizer.json", timeout=120).read().decode())


async def stream(sess, url, body):
    """One streamed request: when the first and last token arrived, and everything that came back."""
    rec = {"t0": time.perf_counter(), "text": "", "ok": False}
    try:
        async with sess.post(url + "/v1/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=7200)) as r:
            if r.status != 200:
                rec["err"] = f"{r.status} {(await r.text())[:200]}"
                return rec
            async for line in r.content:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError):
                    continue
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                piece += "".join((t.get("function") or {}).get("arguments") or "" for t in d.get("tool_calls") or [])
                if piece:
                    rec.setdefault("t_first", time.perf_counter())
                    rec["t_last"] = time.perf_counter()
                    rec["text"] += piece
            rec["ok"] = True
    except Exception as e:
        rec["err"] = repr(e)[:200]
    return rec


def workload(corpus, tok, ctx, levels, seed):
    """Every prompt both engines will see, fixed before either boots: cold prefill prompts and replay turns per wave."""
    rng = random.Random(seed)
    words = open("/usr/share/dict/words").read().split() if Path("/usr/share/dict/words").exists() else list(string.ascii_lowercase)
    n = min(8192, ctx // 2)
    fits = []
    for s in corpus:
        ok = [c for c in s["cuts"] if len(tok.encode("\n".join(m.get("content") or "" for m in s["messages"][:c])).ids) <= int(ctx * 0.6)]
        if len(ok) >= TURNS:
            fits.append((s, ok))
    if not fits:
        raise SystemExit("no replay session fits this context")
    rng.shuffle(fits)
    plan, k = {}, 0
    for C in levels:
        prefill = []
        for _ in range(PREFILL_WAVES):
            wave = []
            for _ in range(C):
                text = "".join(rng.choice(string.ascii_lowercase) for _ in range(24)) + "\n"
                while len(tok.encode(text).ids) < n:
                    text += " ".join(rng.choice(words) for _ in range(300)) + "\n"
                wave.append(text + "\nReply with only the word OK.")
            prefill.append(wave)
        decode = []
        for _ in range(DECODE_WAVES):
            wave = []
            for _ in range(C):
                s, cuts = fits[k % len(fits)]
                k += 1
                start = rng.randrange(0, len(cuts) - TURNS + 1)
                wave.append([{"messages": s["messages"][:c], "tools": s.get("tools") or None} for c in cuts[start:start + TURNS]])
            decode.append(wave)
        plan[C] = {"prefill": prefill, "decode": decode, "prefill_tokens": n}
    return plan


def med(xs):
    return round(statistics.median(xs), 1) if xs else None


async def run(url, served, tok, plan, levels):
    body = lambda msgs, extra={}: {"model": served, "stream": True, "temperature": 0.6, "top_p": 0.95, "seed": 7, "messages": msgs, **extra}
    out = {}
    async with aiohttp.ClientSession() as sess:
        await stream(sess, url, body([{"role": "user", "content": "Say hi."}]))  # warm-up, discarded
        for C in levels:
            p = plan[C]
            prefill, per, agg, fails = [], [], [], 0
            for wave in p["prefill"]:
                recs = await asyncio.gather(*[stream(sess, url, body([{"role": "user", "content": t}])) for t in wave])
                fails += sum(not r["ok"] for r in recs)
                if all(r.get("t_first") for r in recs):
                    prefill.append(C * p["prefill_tokens"] / max(r["t_first"] - r["t0"] for r in recs))
            for wave in p["decode"]:
                recs, t0 = [], time.perf_counter()

                async def agent(turns):
                    for t in turns:
                        r = await stream(sess, url, body(t["messages"], {"chat_template_kwargs": {"enable_thinking": True},
                                                                         **({"tools": t["tools"]} if t["tools"] else {})}))
                        r["tokens"] = len(tok.encode(r["text"]).ids)
                        recs.append(r)
                await asyncio.gather(*[agent(turns) for turns in wave])
                wall = time.perf_counter() - t0
                fails += sum(not r["ok"] for r in recs)
                ok = [r for r in recs if r["ok"] and r.get("t_first")]
                per += [(r["tokens"] - 1) / (r["t_last"] - r["t_first"]) for r in ok if r["tokens"] > 32 and r["t_last"] > r["t_first"]]
                agg.append(sum(r["tokens"] for r in ok) / wall)
            out[C] = {"prefill": [round(x, 1) for x in prefill], "decode_per_stream": [round(x, 1) for x in per],
                      "decode_aggregate": [round(x, 1) for x in agg], "fails": fails}
            lab.log(f"{served} C={C}: prefill {med(prefill)}, per-stream decode {med(per)}, aggregate decode {med(agg)}, fails {fails}")
    return out


def ratio_ci(a, b, n=4000, seed=11):
    """Bootstrap 95% CI of median(b) / median(a), with the point estimate in the middle."""
    if not a or not b:
        return None
    rng = random.Random(seed)
    rs = sorted(statistics.median(rng.choices(b, k=len(b))) / statistics.median(rng.choices(a, k=len(a))) for _ in range(n))
    return [round(rs[int(0.025 * n)], 3), round(statistics.median(b) / statistics.median(a), 3), round(rs[int(0.975 * n)], 3)]


def boot(recipe, launch, args):
    """Rent and start one arm; on a pinned machine, wait for the GPU the other arm gave back."""
    t0 = time.monotonic()
    while True:
        try:
            return lab.rented(recipe, launch, args)
        except SystemExit as e:
            if not args.machine or time.monotonic() - t0 > BOOT:
                raise
            lab.log(f"waiting for machine {args.machine}: {e}")
            time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--replay", default=str(Path.home() / "tuning-kit" / "omp_replay_corpus.jsonl"))
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--any-host", action="store_true")
    ap.add_argument("--max-price", type=float, default=1.5)
    ap.add_argument("--min-inet", type=int, default=300)
    ap.add_argument("--disk", type=int, default=90)
    a = ap.parse_args()
    ra, rb = json.loads(Path(a.a).read_text(encoding="utf-8")), json.loads(Path(a.b).read_text(encoding="utf-8"))
    la, lb = lab.render(ra), lab.render(rb)
    levels = [int(x) for x in a.levels.split(",")]
    # the controls: an unfair comparison is refused, not run
    problems = [f"{k} differs: {x} vs {y}" for k, x, y in (("card", ra["card"], rb["card"]), ("model", ra["model"], rb["model"]),
                ("weights", ra["weights"], rb["weights"]), ("context", la["ctx"], lb["ctx"]), ("concurrency", la["seqs"], lb["seqs"]),
                ("KV pool (set pool=)", ra.get("set", {}).get("pool"), rb.get("set", {}).get("pool"))) if x != y]
    if not ra.get("set", {}).get("pool"):
        problems.append("no KV pool set: give both arms the same `pool` (TabbyAPI cache_size, SGLang --max-total-tokens)")
    chunks = [(l.get("config") or {}).get("text", "").split("chunk_size: ")[1].split()[0] if "chunk_size: " in ((l.get("config") or {}).get("text") or "")
              else l["args"][l["args"].index("--chunked-prefill-size") + 1] if "--chunked-prefill-size" in l["args"] else None for l in (la, lb)]
    if chunks[0] != chunks[1]:
        problems.append(f"prefill chunk differs: {chunks[0]} vs {chunks[1]}")
    for name, r, l in (("A", ra, la), ("B", rb, lb)):  # the pool must reach the engine, not only the recipe
        pool = str(r.get("set", {}).get("pool"))
        if f"cache_size: {pool}" not in ((l.get("config") or {}).get("text") or "") and f"--max-total-tokens {pool}" not in " ".join(l["args"]):
            problems.append(f"{name}'s launch does not apply pool {pool}")
    if la["seqs"] < max(levels):
        problems.append(f"the arms serve {la['seqs']} requests at once; the highest level is {max(levels)}")
    if problems:
        raise SystemExit("not a controlled comparison: " + "; ".join(problems))
    tok = tokenizer(ra["weights"])
    plan = workload([json.loads(l) for l in open(a.replay)], tok, la["ctx"], levels, f"{ra['card']}.{ra['model']}")
    args = argparse.Namespace(on="vast", min_inet=a.min_inet, min_cuda=max(la.get("min_cuda") or 12.9, lb.get("min_cuda") or 12.9),
                              max_price=a.max_price, disk=a.disk, proxy_gpu=None, proxy_vram=0, gpu=None, any_host=a.any_host, machine=None)
    samples, hosts = {}, {}
    for arm, recipe, launch in (("a", ra, la), ("b", rb, lb)):
        endpoint, where, stop = boot(recipe, launch, args)
        try:
            served = json.loads(urllib.request.urlopen(endpoint + "/v1/models", timeout=60).read())["data"][0]["id"]
            hosts[arm] = {"gpu": where.get("gpu"), "host": where.get("host"), "machine": where["handle"].get("machine"),
                          "instance": where["handle"].get("id"), "served": served, "image": launch["image"], "args": launch["args"],
                          "config_sha256": (launch.get("config") or {}).get("sha256")}
            samples[arm] = asyncio.run(run(endpoint, served, tok, plan, levels))
        finally:
            stop()
        args.machine = where["handle"].get("machine")  # B runs where A ran
    if hosts["a"]["machine"] != hosts["b"]["machine"]:
        raise SystemExit("B did not land on A's machine; the comparison is void")
    verdict, replaces = {}, True
    for C in levels:
        verdict[C] = {}
        for m in ("prefill", "decode_per_stream", "decode_aggregate"):
            ci = ratio_ci(samples["a"][C][m], samples["b"][C][m])
            verdict[C][m] = {"median_a": med(samples["a"][C][m]), "median_b": med(samples["b"][C][m]), "ratio_ci95": ci,
                             "result": "b faster" if ci and ci[0] > 1 else "a faster" if ci and ci[2] < 1 else "no difference shown"}
            replaces = replaces and verdict[C][m]["result"] == "b faster"
        replaces = replaces and not samples["b"][C]["fails"]
    at = dt.datetime.now(dt.timezone.utc)
    ea, eb = lab.profile(ra["engine"]).get("engine"), lab.profile(rb["engine"]).get("engine")
    lab.RUNS.mkdir(parents=True, exist_ok=True)
    run_file = lab.RUNS / f"{ra['card']}.{ra['model']}.compare.{at.strftime('%Y%m%dT%H%M%S')}.json"
    run_file.write_text(json.dumps({"hypothesis": f"{eb} is faster than {ea} for {ra['model']} on {ra['card']}",
                                    "criteria": "Decision per metric" + __doc__.split("Decision per metric")[1].split("The decision goes")[0].rstrip(),
                                    "a": {"recipe": a.a, **hosts["a"]}, "b": {"recipe": a.b, **hosts["b"]}, "levels": levels,
                                    "waves": {"prefill": PREFILL_WAVES, "decode": DECODE_WAVES, "turns": TURNS},
                                    "samples": samples, "verdict": verdict, "b_replaces_a": replaces, "at": at.isoformat()}, indent=1) + "\n", encoding="utf-8")
    print(f"\n{ra['card']} {ra['model']}: {eb} (B) vs {ea} (A), same weights, machine {hosts['a']['machine']}")
    print(f"{'C':>3} {'metric':<18} {'A':>9} {'B':>9}  B/A [95% CI]           result")
    for C in levels:
        for m, v in verdict[C].items():
            ci = v["ratio_ci95"] or [None] * 3
            print(f"{C:>3} {m:<18} {v['median_a']!s:>9} {v['median_b']!s:>9}  {ci[1]} [{ci[0]}, {ci[2]}]  {v['result']}")
    prefer = json.loads(PREFER.read_text(encoding="utf-8")) if PREFER.exists() else {}
    prefer[f"{ra['card']}/{ra['model']}"] = {"engine": eb if replaces else ea, "over": ea if replaces else eb, "weights": ra["weights"],
                                             "at": at.strftime("%Y-%m-%d"), "evidence": run_file.name,
                                             "c": {str(C): {m: v["ratio_ci95"] for m, v in verdict[C].items()} for C in levels}}
    PREFER.write_text(json.dumps(dict(sorted(prefer.items())), indent=1) + "\n", encoding="utf-8")
    lab.log(f"{'B replaces A' if replaces else 'A stays'}; evidence {run_file.relative_to(lab.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
