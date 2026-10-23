#!/usr/bin/env python3
"""The recipe lab: run a model on a card, check it, and only then write its recipe.

A recipe is the output of a passing run, never written by hand. It names the weights (pinned), an
engine profile (pinned), the few settings that differ from the profile's defaults, the card, and the
proof. Everything a program needs to launch it is rendered from the profile, so a recipe is ~600 bytes.

    lab.py try <repo>@<revision> --model <id> --engine <profile> --card <card> [--set k=v ...]
               [--on vast|runpod|endpoint] [--endpoint URL] [--proxy-gpu NAME]
    lab.py render <recipe.json>        # the launch a program runs, as JSON
    lab.py check                       # every recipe: format, pins, profile, card, proof (CI)

Gates (all must pass): load, chat, reasoning, tools, context, speed. The full evidence goes to
lab/runs/ (data, kept); the recipe goes to recipes/<card>/ (production). Standard library only.
"""

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import shlex
import string
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "registry"
ENGINES, RECIPES, RUNS, CARDS = REG / "engines", REG / "recipes", ROOT / "lab" / "runs", REG / "cards"
LAUNCHES = REG / "launches"  # launches frozen exactly as they were validated, one per recipe, until a lab run replaces them
GATES = ["load", "chat", "reasoning", "tools", "context", "speed"]
MIN_TPS = 15.0
MAX_RECIPE_BYTES = 1024
# The KV floor: a setup with more than KV_FLOOR_ABOVE_VRAM_GB of GPU memory (the card's vram_gb x cards x machines)
# must hold at least KV_FLOOR_TOKENS (128k) of KV cache. `try` refuses a launch below it, `check` fails a new recipe
# below it, and the plugin export leaves every recipe below it out. Recipes accepted before the floor are listed in
# registry/KV_FLOOR_EXEMPT: they stay in the registry with their evidence, but never reach the plugin.
KV_FLOOR_TOKENS = 131072
KV_FLOOR_ABOVE_VRAM_GB = 8
KV_FLOOR_EXEMPT = "kv-floor-exempt.json"  # next to recipes/


def log(msg):
    print(f"[lab {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- rendering
def config_text(p):
    """A profile's config file: inline `text`, or a `file` next to the profile (a host launcher script)."""
    cfg = p.get("config")
    if not cfg:
        return None
    return cfg["text"] if "text" in cfg else profile_file(cfg["file"]).read_text()


def pin(p):
    """What a recipe pins a profile to: its image digest, for an image built from a pinned commit (`build`, no image) that
    commit, or for a host program the sha256 of everything it launches, so a change to the command, packages, weights or
    launcher invalidates the recipes like a new image would."""
    if p.get("kind") != "host":
        if not p.get("image"):
            if not (p.get("build") or {}).get("commit"):
                raise SystemExit(f"{p.get('id')}: a launch without an image needs build.commit to pin")
            return p["build"]["commit"]
        return p["image"].split("@sha256:")[1]
    keys = ("command", "install", "pip", "env", "port", "weights", "ctx", "seqs", "vision", "wired_limit_reserve_mb")
    # several cards (or machines) count too, but one is the default and stays out, so every single-card pin is unchanged
    topology = {k: p[k] for k in ("cards", "machines") if p.get(k, 1) != 1}
    return hashlib.sha256(json.dumps({**{k: p.get(k) for k in keys}, **topology, "config": config_text(p)}, sort_keys=True).encode()).hexdigest()


def profile_file(name):
    """A template profile in engines/, or a frozen launch in launches/."""
    return next((d / name for d in (ENGINES, LAUNCHES) if (d / name).exists()), ENGINES / name)


def profile(ref):
    """`tabbyapi-exl3@0f83e6198dc3` -> the profile, checked against the digest prefix."""
    name, _, digest = ref.partition("@")
    p = json.loads(profile_file(f"{name}.json").read_text())
    if p.get("kind") == "host" or not p.get("image"):  # a host program, or an image built from a pinned commit
        if p.get("kind") == "host":
            stale = digest and digest != "host" and not pin(p).startswith(digest)
        else:  # a recipe frozen before build commits were pinned says `@build`; it stays valid
            stale = digest and digest != "build" and not pin(p).startswith(digest)
        if stale:
            raise SystemExit(f"{ref}: the profile now pins {pin(p)[:12]}; rerun the recipe")
        return p
    if digest and "@sha256:" in p["image"] and not p["image"].split("@sha256:")[1].startswith(digest):
        raise SystemExit(f"{ref}: the profile's image is now {p['image']}; rerun the recipe")
    return p


def card(card_id):
    """The card record: cards/<vendor>/<card>.json."""
    found = list(CARDS.glob(f"*/{card_id}.json"))
    if not found:
        raise SystemExit(f"unknown card {card_id}")
    return json.loads(found[0].read_text())


def recipe_path(r, launch):
    """recipes/<vendor>/<card>/<model>.<engine kind>.<context>k[.<n>x][.<variant>].json, n = every card in the setup;
    variant = the launch's optional `variant` (a-z, 0-9, dashes), for several recipes of one model, engine kind and
    context on one card that differ in what the host needs (e.g. ram28, ram58: experts in RAM vs read from NVMe)"""
    kind = profile(r["engine"]).get("engine", r["engine"].split("@")[0])
    n = launch.get("machines", 1) * launch.get("cards", 1)
    many = f".{n}x" if n > 1 else ""  # several cards, in one computer or across several
    var = launch.get("variant") or ""
    assert not var or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", var), f"variant {var!r}"
    var = f".{var}" if var else ""
    return RECIPES / card(r["card"])["vendor"] / r["card"] / f"{r['model']}.{kind}.{launch['ctx'] // 1024}k{many}{var}.json"


def dirname(weights):
    repo, rev = weights.split("@")
    return f"{repo.split('/')[1]}-{rev[:8]}"


def render(recipe):
    """The launch contract for a recipe: image, entrypoint, args, env, port, shm, weights and config file."""
    p = profile(recipe["engine"])
    if "defaults" not in p:  # a frozen profile: the launch exactly as it was validated
        cfg = p.get("config")
        if p.get("kind") == "host":  # a program on the host, not a container
            out = {"kind": "host", "command": p["command"], "install": p.get("install"), "port": p["port"], "image": None,
                   "weights": p.get("weights") or [], "config": None, "ctx": p["ctx"], "seqs": p.get("seqs", 1),
                   "vision": p.get("vision", False), "backend": p.get("backend"), "cards": p.get("cards", 1),
                   "env": p.get("env") or {}, **{k: p[k] for k in ("setup", "source", "needs", "variant") if k in p},
                   **({"machines": p["machines"]} if p.get("machines", 1) != 1 else {})}  # one machine is the default and stays out
            # MLX on Apple silicon: pinned packages, a launcher file, its environment, and a GPU wired-memory limit that
            # follows the card's unified memory
            out.update({k: p[k] for k in ("pip",) if p.get(k)})
            text = config_text(p)
            if text:
                out["config"] = {"at": p["config"]["at"], "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            if "wired_limit_reserve_mb" in p:
                out["sysctl"] = {"iogpu.wired_limit_mb": card(recipe["card"])["vram_gb"] * 1024 - p["wired_limit_reserve_mb"]}
            return out
        return {"image": p["image"], "entrypoint": p.get("entrypoint"), "args": p["args"], "port": p["port"], "shm": p.get("shm"),
                **{k: p[k] for k in ("flags", "machines", "build", "setup", "source", "needs", "prepare", "resources", "companion", "variant") if k in p},
                "env": p.get("env") or {}, "weights": p["weights"],
                "config": {**cfg, "sha256": hashlib.sha256(cfg["text"].encode()).hexdigest()} if cfg else None,
                "ctx": p["ctx"], "seqs": p.get("seqs", 1), "vision": p.get("vision", False), "backend": p.get("backend"), "cards": p.get("cards", 1)}
    s = {**p["defaults"], **recipe.get("set", {})}
    name = dirname(recipe["weights"])
    values = {**{k: str(v).lower() if isinstance(v, bool) else v for k, v in s.items()}, "name": name,
              "cache_tokens": int(s.get("pool") or int(s["ctx"]) * int(s["seqs"]) + 1024 * int(s["seqs"])),
              "draft_block": "{draft_mode: mtp}" if s.get("draft") == "mtp" else "{}"}
    sub = lambda t: string.Template(t).substitute(values)
    args = []
    for a in p["args"]:  # an argument that is one whole ${setting} may hold several words, or none
        args += shlex.split(sub(a)) if re.fullmatch(r"\$\{\w+\}", a) else [sub(a)]
    config = "\n".join(sub(line) for line in p["config"]) + "\n" if p.get("config") else None
    repo, rev = recipe["weights"].split("@")
    return {"image": p["image"], "entrypoint": p["entrypoint"], "args": args, "port": p["port"], "shm": p["shm"],
            "env": {k: sub(v) for k, v in (p.get("env") or {}).items()},
            "weights": {"repo": repo, "revision": rev, "at": sub(p["weights_at"])},
            "config": {"at": p["config_at"], "text": config, "sha256": hashlib.sha256(config.encode()).hexdigest()} if config else None,
            "ctx": int(s["ctx"]), "seqs": int(s["seqs"]), "vision": bool(s.get("vision", False)), "backend": p["backend"], "min_cuda": p.get("min_cuda"),
            # an offload profile states what the host must have, from its settings (experts in RAM, tables on NVMe)
            **({"needs": {k: (v if not isinstance(v, str) or k == "fast_storage" else float(sub(v)))
                          for k, v in p["needs"].items()}} if p.get("needs") else {})}


def kv_tokens(recipe, launch=None):
    """The KV cache a recipe holds, as the plugin is told it (serving.kvTokens): a launch's own plugin block, else
    ctx x seqs + 1024 x seqs. A plugin block that reports kvTokens 0 (llama.cpp) counts its ctxTokens."""
    p = profile(recipe["engine"])
    if "plugin" in p:
        s = p["plugin"]["serving"]
        return int(s.get("kvTokens") or s.get("ctxTokens") or 0)
    L = launch or render(recipe)
    return int(L["ctx"]) * int(L["seqs"]) + 1024 * int(L["seqs"])


def setup_vram_gb(recipe, launch=None, vram_gb=None):
    """GPU memory of the whole setup the recipe runs on: the card's vram_gb (or the one given) x cards x machines."""
    L = launch or render(recipe)
    per_card = card(recipe["card"])["vram_gb"] if vram_gb is None else vram_gb
    return (per_card or 0) * (L.get("cards") or 1) * (L.get("machines") or 1)


def below_kv_floor(recipe, launch=None, vram_gb=None):
    """True for a recipe on a setup with more than KV_FLOOR_ABOVE_VRAM_GB that holds less than KV_FLOOR_TOKENS of KV."""
    L = launch or render(recipe)
    return setup_vram_gb(recipe, L, vram_gb) > KV_FLOOR_ABOVE_VRAM_GB and kv_tokens(recipe, L) < KV_FLOOR_TOKENS


def kv_floor_exempt():
    """Recipe paths (relative to recipes/) accepted before the KV floor, kept as evidence and out of the plugin."""
    f = RECIPES.parent / KV_FLOOR_EXEMPT
    return set(json.loads(f.read_text())["recipes"]) if f.exists() else set()


# ----------------------------------------------------------------------------- the server under test
def launch_hash(launch):
    return hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_execution(launch):
    """The typed container contract. Legacy manual prepare commands remain outside the plugin export."""
    assert isinstance(launch.get("env", {}), dict) and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and isinstance(v, str) and "\0" not in v for k, v in launch.get("env", {}).items()), "invalid environment"
    assert isinstance(launch.get("args", []), list) and all(isinstance(x, str) and "\0" not in x for x in launch.get("args", [])), "invalid arguments"
    resources = launch.get("resources", {})
    assert isinstance(resources, dict) and set(resources) <= {"memoryBytes", "memorySwapBytes", "memlockUnlimited", "ipcLock"}, "unknown resource"
    for key in ("memoryBytes", "memorySwapBytes"):
        if key in resources:
            assert type(resources[key]) is int and 0 < resources[key] <= 2**53 - 1, f"invalid {key}"
    if "memorySwapBytes" in resources:
        assert "memoryBytes" in resources and resources["memorySwapBytes"] >= resources["memoryBytes"], "swap total must include memory"
    for key in ("memlockUnlimited", "ipcLock"):
        if key in resources:
            assert type(resources[key]) is bool, f"invalid {key}"
    p = launch.get("prepare")
    if p is None:
        return
    assert isinstance(p, dict) and set(p) == {"at", "args", "verifyArgs", "gpu", "sizeGb"}, "prepare requires at, args, verifyArgs, gpu, sizeGb"
    assert re.fullmatch(r"/(models|nvx|prepared)(/[A-Za-z0-9_.-]+)*", p["at"]) and ".." not in p["at"], "invalid prepared mount"
    for key in ("args", "verifyArgs"):
        assert isinstance(p[key], list) and p[key] and all(isinstance(x, str) and "\0" not in x for x in p[key]), f"invalid prepare.{key}"
    assert type(p["gpu"]) is bool and type(p["sizeGb"]) in (int, float) and 0 < p["sizeGb"] < 100000, "invalid preparation budget"
    weights = launch["weights"] if isinstance(launch["weights"], list) else [launch["weights"]]
    for w in weights:
        assert w["at"] != p["at"] and not p["at"].startswith(w["at"].rstrip("/") + "/"), "input mount masks prepared output"


COMPANION = {"hw", "cards", "image", "entrypoint", "environment", "arguments", "resources", "shm"}
SHARED_DIR = "/run/local-ai/shared"


def check_companion(c):
    """`companion`: a second container on a card of another kind in the same machine, started before the engine (e.g. a
    B70 expert server next to an RTX 3090 engine). It gets the engine's read-only weights and pack mounts and a tmpfs dir
    shared with the engine at SHARED_DIR; the recipe never names host paths. No plugin runs these yet: the exporter keeps
    launches with a companion out of the plugin catalog (candidates)."""
    assert isinstance(c, dict) and set(c) <= COMPANION and {"hw", "cards", "image"} <= set(c), f"companion keys {sorted(c)}"
    card(c["hw"])  # a known card id
    assert type(c["cards"]) is int and 1 <= c["cards"] <= 8, "companion.cards"
    assert isinstance(c["image"], str) and "@sha256:" in c["image"], "companion image must be pinned by digest"
    sub = {"env": c.get("environment", {}), "args": c.get("arguments", []), "resources": c.get("resources", {})}
    if c.get("entrypoint") is not None:
        assert isinstance(c["entrypoint"], str) and "\0" not in c["entrypoint"], "companion.entrypoint"
    if c.get("shm") is not None:
        assert isinstance(c["shm"], str) and re.fullmatch(r"\d+[kmg]", c["shm"]), "companion.shm"
    check_execution(sub)
    for v in list(sub["env"].values()) + sub["args"]:
        if v.startswith("/") and not re.match(r"/(models|nvx|prepared|run/local-ai/shared)(/|$)", v):
            raise AssertionError(f"companion names a host-looking path {v!r}; use the shared dir {SHARED_DIR}")


def call(endpoint, path, body=None, timeout=3600):  # a 128k prompt on a small card takes a while; never cut an answer short
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LAB_API_KEY"):  # an owner's server behind a keyed gateway
        headers["Authorization"] = f"Bearer {os.environ['LAB_API_KEY']}"
    req = urllib.request.Request(endpoint + path, data=json.dumps(body).encode() if body else None, headers=headers)
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.monotonic() - t


def count(endpoint, text):
    """Tokens in text, from the server's tokenizer when it has one (TabbyAPI: /v1/token/encode; vLLM: /tokenize;
    SGLang: /v1/tokenize; llama-server: /tokenize with `content`, which answers `prompt` with an empty list)."""
    for path, body in (("/v1/token/encode", {"text": text}), ("/tokenize", {"prompt": text}), ("/v1/tokenize", {"prompt": text}),
                       ("/tokenize", {"content": text})):
        try:
            out, _ = call(endpoint, path, body, timeout=120)
            n = int(out.get("length") or out.get("count") or len(out.get("tokens") or []))
        except Exception:
            continue
        if n or not text:  # no tokens for a non-empty text is a request the server did not understand
            return n, "tokenizer"
    return len(text) // 4, "estimate"


def chat(endpoint, model, messages, **kw):
    out, secs = call(endpoint, "/v1/chat/completions", {"model": model, "messages": messages, "temperature": 0.6, **kw})
    c, u = out["choices"][0], dict(out.get("usage") or {})
    if (out.get("timings") or {}).get("peak_memory"):  # mlx-vlm: the allocator's peak for this request, decimal GB
        u["peak_memory_gb"] = out["timings"]["peak_memory"]
    if not u.get("prompt_tokens"):  # TabbyAPI called directly leaves usage empty; count what went in and came out
        u["prompt_tokens"], u["counted"] = count(endpoint, "\n".join(m.get("content") or "" for m in messages))
    if not u.get("completion_tokens"):
        m = c.get("message") or {}
        u["completion_tokens"], u["counted"] = count(endpoint, (m.get("reasoning_content") or m.get("reasoning") or "") + (m.get("content") or ""))
    return c, u, secs


def stream_rate(endpoint, model, prompt, window=30.0):
    """Tokens per second over the first `window` seconds after the first token, from a streamed answer."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "stream": True}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LAB_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LAB_API_KEY']}"
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    text, first, at_window, t0 = [], None, None, time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError):
                continue
            piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
            if not piece:
                continue
            now = time.monotonic()
            first = first or now
            text.append(piece)
            if at_window is None and now - first >= window:
                at_window = ("".join(text), now - first)
    whole = "".join(text)
    elapsed = time.monotonic() - (first or t0)
    sample, span = at_window or (whole, elapsed)
    n, _ = count(endpoint, sample)
    total, _ = count(endpoint, whole)
    return (n / span if span else 0), total, time.monotonic() - t0


def served_matches_launch(served, launch):
    """Verify that an endpoint's served model id is consistent with the launch's pinned weights."""
    if not served or not isinstance(served, str):
        return False
    s = served.lower()
    expected = set()
    args = launch.get("args") or []
    for i, a in enumerate(args[:-1]):
        if a in ("--served-model-name", "--model-alias", "-m", "--model"):
            expected.add(str(args[i + 1]).lower())
    for w in launch.get("weights") or []:
        repo = (w.get("repo") or "").split("/")[-1].lower()
        rev = (w.get("revision") or "")[:8].lower()
        if repo:
            expected.add(repo)
        if rev:
            expected.add(rev)
        if w.get("at"):
            expected.add(str(w["at"]).lower())
    if not expected:
        return True
    return any(tok in s or s in tok for tok in expected if tok)


def gates(endpoint, launch):
    """Run every gate; returns (passed, proof, evidence)."""
    ev, ok = {}, {}
    models, _ = call(endpoint, "/v1/models", timeout=30)
    served = models["data"][0]["id"]
    ok["load"] = served_matches_launch(served, launch)
    # chat: an answer that ends by itself
    c, u, _ = chat(endpoint, served, [{"role": "user", "content": "Name three primary colors, comma separated."}])
    ok["chat"] = bool((c["message"].get("content") or "").strip()) and c.get("finish_reason") == "stop"
    ev["chat"] = {"content": c["message"].get("content"), "finish": c.get("finish_reason")}
    # reasoning: thinking comes back separately, and the answer is right
    c, u, _ = chat(endpoint, served, [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}])
    thinking = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    content = c["message"].get("content") or ""
    ok["reasoning"] = bool(thinking.strip()) and "391" in content and "<think>" not in content
    ev["reasoning"] = {"thinking_chars": len(thinking), "content": content[-200:]}
    # tools: a call with the right arguments, then the result used in the reply
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    c, u, _ = chat(endpoint, served, msgs, tools=[tool])
    calls = c["message"].get("tool_calls") or []
    args = {}
    try:
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (ValueError, KeyError):
        pass
    first = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower()
    reply, second = "", {}
    if first:
        msgs += [{"role": "assistant", "content": c["message"].get("content") or "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0].get("id", "0"), "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "overcast"})}]
        c2, _, _ = chat(endpoint, served, msgs, tools=[tool])
        reply = c2["message"].get("content") or ""
        second = {"finish": c2.get("finish_reason"), "calls_again": len(c2["message"].get("tool_calls") or []),
                  "thinking_chars": len(c2["message"].get("reasoning_content") or c2["message"].get("reasoning") or "")}
    ok["tools"] = first and "17" in reply
    ev["tools"] = {"call": calls[:1], "reply": reply[-200:], **second}
    # context: a code planted near the end of a prompt that fills ~85% of the window
    target = int(launch["ctx"] * 0.85)
    filler = " ".join(f"Line {i}: the archive notes that shipment {i * 7 % 997} left dock {i % 13} on schedule." for i in range(target // 24))  # ~23.6 tokens a line (measured: 32,834 tokens from 1,473 lines)
    prompt = filler + " The access code for the vault is 58213. " + "Anything else is routine."
    dropped = []  # seconds each dropped attempt ran before its connection dropped
    for attempt in range(3):  # a rented host's port mapping can silently drop a connection that stays idle for
        started = time.monotonic()
        try:                  # minutes while the prompt is prefilled; the retried prompt answers from the server's
            c, u, secs = chat(endpoint, served, [{"role": "user", "content": prompt + "\n\nWhat is the access code for the vault? Reply with only the code."}])
            break             # prefix cache, so the retry costs seconds, and the gate is the same check
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                raise         # a timeout is the card being slow, not a dropped connection: fail the gate, never retry
            dropped.append(round(time.monotonic() - started, 1))
            if attempt == 2:
                raise
            log("context gate: the connection dropped before the answer; retrying")
            time.sleep(5)
    else:  # unreachable: the last attempt raises
        raise SystemExit("context gate: the connection dropped three times")
    got = u.get("prompt_tokens") or 0
    ok["context"] = "58213" in (c["message"].get("content") or "") and got >= launch["ctx"] * 0.6
    ev["context"] = {"prompt_tokens": got, "seconds": round(secs, 1), "content": (c["message"].get("content") or "")[-80:]}
    if dropped:  # the number came from a retry: say which attempt answered and how long the dropped attempt ran
        ev["context"]["retried"] = attempt
        ev["context"]["dropped_s"] = dropped
    if u.get("peak_memory_gb"):
        ev["context"]["peak_gib"] = round(u["peak_memory_gb"] * 1e9 / 2**30, 2)
    # speed: decode at concurrency 1, measured over the first 30 s of a streamed answer; the answer still runs
    # to its natural end (never capped), and its full length is kept as evidence
    tps, total, secs = stream_rate(endpoint, served, "Write a detailed 600-word story about a lighthouse keeper.")
    ok["speed"] = tps >= MIN_TPS
    ev["speed"] = {"window_s": 30, "tokens_total": total, "seconds_total": round(secs, 1)}
    # a retried answer comes from the server's prefix cache, so its seconds is not a prefill time: leave the
    # number out rather than publish a cached rate (the run can be repeated for a real prefill measurement)
    prefill = None if dropped else (round(got / ev["context"]["seconds"]) if got and ev["context"]["seconds"] else None)
    proof = {"gates": " ".join(g for g in GATES if ok.get(g)), "tps": round(tps, 1), "prefill": prefill, "served": served}
    return all(ok.get(g) for g in GATES), proof, {"ok": ok, **ev}


# ----------------------------------------------------------------------------- where it runs
def save_logs(handle, recipe):
    """The rented container's last 400 log lines, kept with the run evidence."""
    import subprocess
    try:
        out = subprocess.run(["vastai", "logs", str(handle["id"]), "--tail", "400"], capture_output=True, text=True, timeout=90).stdout
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{recipe['card']}.{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}.container.log").write_text(out)
    except Exception as e:
        log(f"could not save the container log: {e}")


def lab_spec(recipe, launch, args):
    """What a rented host runs: the image, provisioning (weights, config file) and then the engine's argv."""
    if launch.get("prepare") or launch.get("resources"):
        raise SystemExit("run this typed model using the SDK steps first, then validate with --on endpoint")
    import rent as vr

    class LabSpec(vr.Spec):
        def __init__(self):
            self.recipe, self.image, self.entrypoint = {"id": "lab"}, launch["image"], launch["entrypoint"]
            self.arguments, self.port, self.disk = list(launch["args"]), launch["port"], args.disk
            self.name = (getattr(args, "label", None) or f"local-ai-lab-{recipe['card']}")[:60]
            tok = Path.home() / ".cache" / "huggingface" / "token"
            self.env = {**(launch.get("env") or {}), **({"HF_TOKEN": tok.read_text().strip()} if tok.exists() else {})}
            self.vram_gb = card(recipe["card"])["vram_gb"] or 0
            self.gpu_override, self.hardware_id = args.proxy_gpu or args.gpu, recipe["card"]
            if args.proxy_gpu:  # a twin card with more memory: search by its own size
                self.vram_gb = args.proxy_vram
            self.provision = []
            self.online = True
            if launch.get("config"):
                text = re.sub(r"^(\s*host:\s*)127\.0\.0\.1", r"\g<1>0.0.0.0", launch["config"]["text"], flags=re.MULTILINE)
                self.provision.append(("asset", launch["config"]["at"], text))
            for w in (launch["weights"] if isinstance(launch["weights"], list) else [launch["weights"]]):
                if w.get("layout", "dir") != "dir":
                    raise SystemExit(f"weights layout {w.get('layout')} cannot be provisioned on a rented host yet")
                self.provision.append(("weights", w["at"], (w["repo"], w["revision"], w.get("files"))))

        def onstart_script(self):  # HF_HUB_OFFLINE=1 in a recipe is for the engine; the download before it must reach the Hub
            return re.sub(r"(?<![\w/.-])([\w/.-]*python3?) -c ", r"env HF_HUB_OFFLINE=0 \1 -c ", vr.Spec.onstart_script(self))
    return LabSpec()


def rented(recipe, launch, args):
    import rent as vr  # Vast and RunPod: offers, create, poll, destroy
    spec = lab_spec(recipe, launch, args)
    needs = launch.get("needs") or {}
    # a host with the RAM the recipe states plus room for the OS and the container runtime, and an NVMe-class disk when it reads from one
    ns = argparse.Namespace(vast_min_inet=args.min_inet, vast_min_cuda=args.min_cuda or launch.get("min_cuda") or 12.9, vast_max_price=args.max_price, cloud="COMMUNITY", disk=max(args.disk, int((needs.get("disk_gb") or 0) + 10)), vast_any_host=getattr(args, "any_host", False),
                            vast_min_ram=getattr(args, "min_ram", None) or (int(needs["host_ram_gb"] * 1.1 + 8) if needs.get("host_ram_gb") else 0),
                            vast_min_disk_bw=getattr(args, "min_disk_bw", None) or (1500 if needs.get("fast_storage") == "nvme" else 0),
                            vast_min_pcie=getattr(args, "min_pcie", 0) or 0, vast_max_inet_cost=getattr(args, "max_inet_cost", 0.02))
    provider = vr.PROVIDERS[args.on](ns)
    if getattr(args, "machine", None):  # compare.py: the machine the other engine just ran on
        every = provider.offers
        provider.offers = lambda spec, exclude: [o for o in every(spec, set()) if o.get("machine_id") == args.machine]
    exclude = set(getattr(args, "exclude_machine", None) or [])  # hosts known not to start the container
    for attempt in range(1, 3):
        handle = provider.create(spec, exclude)
        handle["port"] = spec.port
        t0 = time.monotonic()
        log(f"{args.on} {handle['id']}: {handle.get('gpu')} at {handle.get('cost')}/h")
        try:
            while True:
                ep = handle.get("endpoint")
                if ep and vr.http_probe(f"{ep}/v1/models") == "ok":
                    break
                started, note = provider.poll(handle)
                if not started and time.monotonic() - t0 > 900:
                    exclude.update({handle.get("offer"), handle.get("machine")})
                    raise vr.StartStalled(note)
                if time.monotonic() - t0 > getattr(args, "ready_timeout", 5400):  # pull, download and load: about an hour for 70 GB at 300 Mbps
                    raise SystemExit(f"not ready within {getattr(args, 'ready_timeout', 5400)}s")
                time.sleep(10)
            log(f"ready after {int(time.monotonic() - t0)}s at {handle['endpoint']}")
            return handle["endpoint"], {"on": args.on, "gpu": handle.get("gpu"), "host": handle.get("host"), "cost_h": handle.get("cost"),
                                        "handle": handle, "provider": provider}, \
                lambda: (save_logs(handle, recipe), provider.destroy(handle))
        except vr.StartStalled as stall:
            save_logs(handle, recipe)
            provider.destroy(handle)
            log(f"attempt {attempt}: {stall}; trying another host")
        except BaseException:
            save_logs(handle, recipe)  # any other failure: keep the container's log, never leave the box running
            provider.destroy(handle)
            raise
    raise SystemExit("no host started the container")


# ----------------------------------------------------------------------------- commands
def cmd_try(args):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", args.weights):
        raise SystemExit("weights must be <repo>@<40-hex revision>")
    card(args.card)
    p = profile(args.engine)
    sets = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in p["defaults"]:
            raise SystemExit(f"{k} is not a setting of {p['id']}: {sorted(p['defaults'])}")
        v = int(v) if v.isdigit() else {"true": True, "false": False}.get(v, v)
        if v != p["defaults"][k]:
            sets[k] = v
    recipe = {"model": args.model, "weights": args.weights, "engine": f"{p['id']}@{pin(p)[:12]}",
              "set": sets, "card": args.card}
    launch = render(recipe)
    if below_kv_floor(recipe, launch):
        raise SystemExit(f"{kv_tokens(recipe, launch)} KV tokens on {setup_vram_gb(recipe, launch):g} GB: a setup over "
                         f"{KV_FLOOR_ABOVE_VRAM_GB} GB needs at least {KV_FLOOR_TOKENS} (raise ctx or seqs)")
    if args.dry_run:
        print(json.dumps({"recipe": recipe, "launch": launch}, indent=2))
        return 0
    stop = lambda: None
    if launch.get("kind") == "host" and args.on != "endpoint":
        raise SystemExit(f"{p['id']} runs on the owner's machine, not a rented container: start it and use --on endpoint")
    if args.on == "endpoint":
        endpoint, where = args.endpoint.rstrip("/"), {"on": "owner", "gpu": args.gpu}
    else:
        endpoint, where, stop = rented(recipe, launch, args)
    try:
        passed, proof, evidence = gates(endpoint, launch)
    finally:
        stop()
    at = dt.datetime.now(dt.timezone.utc)
    proof = {"at": at.strftime("%Y-%m-%d"), "on": where["on"], "gpu": where.get("gpu"), "launch_sha256": launch_hash(launch), **proof}
    if args.proxy_gpu and args.on == "endpoint":
        # an owner's larger machine standing in for this card: nothing caps its memory, so the proof states the
        # measured peak against the card's GPU budget, and a run over budget writes no recipe
        peak = evidence.get("context", {}).get("peak_gib")
        sysctl = launch.get("sysctl") or {}
        budget = sysctl["iogpu.wired_limit_mb"] / 1024 if "iogpu.wired_limit_mb" in sysctl else card(args.card)["vram_gb"]
        proof["proxy"] = f"{args.proxy_gpu}, not memory-capped; context-gate peak {peak} GiB of the card's {budget:g} GiB GPU budget"
        if peak is None or peak > budget:
            passed = False
            log(f"proxy run: peak {peak} GiB does not fit the card's {budget:g} GiB")
    elif args.proxy_gpu:
        proof["proxy"] = f"{args.proxy_gpu}, memory capped to the card"
    slug = f"{args.model}.{p['id']}.{launch['ctx'] // 1024}k"
    RUNS.mkdir(parents=True, exist_ok=True)
    run = {"recipe": recipe, "where": {k: v for k, v in where.items() if k not in ("handle", "provider")}, "passed": passed, "proof": proof, "evidence": evidence, "launch_config_sha256": (launch["config"] or {}).get("sha256")}
    text = json.dumps(run, indent=1, ensure_ascii=False)
    run_path = RUNS / f"{args.card}.{slug}.{at.strftime('%Y%m%dT%H%M%S')}.json"
    run_path.write_text(text + "\n")
    proof["log"] = "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]
    log(f"gates: {evidence['ok']}  decode {proof['tps']} tok/s, prefill {proof['prefill']} tok/s; evidence {run_path.relative_to(ROOT)}")
    if not passed:
        log("FAILED: no recipe written")
        return 1
    out = recipe_path(recipe, launch)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else None
    proofs = [proof] + ([q for q in old["proof"] if old.get("weights") == recipe["weights"] and old.get("engine") == recipe["engine"] and old.get("set") == sets][:2] if old else [])
    out.write_text(json.dumps({**recipe, "proof": proofs}, separators=(",", ":")) + "\n")
    log(f"PASSED: wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    return 0


def cmd_convert(args):
    files = [f for f in sorted(RECIPES.glob(f"*/{args.card}/*.json")) if json.loads(f.read_text())["proof"][0].get("legacy")
             and (not args.recipe or f.stem == args.recipe)]
    if not files:
        raise SystemExit(f"{args.card} has no legacy recipe" + (f" {args.recipe}" if args.recipe else ""))
    f = files[0]
    recipe = json.loads(f.read_text())
    launch = render(recipe)
    stop = lambda: None
    if args.on == "endpoint":
        endpoint, where = args.endpoint.rstrip("/"), {"on": "owner", "gpu": args.gpu}
    else:
        endpoint, where, stop = rented(recipe, launch, args)
    try:
        passed, proof, evidence = gates(endpoint, launch)
    finally:
        stop()
    at = dt.datetime.now(dt.timezone.utc)
    proof = {"at": at.strftime("%Y-%m-%d"), "on": where["on"], "gpu": where.get("gpu"), "launch_sha256": launch_hash(launch), **proof}
    RUNS.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"recipe": recipe, "where": {k: v for k, v in where.items() if k not in ("handle", "provider")}, "passed": passed, "proof": proof, "evidence": evidence}, indent=1, ensure_ascii=False)
    (RUNS / f"{args.card}.{f.stem}.{at.strftime('%Y%m%dT%H%M%S')}.json").write_text(text + "\n")
    proof["log"] = "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]
    log(f"gates: {evidence['ok']}  decode {proof['tps']} tok/s, prefill {proof['prefill']} tok/s")
    if not passed:
        log("FAILED: the legacy recipe stays legacy")
        return 1
    recipe["proof"] = [proof]
    f.write_text(json.dumps(recipe, separators=(",", ":")) + "\n")
    log(f"PASSED: {f.relative_to(ROOT)} is now a full recipe")
    return 0


def cmd_proxy(args):
    """A card nobody rents takes its sibling's passing lab recipes, each proof marked proxy (same chip family, same memory)."""
    made = 0
    for f in sorted(RECIPES.glob(f"*/{args.sibling}/*.json")):
        r = json.loads(f.read_text())
        p = r["proof"][0]
        if p.get("legacy") or p.get("proxy") or p.get("reported"):  # only our own lab runs carry over
            continue
        r = {**r, "card": args.card, "proof": [{**p, "proxy": args.sibling}]}
        out = recipe_path(r, render(r))
        if out.exists() and not json.loads(out.read_text())["proof"][0].get("proxy"):
            continue  # a real run on the card wins
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, separators=(",", ":")) + "\n")
        made += 1
    log(f"{args.card}: {made} recipe(s) from {args.sibling}, marked proxy")
    return 0


def cmd_render(args):
    print(json.dumps(render(json.loads(Path(args.recipe).read_text())), indent=2))
    return 0


NEEDS = {"host_ram_gb": (int, float), "disk_gb": (int, float), "fast_storage": str, "cpus": int}


def check_needs(n):
    """`needs`: what the host must have besides the card, for launches that keep weights in system RAM or read them
    from disk while serving. host_ram_gb = free system RAM the server takes (measured peak), disk_gb = space for the
    weights, fast_storage = "nvme" when the weights directory must be on a local NVMe SSD (random reads at runtime), cpus = the
    CPU threads the engine pins work to (plugin >= 6.11 says the machine is too small below that)."""
    assert isinstance(n, dict) and set(n) <= set(NEEDS) and {"host_ram_gb", "disk_gb"} <= set(n), f"needs keys {sorted(n)}"
    for k, v in n.items():
        assert isinstance(v, NEEDS[k]) and not isinstance(v, bool), f"needs.{k} is {type(v).__name__}"
    assert n["host_ram_gb"] > 0 and n["disk_gb"] > 0, "needs sizes must be positive"
    assert n.get("fast_storage", "nvme") == "nvme", "needs.fast_storage is nvme or absent"
    assert n.get("cpus", 1) > 0, "needs.cpus counts CPU threads the engine pins work to"


def cmd_check(_):
    bad = []
    exempt = kv_floor_exempt()
    gone = sorted(x for x in exempt if not (RECIPES / x).exists())
    if gone:
        bad.append(f"registry/{KV_FLOOR_EXEMPT}: lists recipes that do not exist: {', '.join(gone)}")
    for f in sorted(RECIPES.rglob("*.json")):
        r = json.loads(f.read_text())
        name = f.relative_to(ROOT)
        try:
            assert set(r) == {"model", "weights", "engine", "set", "card", "proof"}, f"fields {sorted(r)}"
            assert r["weights"] == "baked-into-image" or re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", r["weights"]), "weights not pinned"
            c = card(r["card"])
            assert f.parent.name == r["card"] and f.parent.parent.name == c["vendor"], "path is not recipes/<vendor>/<card>/"
            launch = render(r)
            assert f == recipe_path(r, launch), f"file name should be {recipe_path(r, launch).name}"
            if launch.get("kind") == "host":  # the SDK's host steps start one program on one machine
                assert type(launch["cards"]) is int and launch["cards"] >= 1, "host cards must be a whole number of cards"
                assert launch.get("machines", 1) == 1, "multi-machine host launches are not supported by the current SDK; use a container launch"
            if "needs" in launch:
                check_needs(launch["needs"])
            if "companion" in launch:
                check_companion(launch["companion"])
            p = r["proof"][0]
            if launch.get("resources") or "verifyArgs" in (launch.get("prepare") or {}):
                check_execution(launch)
                # a reported proof is its publisher's run, not an acceptance: there is no lab hash to bind
                if not p.get("withdrawn") and not p.get("reported"):
                    assert p.get("launch_sha256") == launch_hash(launch), "typed launch changed since acceptance; rerun the recipe"
            if "withdrawn" in p:
                assert p["withdrawn"] is True and p.get("src") and p.get("reason"), "withdrawal needs source and reason"
            need = set() if p.get("reported") or p.get("withdrawn") else {"load", "chat"} if p.get("legacy") else set(GATES)
            assert not r["proof"][0].get("reported") or r["proof"][0].get("src"), "a reported proof names its source"
            assert need <= set(r["proof"][0]["gates"].split()), "latest proof lacks a gate"
            assert f.stat().st_size <= MAX_RECIPE_BYTES, f"{f.stat().st_size} bytes"
            rel = str(f.relative_to(RECIPES))
            low = not p.get("withdrawn") and below_kv_floor(r, launch)
            if rel in exempt:
                assert low, f"listed in {KV_FLOOR_EXEMPT} but not below the KV floor; remove it from the list"
            else:
                assert not low, (f"{kv_tokens(r, launch)} KV tokens on {setup_vram_gb(r, launch):g} GB: below the "
                                 f"{KV_FLOOR_TOKENS}-token floor for setups over {KV_FLOOR_ABOVE_VRAM_GB} GB")
        except (AssertionError, KeyError, ValueError, SystemExit, FileNotFoundError) as e:
            bad.append(f"{name}: {e}")
    print("\n".join(bad) or f"recipes ok: {len(list(RECIPES.rglob('*.json')))}")
    return 1 if bad else 0


def main():
    import signal
    def stop(*_):  # background jobs ignore SIGINT; SIGTERM must unwind so a rented box is always destroyed
        raise SystemExit("terminated")
    signal.signal(signal.SIGTERM, stop)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("try")
    t.add_argument("weights")
    t.add_argument("--model", required=True)
    t.add_argument("--engine", required=True)
    t.add_argument("--card", required=True)
    t.add_argument("--set", action="append", default=[])
    t.add_argument("--on", default="vast", choices=["vast", "runpod", "endpoint"])
    t.add_argument("--endpoint")
    t.add_argument("--gpu", help="provider GPU name (default from the card) or, with --on endpoint, the GPU the owner ran it on")
    t.add_argument("--proxy-gpu", help="run on this twin card, memory capped to the card's")
    t.add_argument("--proxy-vram", type=int, default=16)
    t.add_argument("--min-cuda", type=float)
    t.add_argument("--min-inet", type=int, default=500, help="vast: minimum host downlink, Mbps")
    t.add_argument("--max-price", type=float, default=1.5)
    t.add_argument("--disk", type=int, default=60)
    t.add_argument("--any-host", action="store_true", help="vast: also unverified hosts (the gates are the same; for cards verified hosts never offer)")
    t.add_argument("--min-pcie", type=float, default=0, help="vast: minimum host-to-device GB/s (offload recipes: experts cross PCIe every token)")
    t.add_argument("--max-inet-cost", type=float, default=0.02, help="vast: maximum $ per GB downloaded (0: no limit)")
    t.add_argument("--ready-timeout", type=int, default=5400, help="seconds from rent to a serving engine")
    t.add_argument("--min-ram", type=int, help="vast: minimum host RAM GB (default needs.host_ram_gb x 1.1 + 8; the recipe's needs stay as measured)")
    t.add_argument("--min-disk-bw", type=int, help="vast: minimum disk MB/s (default 1500 when the recipe reads from NVMe)")
    t.add_argument("--exclude-machine", type=int, action="append", help="vast: never rent this machine id (e.g. one whose runtime cannot start the container)")
    t.add_argument("--label", help="rented instance label (default local-ai-lab-<card>), e.g. to tell a campaign's boxes apart")
    t.add_argument("--dry-run", action="store_true")
    cv = sub.add_parser("convert")
    cv.add_argument("card")
    cv.add_argument("--recipe", help="which legacy recipe of the card (its file name without .json); default the first")
    cv.add_argument("--on", default="vast", choices=["vast", "runpod", "endpoint"])
    cv.add_argument("--endpoint")
    cv.add_argument("--gpu")
    cv.add_argument("--proxy-gpu")
    cv.add_argument("--proxy-vram", type=int, default=16)
    cv.add_argument("--min-cuda", type=float)
    cv.add_argument("--min-inet", type=int, default=500, help="vast: minimum host downlink, Mbps")
    cv.add_argument("--max-price", type=float, default=2.0)
    cv.add_argument("--disk", type=int, default=80)
    px = sub.add_parser("proxy")
    px.add_argument("card")
    px.add_argument("--from", dest="sibling", required=True)
    sub.add_parser("render").add_argument("recipe")
    sub.add_parser("check")
    a = ap.parse_args()
    return {"try": cmd_try, "convert": cmd_convert, "proxy": cmd_proxy, "render": cmd_render, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
