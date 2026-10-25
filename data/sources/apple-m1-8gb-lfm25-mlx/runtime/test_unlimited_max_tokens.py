#!/usr/bin/env python3
"""CPU-only regression for the task-local mlx_lm 0.31.3 server patch (unlimited max_tokens = -1).

Never imports mlx or mlx_lm (no Metal/GPU initialisation). Parses each copy's server.py with `ast` and executes the
real source of: the request's max_tokens resolution, APIHandler._validate, APIHandler.validate_model_parameters,
ResponseGenerator._is_batchable, ResponseGenerator._serve_single, APIHandler.handle_completion, the argparse block of
main(), and the copy's own LRUPromptCache (models/cache.py), against small stubs.

  test_unlimited_max_tokens.py PRISTINE_SERVER_PY PATCHED_SERVER_PY
Exit 0 only if every expectation holds for both copies.
"""
import argparse, ast, copy, json, logging, queue, sys, textwrap, types
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from types import SimpleNamespace as NS

# the server flags of the recipe's launch, after the launcher script (tools/test_runner.py checks the launch JSON says the same)
LAUNCH_FLAGS = ["--model", "LiquidAI/LFM2.5-2.6B-MLX-4bit", "--max-tokens", "-1", "--max-kv-size", "32768",
                "--max-prompt-tokens", "24576", "--prompt-cache-size", "1", "--host", "127.0.0.1", "--port", "8080"]
KV, LIMIT = 32768, 24576

def method(tree, cls, name):
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and c.name == cls:
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise KeyError(f"{cls}.{name}")

def owner(tree, name):
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and any(isinstance(f, ast.FunctionDef) and f.name == name for f in c.body):
            return c.name
    raise KeyError(name)

def compile_fn(src, fn):
    ns = {}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), {"List": list, "Dict": dict}, ns)
    return ns[fn.name]

def resolution_stmts(src, tree):
    """The two statements that set self.max_tokens from the body or the CLI default."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for i, st in enumerate(node.body):
                if (isinstance(st, ast.Assign) and ast.get_source_segment(src, st).startswith("self.max_tokens = self.body.get(\"max_completion_tokens\"")):
                    code = textwrap.dedent(ast.get_source_segment(src, st)) + "\n" + textwrap.dedent(ast.get_source_segment(src, node.body[i + 1]))
                    return compile(code, "<resolution>", "exec")
    raise KeyError("max_tokens resolution")

def parser_from_main(src, tree, path):
    main = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "main")
    keep = []
    for st in main.body:
        if "parse_args" in ast.get_source_segment(src, st):
            break
        keep.append(textwrap.dedent(ast.get_source_segment(src, st)))
    block = "\n".join(keep)
    helpers = [f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name != "main" and f.name in block]
    ns = {"argparse": argparse, "json": json}
    # lifted functions keep their annotations unevaluated (PEP 563), as Python 3.14 does by default, so a name used only in
    # an annotation (Optional, nn.Module) need not exist here on Pythons that evaluate annotations at definition (3.12)
    lift = lambda code: exec("from __future__ import annotations\n" + textwrap.dedent(code), ns)
    for f in helpers:  # module-level helpers the parser block calls, executed from the same source
        lift(ast.get_source_segment(src, f))
    usrc = open(path.rsplit("/", 1)[0] + "/utils.py").read()  # _parse_size is imported from the same copy's utils.py
    for f in ast.parse(usrc).body:
        if isinstance(f, ast.FunctionDef) and f.name in block and f.name not in ns:
            lift(ast.get_source_segment(usrc, f))
    exec(block, ns)
    return ns["parser"]

def load(path):
    src = open(path).read()
    tree = ast.parse(src)
    h, g = owner(tree, "validate_model_parameters"), owner(tree, "_is_batchable")
    return NS(validate=compile_fn(src, method(tree, h, "_validate")),
              vmp=compile_fn(src, method(tree, h, "validate_model_parameters")),
              batchable=compile_fn(src, method(tree, g, "_is_batchable")),
              resolve=resolution_stmts(src, tree), parser=parser_from_main(src, tree, path))

def handler(m, body, cli_max, cli_draft=None):
    h = NS(body=body, response_generator=NS(cli_args=NS(max_tokens=cli_max, num_draft_tokens=3, draft_model=cli_draft, temp=0.0,
                                                        top_p=1.0, top_k=0, min_p=0.0)))
    exec(m.resolve, {}, {"self": h})
    h.requested_draft_model = body.get("draft_model", "default_model")
    h.stream, h.temperature, h.top_p, h.top_k, h.min_p = body.get("stream", False), body.get("temperature", 0.8), 1.0, 0, 0.0
    h.num_draft_tokens, h.repetition_penalty, h.repetition_context_size = 3, 0.0, 20
    h.presence_penalty, h.frequency_penalty, h.logit_bias, h.logprobs, h.top_logprobs, h.seed = 0.0, 0.0, None, False, -1, None
    h.adapter, h.xtc_probability, h.xtc_threshold, h.chat_template_kwargs = None, 0.0, 0.0, None
    h.presence_context_size, h.frequency_context_size, h.requested_model = 20, 20, "default_model"
    h._validate = lambda *a, **k: m.validate(h, *a, **k)
    return h

def bounded_cache_checks(pristine_path, patched_path):
    """The patched _bounded_prompt_cache turns every KVCache of the model's own cache into a RotatingKVCache of the bound
    (keep=4) and keeps other caches; with no bound it returns the model's cache unchanged. The sequential path calls it
    with the CLI bound; the pristine copy has no such function and calls make_prompt_cache(model)."""
    src, psrc = open(patched_path).read(), open(pristine_path).read()
    fn = next((f for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == "_bounded_prompt_cache"), None)
    if fn is None:
        return False, "patched copy has no _bounded_prompt_cache"
    class KVCache: pass
    class ArraysCache: pass
    class RotatingKVCache:
        def __init__(self, max_size, keep=0): self.max_size, self.keep = max_size, keep
    seen = []
    def make_prompt_cache(model, max_kv_size=None):
        seen.append(max_kv_size)
        return [KVCache(), ArraysCache(), KVCache()]  # like LFM2: attention layers grow, conv layers do not
    ns = {"make_prompt_cache": make_prompt_cache, "KVCache": KVCache, "RotatingKVCache": RotatingKVCache}
    exec("from __future__ import annotations\n" + ast.get_source_segment(src, fn), ns)
    b, u = ns["_bounded_prompt_cache"](object(), 24576), ns["_bounded_prompt_cache"](object(), None)
    kinds = [type(c).__name__ for c in b]
    good = (kinds == ["RotatingKVCache", "ArraysCache", "RotatingKVCache"] and all(c.max_size == 24576 and c.keep == 4 for c in b[::2])
            and [type(c).__name__ for c in u] == ["KVCache", "ArraysCache", "KVCache"] and seen == [24576, None])
    flat, pflat = " ".join(src.split()), " ".join(psrc.split())
    good &= "cache = _bounded_prompt_cache(self.model_provider.model, self.cli_args.max_kv_size)" in flat
    good &= "cache = make_prompt_cache(self.model_provider.model)" in pflat and "_bounded_prompt_cache" not in psrc
    return good, f"bound 24576 -> {kinds}; no bound -> unchanged; sequential path uses the CLI bound"

def lift(src, names, ns):
    """Execute the module-level definitions `names` from src into ns (annotations unevaluated)."""
    tree = ast.parse(src)
    for f in tree.body:
        if isinstance(f, (ast.FunctionDef, ast.ClassDef)) and f.name in names:
            exec("from __future__ import annotations\n" + ast.get_source_segment(src, f), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise KeyError(", ".join(missing))
    return ns

def serve_single(path, n_tokens, limit):
    """Run the copy's own ResponseGenerator._serve_single for a prompt of n_tokens with --max-prompt-tokens limit.
    Returns (first item the request queue received, whether generation started, whether the cache was saved)."""
    src = open(path).read()
    tree = ast.parse(src)
    fn = method(tree, owner(tree, "_serve_single"), "_serve_single")
    calls = {"generate": 0, "insert": 0}
    class GenerationContext:
        def __init__(self, **kw): self.__dict__.update(kw); self._should_stop = False
    def stream_generate(**kw):
        calls["generate"] += 1
        yield NS(finish_reason="stop", token=1, text="x", logprobs={1: NS(item=lambda: 0.0)})
    ns = {"GenerationContext": GenerationContext, "stream_generate": stream_generate, "mx": NS(random=NS(seed=lambda s: None)),
          "_make_sampler": lambda a, t: None, "_make_logits_processors": lambda a: [], "make_prompt_cache": lambda *a: [],
          "_bounded_prompt_cache": lambda *a: [], "Response": lambda *a: a, "_format_top_logprobs": lambda *a: ()}
    if "_check_prompt_length" in src:
        lift(src, ["PromptTooLong", "_check_prompt_length"], ns)
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    cache = NS(fetch_nearest_cache=lambda key, prompt: (None, prompt),
               insert_cache=lambda *a: calls.__setitem__("insert", calls["insert"] + 1))
    tok = NS(has_thinking=False, has_tool_calling=False, tool_parser=None)
    me = NS(model_provider=NS(model=object(), tokenizer=tok, draft_model=None, model_key="m"),
            cli_args=NS(max_prompt_tokens=limit, max_kv_size=KV, prefill_step_size=2048),
            _tokenize=lambda t, r, a: (list(range(n_tokens)), None, None, None),
            _make_state_machine=lambda *a, **k: (NS(make_state=lambda: 0, match=lambda s, t: (s, None, None)), []),
            prompt_cache=cache, _log_cache_stats=lambda: None, _is_distributed=False)
    q = queue.Queue()
    ns["_serve_single"](me, (q, NS(), NS(seed=None, max_tokens=-1, num_draft_tokens=3, stop_words=[], top_logprobs=-1)))
    return q.get_nowait(), calls["generate"] > 0, calls["insert"] > 0

def prompt_limit_checks(pristine_path, patched_path):
    """The prompt limit, in each copy's own _serve_single: the patched copy rejects a prompt one token over the limit
    before the generation context (so the client gets an error, not a stream) and before any generation; a prompt at the
    limit, or any prompt with no limit, is served. The pristine copy serves all of them."""
    rows, ok = [], True
    for label, n, limit, exp_p, exp_q in [("one token over the limit", LIMIT + 1, LIMIT, "served", "rejected"),
                                          ("at the limit", LIMIT, LIMIT, "served", "served"),
                                          ("far over, no limit set", 10 * LIMIT, None, "served", "served")]:
        got = []
        for path in (pristine_path, patched_path):
            first, generated, saved = serve_single(path, n, limit)
            if isinstance(first, Exception):
                got.append("rejected" if type(first).__name__ == "PromptTooLong" and not generated and not saved else f"error {first!r}")
            else:
                got.append("served" if type(first).__name__ == "GenerationContext" and generated and saved else "broken")
        good = got == [exp_p, exp_q]
        ok &= good
        rows.append(f"{'PASS' if good else 'FAIL'}  prompt limit: {label:28s} pristine {got[0]}; patched {got[1]}")
    return ok, rows

def status_checks(pristine_path, patched_path):
    """What the client gets when generate() raises before the stream starts, from each copy's own handle_completion: the
    patched copy answers an over-long prompt with 400 and keeps 404 for every other error, as the pristine copy does."""
    rows, ok = [], True
    out = {}
    for name, path in (("pristine", pristine_path), ("patched", patched_path)):
        src = open(path).read()
        tree = ast.parse(src)
        ns = {k: (lambda **kw: NS(**kw)) for k in ("GenerationArguments", "ModelDescription", "SamplingArguments", "LogitsProcessorArguments")}
        ns.update(json=json, logging=logging)
        if "_error_status" in src:
            lift(src, ["PromptTooLong", "_error_status"], ns)
        exec("from __future__ import annotations\n" + textwrap.dedent(ast.get_source_segment(src, method(tree, owner(tree, "handle_completion"), "handle_completion"))), ns)
        too_long = ns.get("PromptTooLong", type("PromptTooLong", (ValueError,), {}))
        for kind, exc in (("over-long prompt", too_long("the prompt is 24577 tokens")), ("other error", ValueError("model not found"))):
            class H:
                stream = False
                def __init__(self): self.sent, self.body = [], b""; self.wfile = NS(write=self.write)
                def __getattr__(self, k): return None
                def write(self, b): self.body += b
                def _set_completion_headers(self, code): self.sent.append(code)
                def end_headers(self): pass
            h = H()
            def generate(*a, **k): raise exc
            h.response_generator = NS(generate=generate)
            ns["handle_completion"](h, NS(tools=None), [])
            out[(name, kind)] = (h.sent, json.loads(h.body)["error"])
    for kind, exp_p, exp_q in (("over-long prompt", [404], [400]), ("other error", [404], [404])):
        (sp, ep), (sq, eq) = out[("pristine", kind)], out[("patched", kind)]
        good = sp == exp_p and sq == exp_q and ep == eq
        ok &= good
        rows.append(f"{'PASS' if good else 'FAIL'}  error status: {kind:27s} pristine {sp}; patched {sq} ({eq!r})")
    return ok, rows

def prompt_cache_checks(pristine_path, patched_path):
    """The prompt cache the server keeps between requests: run() builds LRUPromptCache(cli_args.prompt_cache_size) in both
    copies, and the copy's own LRUPromptCache, at the launch's --prompt-cache-size 1, keeps only the last of three
    distinct near-bound conversations (at the stock default of 10 it keeps all three)."""
    rows, ok = [], True
    for name, path in (("pristine", pristine_path), ("patched", patched_path)):
        src = open(path).read()
        flat = " ".join(src.split())
        good = "prompt_cache = LRUPromptCache(model_provider.cli_args.prompt_cache_size)" in flat
        ok &= good
        rows.append(f"{'PASS' if good else 'FAIL'}  prompt cache: {name:27s} run() sizes it from --prompt-cache-size")
    csrc = open(patched_path.rsplit("/", 1)[0] + "/models/cache.py").read()
    mod = types.ModuleType("lifted_cache")  # dataclass looks its class's module up in sys.modules
    mod.__dict__.update(copy=copy, deque=deque, dataclass=dataclass, Any=Any, List=List, Dict=Dict, Optional=Optional)
    sys.modules["lifted_cache"] = mod
    ns = lift(csrc, ["can_trim_prompt_cache", "trim_prompt_cache", "PromptTrieResult", "PromptTrie", "LRUPromptCache"], mod.__dict__)
    class Layer:  # an LFM2 cache: a RotatingKVCache past its window is not trimmable, an ArraysCache never is
        def __init__(self, nbytes): self.nbytes = nbytes
        def is_trimmable(self): return False
    patched = load(patched_path)
    for label, argv, expect in (("launch flags", LAUNCH_FLAGS, 1), ("stock default", ["--model", "m"], 3)):
        size = patched.parser.parse_args(argv).prompt_cache_size
        c = ns["LRUPromptCache"](size)
        for k in range(3):  # three conversations with no common prefix, each about 23K tokens of KV (16 KiB a token)
            c.insert_cache("m", [1000 * (k + 1) + i for i in range(23000)], [Layer(23000 * 16384)])
        good = len(c) == expect and c.nbytes == expect * 23000 * 16384
        ok &= good
        rows.append(f"{'PASS' if good else 'FAIL'}  prompt cache: {label:27s} --prompt-cache-size {size} keeps {len(c)} of 3 ({c.nbytes / 2**30:.2f} GiB)")
    return ok, rows

def accepts(m, body, cli_max, cli_draft=None):
    h = handler(m, body, cli_max, cli_draft)
    try:
        m.vmp(h)
        return True, h.max_tokens
    except (ValueError, TypeError) as e:
        return False, str(e)[:70]

def main():
    pristine, patched = load(sys.argv[1]), load(sys.argv[2])
    # case: (label, body, cli --max-tokens, cli draft, expected pristine accept, expected patched accept)
    cases = [
        ("lab request, CLI --max-tokens -1", {"temperature": 0.8}, -1, None, False, True),
        ("body max_tokens -1", {"max_tokens": -1}, 512, None, False, True),
        ("body max_completion_tokens -1", {"max_completion_tokens": -1}, 512, None, False, True),
        ("CLI default 512 unchanged", {}, 512, None, True, True),
        ("body finite 100 unchanged", {"max_tokens": 100}, -1, None, True, True),
        ("body 0 unchanged", {"max_tokens": 0}, 512, None, True, True),
        ("body -2 still rejected", {"max_tokens": -2}, 512, None, False, False),
        ("CLI -5 still rejected", {}, -5, None, False, False),
        ("body non-int still rejected", {"max_tokens": "100"}, 512, None, False, False),
        ("-1 with CLI draft model rejected", {}, -1, "some/draft", False, False),
        ("-1 with request draft model rejected", {"draft_model": "x/y"}, -1, None, False, False),
    ]
    ok = True
    for label, body, cli, draft, exp_p, exp_q in cases:
        rp, rq = accepts(pristine, body, cli, draft), accepts(patched, body, cli, draft)
        good = rp[0] == exp_p and rq[0] == exp_q
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {label:40s} pristine={'accept' if rp[0] else 'reject'} ({rp[1]})  patched={'accept' if rq[0] else 'reject'} ({rq[1]})")
    # routing: -1 must leave the batch path (BatchGenerator stops at num_tokens >= max_tokens); finite unchanged
    # with --max-kv-size every request stays on the sequential path, where the bounded cache is made
    for label, mt, seed, kv, exp_p, exp_q in [("route max_tokens -1", -1, None, None, True, False),
                                              ("route finite 512", 512, None, None, True, True),
                                              ("route seed set", 512, 7, None, False, False),
                                              ("route finite 512 with --max-kv-size", 512, None, (KV, None), True, False),
                                              ("route finite 512 with --max-prompt-tokens", 512, None, (None, LIMIT), True, False)]:
        kv, limit = kv or (None, None)
        prov = NS(model_provider=NS(is_batchable=True), cli_args=NS(max_kv_size=kv, max_prompt_tokens=limit))
        a = NS(max_tokens=mt, seed=seed)
        bp, bq = pristine.batchable(prov, a), patched.batchable(prov, a)
        good = bp == exp_p and bq == exp_q
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {label:40s} pristine batch={bp}  patched batch={bq}")
    # CLI: argparse accepts --max-tokens -1 and keeps default 512 in both copies
    for name, m in (("pristine", pristine), ("patched", patched)):
        a, d = m.parser.parse_args(["--max-tokens", "-1"]), m.parser.parse_args([])
        good = a.max_tokens == -1 and d.max_tokens == 512
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  CLI parse {name:31s} --max-tokens -1 -> {a.max_tokens}; default -> {d.max_tokens}")
    # CLI: only the patched copy knows --max-kv-size (an int; unset by default)
    try:
        pristine.parser.parse_args(["--max-kv-size", "24576"]); pk = "accepted"
    except SystemExit:
        pk = "rejected"
    try:
        qa, qd = patched.parser.parse_args(["--max-kv-size", "24576"]), patched.parser.parse_args([])
    except SystemExit:
        qa = qd = NS(max_kv_size="rejected")
    good = pk == "rejected" and qa.max_kv_size == 24576 and qd.max_kv_size is None
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  CLI parse --max-kv-size              pristine {pk}; patched 24576 -> {qa.max_kv_size}; default -> {qd.max_kv_size}")
    # CLI: the launch's own flags; only the patched copy accepts them all
    try:
        pristine.parser.parse_args(LAUNCH_FLAGS); pk = "accepted"
    except SystemExit:
        pk = "rejected"
    try:
        qa = patched.parser.parse_args(LAUNCH_FLAGS)
        got = (qa.max_tokens, qa.max_kv_size, qa.max_prompt_tokens, qa.prompt_cache_size)
    except SystemExit:
        got = "rejected"
    good = pk == "rejected" and got == (-1, KV, LIMIT, 1)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  CLI parse launch flags               pristine {pk}; patched (max_tokens, kv, prompt limit, cache size) {got}")
    # the bound: the patched copy's own _bounded_prompt_cache, run against stand-in cache classes (no mlx import)
    good, detail = bounded_cache_checks(sys.argv[1], sys.argv[2])
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  bounded cache                         {detail}")
    for check in (prompt_limit_checks, status_checks, prompt_cache_checks):
        try:
            good, rows = check(sys.argv[1], sys.argv[2])
        except Exception as e:  # a copy without the code under test fails here, never crashes the run
            good, rows = False, [f"FAIL  {check.__name__}: {type(e).__name__}: {e}"]
        ok &= good
        print("\n".join(rows))
    # dispatch: a request joins an existing batch only through _is_batchable; otherwise the batch drains and the request is
    # requeued; with no batch generator a non-batchable request goes to _serve_single (so -1 never reaches BatchGenerator)
    for name, path in (("pristine", sys.argv[1]), ("patched", sys.argv[2])):
        src = open(path).read()
        gen = ast.get_source_segment(src, method(ast.parse(src), owner(ast.parse(src), "_is_batchable"), "_generate")) or ""
        flat = " ".join(gen.split())
        checks = {
            "join existing batch requires _is_batchable": "batch_generator is not None and current_model == args.model and self._is_batchable(args)" in flat,
            "otherwise drain the batch and requeue": "else: drain_batch = True unprocessed_requests.append((rqueue, request, args))" in flat,
            "no generator: non-batchable -> _serve_single": "if not self._is_batchable(args): self._serve_single((rqueue, request, args))" in flat,
        }
        for label, good in checks.items():
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  dispatch {name:30s} {label}")
    print("REGRESSION", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
