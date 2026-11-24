"""Small OpenAI-compatible chat server for Inkling 2 Mini on the fast mlx-lm path.

Why this exists: on an M4 Pro, mlx_lm.server runs ~20% slower than the library's
stream_generate, and it returns Inkling's raw control tokens. This server calls
stream_generate directly for standalone requests, can batch simultaneous decode,
and splits the reply into `reasoning_content`, `content` and `tool_calls`.

  python inkling_server.py --model DIR [--port 8790] [--prefill-step 512] [--max-context N]

Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not),
POST /v1/token/encode ({"text"} -> {"tokens", "length"}, the model's tokenizer).
Requests that leave sampling fields out get the model's generation_config values.
Text only: the vision tower is not loaded (saves 0.64 GB), so image parts get a 400.
Keeps one prompt prefix in memory. Use --no-prompt-cache to disable it.
"""

import argparse
import atexit
import json
import pathlib
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import closing
from types import SimpleNamespace
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER = re.compile(r"(<\|[a-z0-9_]+\|>)")


def parse_reply(text, final):
    """Split Inkling output into reasoning, content and tool calls.

    While streaming (final=False), hold back a trailing '<|...' that may be the
    start of a marker so it never leaks into the visible text.
    """
    if not final:
        cut = text.rfind("<|")
        if cut != -1 and "|>" not in text[cut:]:
            text = text[:cut]
        elif text.endswith("<"):
            text = text[:-1]
    reasoning, content, tools = [], [], []
    chan, header, ended = None, "", False
    for part in MARKER.split(text):
        if not part:
            continue
        if MARKER.fullmatch(part):
            if part == "<|message_model|>":
                chan, header = "header", ""
            elif part == "<|content_thinking|>":
                chan = "reasoning"
            elif part == "<|content_text|>":
                chan = "content"
            elif part == "<|content_invoke_tool_json|>":
                chan = "tool"
                tools.append({"name": header.strip(), "args": "", "done": False})
            elif part == "<|end_message|>":
                if chan == "tool":
                    tools[-1]["done"] = True
                chan = None
            elif part == "<|content_model_end_sampling|>":
                chan, ended = None, True
            continue
        if chan == "header":
            header += part
        elif chan == "reasoning":
            reasoning.append(part)
        elif chan == "tool":
            tools[-1]["args"] += part
        elif chan == "content" or part.strip():
            content.append(part)
    return "".join(reasoning), "".join(content), tools, ended


def openai_tool_calls(tools):
    out = []
    for t in tools:
        if not t["done"]:
            continue
        try:
            body = json.loads(t["args"])
            name, args = body.get("name", t["name"]), body.get("args", {})
        except json.JSONDecodeError:
            name, args = t["name"], t["args"]
        out.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        })
    return out


def normalize_messages(messages):
    """Adapt OpenAI-style messages to what Inkling's chat template accepts."""
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("image", "image_url", "input_image"):
                    raise ValueError("this server is text-only; image input is not supported")
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append(tc)
            m["tool_calls"] = calls
        if content is None:
            m["content"] = ""
        out.append(m)
    return out


def snapshot_prompt_cache(cache):
    """Copy cache metadata and MLX array handles for this Inkling cache layout.

    MLX array assignment replaces the value of its Python handle. Separate
    handles keep the old buffers alive even when decode updates a ring slot.
    """
    snapshot = []
    for visit in cache:
        kv, conv = visit
        saved_kv = copy(kv)
        if isinstance(kv.keys, tuple):
            saved_kv.keys = tuple(map(copy, kv.keys))
            saved_kv.values = tuple(map(copy, kv.values))
        else:
            saved_kv.keys = copy(kv.keys)
            saved_kv.values = copy(kv.values)
        saved_conv = copy(conv)
        saved_conv.cache = [copy(state) for state in conv.cache]
        # The unbatched ArraysCache has no lengths or left padding.
        snapshot.append(type(visit)(saved_kv, saved_conv))
    return snapshot


class Job:
    def __init__(self, prompt, max_tokens, sampler_args):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.sampler_args = sampler_args
        self.events = queue.Queue()
        self.cancelled = False
        self.finished = False


def text_model_dir(path):
    """mlx-community/Inkling-2-Mini-4bit's config.json leaves out vision_config.text_hidden_size, so
    transformers 5.24 fills in its default (4608) and then rejects it for not matching text_config.hidden_size
    (2304) when loading the tokenizer. Return a temporary folder that links the model's files with that one
    value set, or `path` itself when the config needs no fix (or is not a local folder)."""
    src = pathlib.Path(path)
    try:
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path
    vision, text = cfg.get("vision_config"), cfg.get("text_config")
    if not (isinstance(vision, dict) and isinstance(text, dict) and "hidden_size" in text
            and vision.get("text_hidden_size") != text["hidden_size"]):
        return path
    dst = pathlib.Path(tempfile.mkdtemp(prefix="inkling2-model-"))
    atexit.register(shutil.rmtree, dst, ignore_errors=True)
    for f in src.iterdir():
        if f.name != "config.json":
            (dst / f.name).symlink_to(f.resolve())
    vision["text_hidden_size"] = text["hidden_size"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"using {dst} (links to {src}, config.json with vision_config.text_hidden_size = "
          f"{text['hidden_size']})", flush=True)
    return str(dst)


class Engine(threading.Thread):
    """Owns the model; all MLX work happens on this one thread."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.error = None
        self.tokenizer = None
        self.prompt_snapshot = None
        self.cache_types = None
        self.pending = None

    def run(self):
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler

            # Long prompts swap without these: MLX's default free-buffer cache keeps
            # gigabytes of freed chunk intermediates around, and without a memory
            # guideline it does not wait for them to be released before allocating.
            recommended = mx.device_info()["max_recommended_working_set_size"]
            limit = (self.args.memory_limit_gb * 1e9 if self.args.memory_limit_gb
                     else recommended - 2.35e9)
            mx.set_memory_limit(int(limit))
            mx.set_cache_limit(int(self.args.cache_limit_mb * 2**20))
            print(f"MLX memory guideline {limit / 1e9:.2f} GB, free-buffer cache "
                  f"{self.args.cache_limit_mb} MiB (GPU working set {recommended / 1e9:.2f} GB)",
                  flush=True)

            self.stream_generate, self.make_sampler = stream_generate, make_sampler
            cfg = None if self.args.vision else {"vision_config": None}
            cfg = cfg or {}
            if self.args.kv_bits is not None:
                cfg.update(kv_bits=self.args.kv_bits, kv_start=self.args.kv_start)
            self.model, self.tokenizer = load(text_model_dir(self.args.model), model_config=cfg)
            if (not self.args.no_prompt_cache
                    and getattr(self.model, "model_type", None) == "inkling2_loop"):
                from mlx_lm.models.cache import (
                    ArraysCache, CacheList, KVCache, QuantizedKVCache, RotatingKVCache,
                )

                self.cache_types = CacheList, (KVCache, RotatingKVCache, QuantizedKVCache), ArraysCache
            self.mx = mx
            self._generate(Job(self.tokenizer.encode("hello"), 8, {"temp": 0.0}), warmup=True)
            self.model_bytes = mx.get_active_memory()
            self.batch_memory_cap = min(int(limit), 20_200_000_000)
            if (self.args.decode_batch_size > 1
                    and getattr(self.model, "model_type", None) != "inkling2_loop"):
                raise ValueError("batched decode requires the text-only Inkling 2 Mini model")
        except Exception as e:  # surfaced by main()
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.pending or self.jobs.get()
            self.pending = None
            jobs = [job]
            if job.cancelled:
                job.events.put(("done", None))
                continue
            try:
                if self.args.decode_batch_size > 1:
                    self._admit(jobs)
                if len(jobs) > 1:
                    self._generate_batch(jobs)
                else:
                    self._generate(job)
            except Exception as e:
                for failed in jobs:
                    if not failed.finished:
                        failed.events.put(("error", str(e)))

    def _admit(self, jobs):
        # Keep FIFO order and reserve for the padded batch at its largest
        # requested length, including output. Unlimited requests run solo.
        def length(job):
            return len(job.prompt) + job.max_tokens if job.max_tokens > 0 else None

        longest = length(jobs[0])
        if longest is None or jobs[0].cancelled:
            return
        deadline = time.monotonic() + self.args.decode_batch_window_ms / 1000
        while len(jobs) < self.args.decode_batch_size:
            try:
                remaining = deadline - time.monotonic()
                job = self.jobs.get(timeout=remaining) if remaining > 0 else self.jobs.get_nowait()
            except queue.Empty:
                return
            size = length(job)
            if size is not None and not job.cancelled:
                padded = (max(size, longest) + 255) // 256 * 256
                # 8 global KV visits (32 KiB/token), 40 local visits and
                # convolution state (~173 MB/stream), plus prefill workspace.
                estimate = self.model_bytes + 1_250_000_000 + (
                    len(jobs) + 1
                ) * (padded * 32768 + 173_000_000)
                if estimate < self.batch_memory_cap:
                    longest = max(size, longest)
                    jobs.append(job)
                    continue
            self.pending = job
            return

    def _prefill_batch_job(self, job):
        # Read prompts one at a time with the same chunk shape and final-token
        # pass as stream_generate. A multi-row prefill would multiply scratch.
        mx = self.mx
        cache = self.model.make_cache()
        prompt = mx.array(job.prompt)
        started = time.perf_counter()
        for start in range(0, len(job.prompt) - 1, self.args.prefill_step):
            if job.cancelled:
                return None
            end = min(start + self.args.prefill_step, len(job.prompt) - 1)
            self.model(prompt[None, start:end], cache=cache)
            mx.eval([visit.state for visit in cache])
            mx.clear_cache()
        logits = self.model(prompt[None, -1:], cache=cache)[:, -1, :]
        try:
            tokens = job.sampler(logits - mx.logsumexp(logits, keepdims=True))
        except ValueError as error:
            job.finished = True
            job.events.put(("error", str(error)))
            return None
        mx.eval(tokens, [visit.state for visit in cache])
        job.prompt_tps = len(job.prompt) / (time.perf_counter() - started)
        return cache, tokens

    def _batch_token(self, job, token):
        if job.generated == 0:
            job.started = time.perf_counter()
        job.generated += 1
        stopped = token in self.tokenizer.eos_token_ids
        if not stopped:
            job.detokenizer.add_token(token)
        done = stopped or job.cancelled or job.generated >= job.max_tokens
        if done:
            job.detokenizer.finalize()
        job.events.put(("text", job.detokenizer.last_segment))
        if done:
            job.finished = True
            job.events.put(("done", SimpleNamespace(
                generation_tokens=job.generated,
                prompt_tps=job.prompt_tps,
                generation_tps=job.generated / max(time.perf_counter() - job.started, 1e-6),
                peak_memory=self.mx.get_peak_memory() / 1e9,
                finish_reason="stop" if stopped else "length",
            )))
        return done

    def _generate_batch(self, jobs):
        from mlx_lm.generate import generation_stream, wired_limit

        self.prompt_snapshot = None
        mx = self.mx
        active, caches, tokens = [], [], []
        print(f"decode cohort: {len(jobs)} prompts={[len(j.prompt) for j in jobs]}", flush=True)
        with wired_limit(self.model, [generation_stream]), mx.stream(generation_stream):
            for job in jobs:
                if job.cancelled:
                    job.finished = True
                    job.events.put(("done", None))
                    continue
                job.detokenizer = self.tokenizer.detokenizer
                job.generated = 0
                try:
                    job.sampler = self.make_sampler(**job.sampler_args)
                except ValueError as error:
                    job.finished = True
                    job.events.put(("error", str(error)))
                    continue
                result = self._prefill_batch_job(job)
                if result is None:
                    if not job.finished:
                        job.finished = True
                        job.events.put(("done", None))
                    continue
                cache, sampled = result
                active.append(job)
                caches.append(cache)
                tokens.append(sampled)
            if not active:
                return
            if len(caches) == 1:
                batch_cache = caches[0]
            else:
                # Merge and release one visit at a time. Holding both complete
                # unbatched and batched histories can exceed the memory budget.
                del cache, result
                batch_cache = []
                for i in range(len(caches[0])):
                    visits = [cache[i] for cache in caches]
                    merged = visits[0].merge(visits)
                    mx.eval(merged.state)
                    batch_cache.append(merged)
                    for cache in caches:
                        cache[i] = None
                del visits
            sampled = mx.concatenate(tokens)
            while active:
                keep = [i for i, token in enumerate(sampled.tolist())
                        if not self._batch_token(active[i], token)]
                if not keep:
                    break
                if len(keep) != len(active):
                    active = [active[i] for i in keep]
                    # Release the old rows as each visit shrinks, like the merge.
                    for visit in batch_cache:
                        visit.filter(keep)
                        mx.eval(visit.state)
                    sampled = sampled[mx.array(keep)]
                logits = self.model(sampled[:, None], cache=batch_cache)[:, -1, :]
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                sampled = mx.concatenate([
                    job.sampler(logprobs[i : i + 1]) for i, job in enumerate(active)
                ])

    def _generate(self, job, *, warmup=False):
        sampler = self.make_sampler(**job.sampler_args)
        prompt = job.prompt
        step = self.args.prefill_step
        # generate_step reserves the last prompt token for a separate forward
        # pass. Resume only at a full prefill block to preserve the cold
        # request's exact forward-pass shapes and cache layout.
        checkpoint = (len(prompt) - 1) // step * step if step > 0 else 0
        previous = self.prompt_snapshot
        self.prompt_snapshot = None
        reused = 0
        saved = None
        kwargs = {}
        if (not warmup and self.cache_types
                and 0 < checkpoint <= self.args.prompt_cache_max_tokens):
            cache_list, kv_types, arrays_cache = self.cache_types
            if (previous is not None and len(previous[0]) <= checkpoint
                    and prompt[:len(previous[0])] == previous[0]):
                prefix, live = previous
                reused = len(prefix)
            else:
                live = self.model.make_cache()
                if not all(
                    type(visit) is cache_list and len(visit.caches) == 2
                    and type(visit[0]) in kv_types and type(visit[1]) is arrays_cache
                    and visit[1].left_padding is None and visit[1].lengths is None
                    for visit in live
                ):
                    live = None
            # The callback with processed == total runs after decode has
            # started. If the saved block already matches the requested
            # checkpoint, the initial callback preserves it before any updates.
            def save_checkpoint(processed, total):
                nonlocal saved
                if processed < total and reused + processed == checkpoint:
                    saved = prompt[:checkpoint], snapshot_prompt_cache(live)

            if live is not None:
                kwargs["prompt_cache"] = live
                kwargs["prompt_progress_callback"] = save_checkpoint
        del previous

        last = None
        responses = self.stream_generate(
            self.model, self.tokenizer, prompt[reused:], max_tokens=job.max_tokens,
            sampler=sampler, prefill_step_size=step, **kwargs,
        )
        with closing(responses):
            for r in responses:
                last = r
                job.events.put(("text", r.text))
                if job.cancelled:
                    break
        self.prompt_snapshot = saved
        if not warmup:
            size = sum(visit.nbytes for visit in saved[1]) if saved else 0
            print(f"prompt cache: reused={reused} prefilled={len(prompt) - reused} "
                  f"saved={len(saved[0]) if saved else 0} bytes={size}", flush=True)
        job.events.put(("done", last))


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None
    model_id = "inkling-2-mini"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        if self.server.verbose:
            super().log_message(fmt, *a)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": self.model_id, "object": "model", "owned_by": "local"}]})
        if self.path.rstrip("/") in ("/health", ""):
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/token/encode", "/tokenize", "/v1/tokenize"):
            return self._encode()
        if path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = normalize_messages(req["messages"])
            kwargs = {}
            if req.get("tools"):
                kwargs["tools"] = req["tools"]
            if req.get("reasoning_effort") is not None:
                kwargs["reasoning_effort"] = req["reasoning_effort"]
            prompt = self.engine.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad request: {e}"}})
        limit = self.server.max_context
        if limit and len(prompt) > limit:
            return self._json(400, {"error": {
                "message": f"prompt is {len(prompt)} tokens; this server's limit is {limit} "
                           "(--max-context), set from what fits in GPU memory"}})

        defaults = self.server.sampling_defaults
        pick = lambda key: req[key] if req.get(key) is not None else defaults.get(key)
        sampler_args = {
            "temp": float(pick("temperature") if pick("temperature") is not None else 1.0),
            "top_p": float(pick("top_p") or 0.0),
            "top_k": int(pick("top_k") or 0),
            "min_p": float(pick("min_p") or 0.0),
        }
        max_tokens = int(req.get("max_completion_tokens") or req.get("max_tokens")
                         or self.server.default_max_tokens)
        job = Job(prompt, max_tokens, sampler_args)
        self.engine.jobs.put(job)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if req.get("stream"):
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            return self._stream(job, cid, created, len(prompt), include_usage)
        return self._complete(job, cid, created, len(prompt))

    def _encode(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = body.get("text") if body.get("text") is not None else body.get("prompt", "")
            tok = getattr(self.engine.tokenizer, "_tokenizer", self.engine.tokenizer)
            ids = tok.encode(text or "", add_special_tokens=False)
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad request: {e}"}})
        self._json(200, {"tokens": ids, "length": len(ids), "count": len(ids)})

    def _collect(self, job):
        text = ""
        while True:
            kind, val = job.events.get()
            if kind == "error":
                raise RuntimeError(val)
            if kind == "done":
                return text, val
            text += val
            yield text

    def _finish(self, last, tools):
        if tools:
            return "tool_calls"
        return "length" if last is not None and last.finish_reason == "length" else "stop"

    def _usage(self, n_prompt, last):
        n = last.generation_tokens if last is not None else 0
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n,
                 "total_tokens": n_prompt + n}
        if last is not None:
            usage["peak_memory_gb"] = round(last.peak_memory, 3)
            usage["prompt_tps"] = round(last.prompt_tps, 1)
            usage["generation_tps"] = round(last.generation_tps, 2)
        return usage

    def _complete(self, job, cid, created, n_prompt):
        gen = self._collect(job)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            text, last = stop.value
        except RuntimeError as e:
            return self._json(500, {"error": {"message": str(e)}})
        reasoning, content, tools, _ = parse_reply(text, final=True)
        calls = openai_tool_calls(tools)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": self.model_id,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": self._finish(last, calls)}],
            "usage": self._usage(n_prompt, last),
        })

    def _stream(self, job, cid, created, n_prompt, include_usage):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(delta=None, finish=None, usage=None):
            ev = {"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": self.model_id,
                  "choices": [] if usage else [{"index": 0, "delta": delta or {},
                                                "finish_reason": finish}]}
            if usage:
                ev["usage"] = usage
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()

        sent_r = sent_c = 0
        try:
            send({"role": "assistant", "content": ""})
            gen = self._collect(job)
            try:
                while True:
                    text = next(gen)
                    reasoning, content, _, _ = parse_reply(text, final=False)
                    if len(reasoning) > sent_r:
                        send({"reasoning_content": reasoning[sent_r:]})
                        sent_r = len(reasoning)
                    if len(content) > sent_c:
                        send({"content": content[sent_c:]})
                        sent_c = len(content)
            except StopIteration as stop:
                text, last = stop.value
            reasoning, content, tools, _ = parse_reply(text, final=True)
            if len(reasoning) > sent_r:
                send({"reasoning_content": reasoning[sent_r:]})
            if len(content) > sent_c:
                send({"content": content[sent_c:]})
            calls = openai_tool_calls(tools)
            if calls:
                send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            send(finish=self._finish(last, calls))
            if include_usage:
                send(usage=self._usage(n_prompt, last))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            job.cancelled = True
            for kind, _ in iter(job.events.get, None):  # drain until the engine stops
                if kind in ("done", "error"):
                    break
        except RuntimeError as e:
            try:
                self.wfile.write(f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode())
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--model-id", default="inkling-2-mini")
    ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8), nargs="?", const=8,
                    help="pack full-attention KV after --kv-start (8 bits when used without a value)")
    ap.add_argument("--kv-start", type=int, default=65536,
                    help="cached tokens before switching at a forward boundary (default: 65536)")
    ap.add_argument("--no-prompt-cache", action="store_true",
                    help="disable prompt-prefix reuse across requests")
    ap.add_argument("--prompt-cache-max-tokens", type=int, default=16384,
                    help="maximum saved prefix length in tokens "
                         "(default: 16384)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-context", type=int, default=0,
                    help="refuse prompts longer than this many tokens (0 = no limit)")
    ap.add_argument("--memory-limit-gb", type=float, default=0.0,
                    help="MLX memory guideline (default: GPU working set minus 2.35 GB)")
    ap.add_argument("--cache-limit-mb", type=int, default=64,
                    help="MLX free-buffer cache (default: 64 MiB)")
    ap.add_argument("--vision", action="store_true",
                    help="load the vision tower too (no image API here; only for parity)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--decode-batch-size", type=int, choices=(1, 2, 3, 4), default=1,
                    help="maximum simultaneous decode requests (default: 1)")
    ap.add_argument("--decode-batch-window-ms", type=float, default=15,
                    help="maximum time to collect a decode cohort (default: 15 ms)")
    args = ap.parse_args()
    if args.kv_start < 0:
        ap.error("--kv-start must be nonnegative")
    if not 0 <= args.decode_batch_window_ms <= 100:
        ap.error("--decode-batch-window-ms must be between 0 and 100")
    if args.decode_batch_size > 1:
        if args.vision:
            ap.error("batched decode requires the text-only model")
        if not 1 <= args.prefill_step <= 512:
            ap.error("batched decode requires --prefill-step between 1 and 512")
        if args.kv_bits is not None:
            # Batched decode was only qualified with BF16 KV.
            ap.error("batched decode and --kv-bits cannot be combined")
    if args.prompt_cache_max_tokens < 0:
        ap.error("--prompt-cache-max-tokens must be nonnegative")

    engine = Engine(args)
    engine.start()
    engine.ready.wait()
    if engine.error:
        raise SystemExit(f"model load failed: {engine.error}")
    Handler.engine, Handler.model_id = engine, args.model_id
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose, server.default_max_tokens = args.verbose, args.max_tokens
    server.max_context = args.max_context
    gen_cfg = pathlib.Path(args.model) / "generation_config.json"
    cfg = json.loads(gen_cfg.read_text()) if gen_cfg.exists() else {}
    server.sampling_defaults = {k: cfg[k] for k in ("temperature", "top_p", "top_k", "min_p")
                                if cfg.get(k) is not None}
    print(f"serving {args.model_id} on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
