#!/usr/bin/env python3
"""OpenAI-compatible server for Hy4-flash (HYV4FlashForConditionalGeneration) on exllamav3.

Loads through exllamav3.model_init (all of its flags pass through: -m, -cs, -cq, --mtp, -ndt, -dc, -ngr, -mcs, ...),
plus the vision component. Prompts are rendered with the checkpoint's own chat_template.jinja (reasoning_effort
high / no_think, tools); images replace the template's <image start><pad><end> triple with the vision embedding
(whose token string carries the start / end tokens itself). Output is split into reasoning_content / content /
tool_calls following the template's <think:opensource> and <tool_calls:opensource> markers.

  python serve_hy4.py -m MODEL_DIR -cs 1048576 -cq 8,8 --mtp -ndt 6 -ngr --port 30200
"""
import argparse, asyncio, base64, io, json, os, re, sys, time, uuid, urllib.request

ap = argparse.ArgumentParser()
from exllamav3 import model_init
model_init.add_args(ap, cache = True, add_draft_model_args = True, default_cache_size = 1048576, default_chunk_size = 8192)
ap.add_argument("--host", default = "127.0.0.1")
ap.add_argument("--port", type = int, default = 30200)
ap.add_argument("--served-name", default = "hy4-flash")
ap.add_argument("--max-batch-size", type = int, default = 4)
ap.add_argument("--no-vision", action = "store_true")
ap.add_argument("--default-reasoning", default = "high", choices = ["high", "no_think"])
args = ap.parse_args()

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn, jinja2
from PIL import Image
from exllamav3 import Model, AsyncGenerator, AsyncJob
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler

t_start = time.time()
model, config, cache, tok, draft_model, draft_config, draft_cache = model_init.init(args)
vision = None
if not args.no_vision:
    vision = Model.from_config(config, component = "vision")
    vision.load(progressbar = False)
    print(" -- vision tower loaded", flush = True)
CTX = args.cache_size
# Ring-cached sliding MLA layers (EXL3_MLA_RING_PAGES): one shared page map for the main cache
from exllamav3.cache.mla_ring import _RING_MAPS
RING = _RING_MAPS.get(next((L.cache_id for L in (cache.layers.values() if isinstance(cache.layers, dict) else []) if hasattr(L, "ring_map")), None)) if os.environ.get("EXL3_MLA_RING_PAGES") else None
if RING is None and os.environ.get("EXL3_MLA_RING_PAGES") and _RING_MAPS:
    RING = next(iter(_RING_MAPS.values()))
print(f" -- MLA ring: {('%d pages, window %d' % (RING.num_pages, RING.window)) if RING else 'off'}", flush = True)
EOS = list(tok.config.eos_token_id_list) if getattr(tok.config, "eos_token_id_list", None) else [tok.eos_token_id]
t_load = time.time() - t_start
print(f" -- loaded in {t_load:.1f} s, ctx {CTX}, draft {'mtp' if draft_model is not None else 'off'}, eos {EOS}", flush = True)

# ------------------------------------------------------------------ template
_tpl_src = open(os.path.join(args.model_dir, "chat_template.jinja")).read()
_env = jinja2.Environment(trim_blocks = True, lstrip_blocks = True, extensions = ["jinja2.ext.loopcontrols"])
def _raise(msg): raise ValueError(msg)
_env.globals["raise_exception"] = _raise
_env.filters["tojson"] = lambda v, ensure_ascii = False, indent = None: json.dumps(v, ensure_ascii = ensure_ascii, indent = indent)
TPL = _env.from_string(_tpl_src)
HY = ":opensource"
IMG_TRIPLE = f"<｜hy_image▁start{HY}｜><｜hy_image▁pad{HY}｜><｜hy_image▁end{HY}｜>"
THINK_B, THINK_E = f"<think{HY}>", f"</think{HY}>"
TC_B, TC_E = f"<tool_calls{HY}>", f"</tool_calls{HY}>"
CALL_RE = re.compile(rf"<tool_call{HY}>(.*?)</tool_call{HY}>", re.S)
ARG_RE = re.compile(rf"<arg_key{HY}>(.*?)</arg_key{HY}>\s*<arg_value{HY}>(.*?)</arg_value{HY}>", re.S)
END_MARKS = [f"<｜hy_end{HY}｜>"]

app = FastAPI()
G = {"gen": None, "active": 0, "stats": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                          "accepted_draft_tokens": 0, "rejected_draft_tokens": 0, "images": 0}}


def load_image(part):
    src = part.get("image_url") or part.get("image") or part.get("input_image")
    if isinstance(src, dict):
        src = src.get("url") or src.get("data")
    if not isinstance(src, str):
        raise ValueError("image part without url")
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(src, headers = {"User-Agent": "hy4-server"}), timeout = 30) as r:
            data = r.read()
    else:
        data = open(src, "rb").read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def render(body):
    msgs = body.get("messages") or []
    tools = body.get("tools") or None
    effort = body.get("reasoning_effort")
    ctk = body.get("chat_template_kwargs") or {}
    if "thinking" in ctk or "enable_thinking" in ctk:
        effort = "high" if (ctk.get("thinking", ctk.get("enable_thinking"))) else "no_think"
    effort = effort or args.default_reasoning
    effort = "no_think" if effort in ("none", "no_think", "minimal", False) else "high"
    images = []
    norm = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try: fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError: fn["arguments"] = {"input": fn["arguments"]}
                tcs.append(tc)
            m["tool_calls"] = tcs
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and (p.get("type") in ("image_url", "image", "input_image") or "image_url" in p):
                    if vision is None:
                        raise ValueError("vision is disabled on this server")
                    images.append(load_image(p))
        norm.append(m)
    text = TPL.render(messages = norm, tools = tools, add_generation_prompt = True, reasoning_effort = effort,
                      **{k: v for k, v in ctk.items() if k not in ("thinking", "enable_thinking")})
    embs = []
    if images:
        assert text.count(IMG_TRIPLE) == len(images), (text.count(IMG_TRIPLE), len(images))
        for img in images:
            e = vision.get_image_embeddings(tokenizer = tok, image = img)
            embs.append(e)
            text = text.replace(IMG_TRIPLE, e.text_alias, 1)
        G["stats"]["images"] += len(images)
    ids = tok.encode(text, encode_special_tokens = True, add_bos = False, embeddings = embs or None)
    return ids, effort != "no_think", embs


def split_output(text, thinking):
    """-> (reasoning, content, tool_calls). Generation starts inside <think> when thinking."""
    reasoning, rest = "", text
    if thinking:
        i = rest.find(THINK_E)
        if i < 0:
            return rest, "", []
        reasoning, rest = rest[:i], rest[i + len(THINK_E):]
    calls = []
    j = rest.find(TC_B)
    if j >= 0:
        block = rest[j + len(TC_B):]
        k = block.find(TC_E)
        block = block if k < 0 else block[:k]
        for n, body in enumerate(CALL_RE.findall(block)):
            name = re.split(rf"<arg_key{HY}>", body, 1)[0].strip()
            argd = {}
            for key, val in ARG_RE.findall(body):
                v = val.strip()
                try: argd[key.strip()] = json.loads(v)
                except json.JSONDecodeError: argd[key.strip()] = val
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(argd, ensure_ascii = False)}})
        rest = rest[:j]
    for m in END_MARKS:
        rest = rest.replace(m, "")
    return reasoning, rest, calls


def make_sampler(body):
    t = body.get("temperature")
    t = 1.0 if t is None else float(t)
    if t <= 0:
        return ArgmaxSampler()
    return ComboSampler(temperature = t, top_p = float(body.get("top_p") or 1.0), top_k = int(body.get("top_k") or 0),
                        min_p = float(body.get("min_p") or 0.0))


@app.on_event("startup")
async def startup():
    G["gen"] = AsyncGenerator(model = model, cache = cache, tokenizer = tok, max_batch_size = args.max_batch_size,
                              max_chunk_size = args.chunk_size, draft_model = draft_model, draft_cache = draft_cache,
                              num_draft_tokens = args.num_draft_tokens if draft_model is not None else None,
                              draft_confidence = args.draft_confidence, dynamic_draft_tokens = bool(args.dynamic_draft),
                              record_draft_stats = draft_model is not None, enable_defrag = RING is None, sparse_draft_head = args.sparse_draft_head, hybrid_draft_tokens = args.hybrid_draft_tokens)
    if RING is not None:
        G["gen"].generator.mla_ring = RING


async def run_job(ids, body, embs):
    gen = G["gen"]
    G["active"] += 1
    room = CTX - ids.shape[-1] - (args.num_draft_tokens or 8) - 4
    mx = body.get("max_tokens") or body.get("max_completion_tokens")
    max_new = room if mx is None else min(int(mx), room)
    job = AsyncJob(gen, input_ids = ids, max_new_tokens = max(1, max_new), sampler = make_sampler(body),
                   stop_conditions = EOS, decode_special_tokens = True, embeddings = embs or None)
    G["stats"]["requests"] += 1
    G["stats"]["prompt_tokens"] += ids.shape[-1]
    n, text, done = 0, "", False
    try:
        while not job.cancelled:
            r = await job.queue.get()
            batch = [r]
            while not job.queue.empty():
                batch.append(job.queue.get_nowait())
            chunk_parts, fin, stop_loop = [], None, False
            for r in batch:
                if not isinstance(r, dict):
                    if isinstance(r, Exception):
                        raise r
                    stop_loop = True
                    break
                if r.get("stage") == "streaming":
                    c = r.get("text") or ""
                    if c:
                        chunk_parts.append(c)
                    tids = r.get("token_ids")
                    n += int(tids.numel()) if tids is not None else 0
                if r.get("eos"):
                    fin = "length" if r.get("eos_reason") == "max_new_tokens" else "stop"
                    G["stats"]["completion_tokens"] += n
                    G["stats"]["accepted_draft_tokens"] += int(r.get("accepted_draft_tokens", 0) or 0)
                    G["stats"]["rejected_draft_tokens"] += int(r.get("rejected_draft_tokens", 0) or 0)
                    done = True
                    stop_loop = True
                    break
            chunk = "".join(chunk_parts)
            if chunk:
                text += chunk
            if chunk or fin is not None:
                yield text, chunk, n, fin
            if stop_loop:
                break
        done = True
    finally:
        G["active"] = max(0, G["active"] - 1)
        if not done:
            try: await job.cancel()
            except Exception: pass


def sse(o):
    return "data: " + json.dumps(o, ensure_ascii = False) + "\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/token/encode")
async def token_encode(req: Request):
    body = await req.json()
    text = body.get("text") or ""
    ids = tok.encode(text, encode_special_tokens = False, add_bos = False)
    toks = ids[0].tolist() if ids.numel() else []
    return {"tokens": toks, "length": len(toks)}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local", "max_model_len": CTX}]}


@app.get("/stats")
async def stats():
    s = dict(G["stats"]); s["active"] = G["active"]; s["load_s"] = t_load; s["tier"] = {}
    s["vram_alloc_gib"] = torch.cuda.memory_allocated() / 2 ** 30
    g = G["gen"].generator if G["gen"] is not None else None
    if g is not None:
        s["num_draft_tokens"] = getattr(g, "num_draft_tokens", None)
        s["draft_confidence"] = getattr(g, "draft_confidence", None)
    if RING is not None:
        s["ring"] = {"pages": RING.num_pages, "resident": len(RING.g2r), "evictions": RING.evictions}
    return s


@app.post("/admin/draft")
async def admin_draft(req: Request):
    """Live draft knobs: {"num_draft_tokens": k, "draft_confidence": c}"""
    b = await req.json()
    g = G["gen"].generator
    out = {}
    if "num_draft_tokens" in b and int(b["num_draft_tokens"]) > (args.num_draft_tokens or 0):
        return JSONResponse({"error": f"num_draft_tokens can only be lowered live (startup value {args.num_draft_tokens})"}, status_code = 400)
    for k in ("num_draft_tokens", "draft_confidence"):
        if k in b and hasattr(g, k):
            setattr(g, k, type(getattr(g, k))(b[k]) if getattr(g, k) is not None else b[k]); out[k] = getattr(g, k)
    return out


@app.post("/debug/render")
async def debug_render(req: Request):
    body = await req.json()
    ids, thinking, embs = render(body)
    return {"tokens": int(ids.shape[-1]), "thinking": thinking, "text": tok.decode(ids[0][ids[0] >= 0].unsqueeze(0), decode_special_tokens = True)[0] if hasattr(tok, "decode") else None}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    try:
        ids, thinking, embs = render(body)
    except (ValueError, AssertionError, OSError) as e:
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code = 400)
    P = ids.shape[-1]
    if P >= CTX:
        return JSONResponse({"error": {"message": f"prompt {P} tokens >= context {CTX}", "type": "invalid_request_error"}}, status_code = 400)
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    want_tools = bool(body.get("tools"))
    if body.get("stream"):
        async def events():
            sent_r, sent_c = 0, 0
            yield sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            last_dc = time.monotonic()
            async for text, chunk, n, fin in run_job(ids, body, embs):
                now_dc = time.monotonic()
                if now_dc - last_dc >= 0.5:
                    last_dc = now_dc
                    if await req.is_disconnected():
                        break
                r, c, calls = split_output(text, thinking)
                def hold_tag(s, tags):
                    n, hold = len(s), 0
                    for tag in tags:
                        for k in range(min(n, len(tag) - 1), hold, -1):
                            if s.endswith(tag[:k]):
                                hold = k
                                break
                    return s[:-hold] if hold else s
                r_s = r if (not thinking or THINK_E in text) else hold_tag(r, (THINK_E,))
                c_s = c if fin is not None else hold_tag(c, (TC_B, *END_MARKS))
                delta = {}
                if len(r_s) > sent_r:
                    delta["reasoning_content"] = r_s[sent_r:]; sent_r = len(r_s)
                if len(c_s) > sent_c:
                    delta["content"] = c_s[sent_c:]; sent_c = len(c_s)
                if delta:
                    yield sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                if fin is not None:
                    if calls and want_tools:
                        yield sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                                   "choices": [{"index": 0, "delta": {"tool_calls": [dict(t, index = i) for i, t in enumerate(calls)]},
                                                "finish_reason": None}]})
                    yield sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                               "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if (calls and want_tools) else fin}],
                               "usage": {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}})
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type = "text/event-stream")
    text, n, fin = "", 0, "stop"
    last_dc = time.monotonic()
    async for text, chunk, n, f in run_job(ids, body, embs):
        if f is not None:
            fin = f
        now_dc = time.monotonic()
        if now_dc - last_dc >= 0.5:
            last_dc = now_dc
            if await req.is_disconnected():
                break
    r, c, calls = split_output(text, thinking)
    msg = {"role": "assistant", "content": c}
    if r:
        msg["reasoning_content"] = r
    if calls and want_tools:
        msg["tool_calls"] = calls; fin = "tool_calls"
    return {"id": rid, "object": "chat.completion", "created": created, "model": args.served_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}}


if __name__ == "__main__":
    uvicorn.run(app, host = args.host, port = args.port, log_level = "warning")
