#!/usr/bin/env python3
"""Omarchy Local AI gateway: one endpoint, three dialects, one engine.

Listens on GATEWAY_PORT (default 12434) and forwards to the engine at
UPSTREAM (default http://engine:8000), which speaks OpenAI chat completions.

  /v1/models, /v1/chat/completions   pass through unchanged (streaming too)
  /v1/messages                       Anthropic Messages  -> chat completions
  /v1/messages/count_tokens          rough estimate, Anthropic shape
  /v1/responses                      OpenAI Responses    -> chat completions
  /health                            200 when the engine answers /v1/models

Auth: when GATEWAY_KEY_FILE names a non-empty file, every request must carry
its value as `Authorization: Bearer <key>` or `x-api-key: <key>`. The file is
read on each request, so the plugin can rotate the key without a restart.

Usage: when GATEWAY_USAGE_LOG names a file, one JSON line per answered
request is appended to it: {t, api, model, prompt, completion, estimated,
ttft_ms, ms}. Tokens come from the engine's usage block when it sends one,
otherwise one per streamed delta (estimated: true). The plugin reads the file
for the tokens-per-day and decode-speed figures on the card.

Connections: at most GATEWAY_MAX_CONNECTIONS (default 64) are served at once;
one more is closed at once. Each request, its headers and body, must arrive
within GATEWAY_REQUEST_DEADLINE seconds (default 10) of the gateway starting
to wait for it, idle keep-alive included, or its connection is closed. Every
write to a client must finish within GATEWAY_WRITE_DEADLINE seconds (default
30). An answer without the key (/health, 401) closes its connection, so
pipelined requests are never answered. So a client without the key holds a
thread no longer than one request's deadline, however it sends or reads.

Standard library only. No configuration beyond the environment above.
"""

import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://engine:8000").rstrip("/")
PORT = int(os.environ.get("GATEWAY_PORT", "12434"))
KEY_FILE = os.environ.get("GATEWAY_KEY_FILE", "")
MODEL_ALIAS = os.environ.get("MODEL", "")  # served model id; client model names are mapped onto it
TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "600"))
USAGE_LOG = os.environ.get("GATEWAY_USAGE_LOG", "")
USAGE_LOCK = threading.Lock()
REQUEST_DEADLINE = float(os.environ.get("GATEWAY_REQUEST_DEADLINE", "10"))
WRITE_DEADLINE = float(os.environ.get("GATEWAY_WRITE_DEADLINE", "30"))
MAX_CONNECTIONS = int(os.environ.get("GATEWAY_MAX_CONNECTIONS", "64"))


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr, flush=True)


def required_key():
    if not KEY_FILE:
        return ""
    try:
        with open(KEY_FILE) as handle:
            return handle.read().strip()
    except OSError:
        return ""


class Meter:
    """One answer's accounting: tokens from the engine's usage when it sends one, else one per delta; time to first token."""

    def __init__(self, api, model):
        self.api, self.model, self.t0 = api, model, time.monotonic()
        self.first, self.counted, self.usage = None, 0, None

    def chunk(self, obj):
        if obj.get("usage"):
            self.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                if self.first is None:
                    self.first = time.monotonic()
                self.counted += 1

    def done(self, usage=None):
        if not USAGE_LOG:
            return
        usage = usage or self.usage or {}
        now = time.monotonic()
        row = {"t": int(time.time()), "api": self.api, "model": self.model, "prompt": usage.get("prompt_tokens", 0),
               "completion": usage.get("completion_tokens") or self.counted, "estimated": not usage,
               "ttft_ms": int(((self.first or now) - self.t0) * 1000), "ms": int((now - self.t0) * 1000)}
        try:
            with USAGE_LOCK, open(USAGE_LOG, "a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError:
            pass


# ----------------------------------------------------------------------------- upstream
class Upstream:
    """One HTTP request to the engine. `stream()` yields SSE data payloads as parsed JSON."""

    def __init__(self, method, path, body=None):
        parsed = urllib.parse.urlparse(UPSTREAM)
        self.conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=TIMEOUT)
        headers = {"Content-Type": "application/json", "Accept": "*/*"}
        data = json.dumps(body).encode() if body is not None else None
        self.conn.request(method, path, body=data, headers=headers)
        self.response = self.conn.getresponse()
        self.status = self.response.status

    def json(self):
        raw = self.response.read()
        self.conn.close()
        return json.loads(raw) if raw else {}

    def raw(self):
        raw = self.response.read()
        self.conn.close()
        return raw

    def stream(self):
        try:
            buffer = b""
            while True:
                chunk = self.response.read1(65536) if hasattr(self.response, "read1") else self.response.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        finally:
            self.conn.close()


def upstream_model(requested):
    return MODEL_ALIAS or requested


def scrub_schema(schema):
    """Tool parameter schemas as the engine can take them. llama.cpp compiles them to a grammar and rejects
    regex escapes it does not know (Claude Code ships a pattern with \\- ); `pattern` and `format` are
    validation hints the model does not need to call the tool, so they are dropped at every depth."""
    if isinstance(schema, dict):
        return {k: scrub_schema(v) for k, v in schema.items() if k not in ("pattern", "format")}
    if isinstance(schema, list):
        return [scrub_schema(v) for v in schema]
    return schema


def hoist_system(messages):
    """One leading system message. Chat templates such as Qwen's refuse a system message anywhere
    else, and clients send them mid-conversation (Codex developer items, Claude's per-turn system)."""
    system = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not system:
        return rest
    return [{"role": "system", "content": "\n\n".join(c for c in system if c)}] + rest


def chat_content(content):
    """Preserve image inputs when translating Messages or Responses to Chat."""
    if not isinstance(content, list):
        return content
    parts = []
    for block in content:
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind in ("image", "input_image"):
            source = block.get("source") or {}
            url = block.get("image_url") or source.get("url")
            if source.get("type") == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
            if not url:
                raise ValueError("image input requires a URL or base64 data")
            image = {"url": url}
            if block.get("detail"):
                image["detail"] = block["detail"]
            parts.append({"type": "image_url", "image_url": image})
    return parts if any(p["type"] == "image_url" for p in parts) else "\n".join(p["text"] for p in parts)


# ----------------------------------------------------------------------------- Anthropic Messages -> chat
def anthropic_to_chat(req):
    messages = []
    system = req.get("system")
    if isinstance(system, list):
        system = "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    if system:
        messages.append({"role": "system", "content": system})
    for message in req.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts, tool_calls, tool_results = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind in ("text", "image"):
                parts.append(block)
            elif kind == "tool_use":
                tool_calls.append({"id": block.get("id"), "type": "function",
                                   "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}})
            elif kind == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    # Chat tool results are text; images returned by Read become
                    # user image blocks after the corresponding tool results.
                    parts.extend(b for b in inner if b.get("type") == "image")
                    inner = chat_content([b for b in inner if b.get("type") != "image"])
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": inner if isinstance(inner, str) else json.dumps(inner)})
        if role == "assistant":
            entry = {"role": "assistant", "content": chat_content(parts) if parts else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
        else:
            if tool_results:
                messages.extend(tool_results)
            if parts:
                messages.append({"role": "user", "content": chat_content(parts)})
    out = {"model": upstream_model(req.get("model")), "messages": hoist_system(messages), "stream": bool(req.get("stream"))}
    if req.get("max_tokens"):
        out["max_tokens"] = req["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in req:
            out[key] = req[key]
    if req.get("stop_sequences"):
        out["stop"] = req["stop_sequences"]
    tools = req.get("tools") or []
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
                                                          "parameters": scrub_schema(t.get("input_schema") or {"type": "object", "properties": {}})}}
                        for t in tools if t.get("name")]
        choice = req.get("tool_choice") or {}
        kind = choice.get("type")
        if kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif kind == "none":
            out["tool_choice"] = "none"
        else:
            out["tool_choice"] = "auto"
    return out


def anthropic_stop(finish_reason, had_tools):
    if had_tools or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def chat_to_anthropic(resp, model):
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        content.append({"type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": function.get("name"), "input": arguments})
    usage = resp.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": anthropic_stop(choice.get("finish_reason"), bool(tool_calls)), "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def anthropic_stream(handler, upstream, model, meter=None):
    """Translate chat SSE deltas into the Anthropic event stream."""
    def event(name, data):
        handler.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        handler.wfile.flush()

    message_id = "msg_" + uuid.uuid4().hex[:24]
    event("message_start", {"type": "message_start", "message": {
        "id": message_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
    index = -1
    text_open = False
    tools = {}  # chat tool_call index -> anthropic block index
    finish = None
    usage = {}
    for chunk in upstream.stream():
        if meter:
            meter.chunk(chunk)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not text_open:
                    index += 1
                    text_open = True
                    event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
                event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": delta["content"]}})
            for call in delta.get("tool_calls") or []:
                position = call.get("index", 0)
                if position not in tools:
                    if text_open:
                        event("content_block_stop", {"type": "content_block_stop", "index": index})
                        text_open = False
                    index += 1
                    tools[position] = index
                    function = call.get("function") or {}
                    event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                        "type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", "name": function.get("name") or "", "input": {}}})
                arguments = (call.get("function") or {}).get("arguments")
                if arguments:
                    event("content_block_delta", {"type": "content_block_delta", "index": tools[position], "delta": {"type": "input_json_delta", "partial_json": arguments}})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if text_open:
        event("content_block_stop", {"type": "content_block_stop", "index": index})
    for block_index in tools.values():
        event("content_block_stop", {"type": "content_block_stop", "index": block_index})
    event("message_delta", {"type": "message_delta", "delta": {"stop_reason": anthropic_stop(finish, bool(tools)), "stop_sequence": None},
                            "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    event("message_stop", {"type": "message_stop"})
    if meter:
        meter.done(usage)


# ----------------------------------------------------------------------------- OpenAI Responses -> chat
def responses_to_chat(req):
    messages = []
    if req.get("instructions"):
        messages.append({"role": "system", "content": req["instructions"]})
    items = req.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items or []:
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "message":
            role = item.get("role", "user")
            content = item.get("content")
            content = chat_content(content)
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": content or ""})
        elif kind == "function_call":
            call = {"id": item.get("call_id") or item.get("id"), "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}}
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output if isinstance(output, str) else json.dumps(output)})
        elif kind == "reasoning":
            continue
    out = {"model": upstream_model(req.get("model")), "messages": hoist_system(messages), "stream": bool(req.get("stream"))}
    if req.get("max_output_tokens"):
        out["max_tokens"] = req["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if key in req and req[key] is not None:
            out[key] = req[key]
    tools = [t for t in req.get("tools") or [] if t.get("type") == "function"]
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                                          "parameters": scrub_schema(t.get("parameters") or {"type": "object", "properties": {}})}} for t in tools]
        choice = req.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif choice in ("required", "none", "auto"):
            out["tool_choice"] = choice
        else:
            out["tool_choice"] = "auto"
    if req.get("parallel_tool_calls") is False:
        out["parallel_tool_calls"] = False
    return out


def responses_envelope(model, req):
    return {"id": "resp_" + uuid.uuid4().hex[:24], "object": "response", "created_at": int(time.time()), "model": model,
            "status": "in_progress", "output": [], "error": None, "incomplete_details": None,
            "instructions": req.get("instructions"), "parallel_tool_calls": req.get("parallel_tool_calls", True),
            "tool_choice": req.get("tool_choice", "auto"), "tools": req.get("tools") or [], "usage": None, "metadata": req.get("metadata") or {}}


def chat_to_responses(resp, model, req):
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    if message.get("content"):
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:24], "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append({"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24], "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:24],
                       "name": function.get("name"), "arguments": function.get("arguments") or "{}", "status": "completed"})
    usage = resp.get("usage") or {}
    envelope = responses_envelope(model, req)
    envelope.update({"status": "completed", "output": output,
                     "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                               "total_tokens": usage.get("total_tokens", 0), "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    return envelope


def responses_stream(handler, upstream, model, req, meter=None):
    """Translate chat SSE deltas into the Responses event stream Codex consumes."""
    sequence = [0]

    def event(name, data):
        sequence[0] += 1
        data = dict(data, type=name, sequence_number=sequence[0])
        handler.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        handler.wfile.flush()

    envelope = responses_envelope(model, req)
    event("response.created", {"response": envelope})
    event("response.in_progress", {"response": envelope})
    output = []
    output_index = -1
    text_item = None  # (index, id, accumulated text)
    calls = {}  # chat index -> {"index", "id", "call_id", "name", "arguments"}
    usage = {}
    for chunk in upstream.stream():
        if meter:
            meter.chunk(chunk)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if text_item is None:
                    output_index += 1
                    text_item = [output_index, "msg_" + uuid.uuid4().hex[:24], ""]
                    item = {"type": "message", "id": text_item[1], "status": "in_progress", "role": "assistant", "content": []}
                    event("response.output_item.added", {"output_index": output_index, "item": item})
                    event("response.content_part.added", {"item_id": text_item[1], "output_index": output_index, "content_index": 0,
                                                          "part": {"type": "output_text", "text": "", "annotations": []}})
                text_item[2] += delta["content"]
                event("response.output_text.delta", {"item_id": text_item[1], "output_index": text_item[0], "content_index": 0, "delta": delta["content"]})
            for call in delta.get("tool_calls") or []:
                position = call.get("index", 0)
                if position not in calls:
                    if text_item is not None:
                        finish_text(event, text_item, output)
                        text_item = None
                    output_index += 1
                    function = call.get("function") or {}
                    calls[position] = {"index": output_index, "id": "fc_" + uuid.uuid4().hex[:24],
                                       "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:24], "name": function.get("name") or "", "arguments": ""}
                    item = {"type": "function_call", "id": calls[position]["id"], "call_id": calls[position]["call_id"],
                            "name": calls[position]["name"], "arguments": "", "status": "in_progress"}
                    event("response.output_item.added", {"output_index": output_index, "item": item})
                function = call.get("function") or {}
                if function.get("name") and not calls[position]["name"]:
                    calls[position]["name"] = function["name"]
                if function.get("arguments"):
                    calls[position]["arguments"] += function["arguments"]
                    event("response.function_call_arguments.delta", {"item_id": calls[position]["id"], "output_index": calls[position]["index"], "delta": function["arguments"]})
    if text_item is not None:
        finish_text(event, text_item, output)
    for call in calls.values():
        event("response.function_call_arguments.done", {"item_id": call["id"], "output_index": call["index"], "arguments": call["arguments"]})
        item = {"type": "function_call", "id": call["id"], "call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"], "status": "completed"}
        output.append((call["index"], item))
        event("response.output_item.done", {"output_index": call["index"], "item": item})
    envelope.update({"status": "completed", "output": [item for _, item in sorted(output, key=lambda pair: pair[0])],
                     "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                               "total_tokens": usage.get("total_tokens", 0), "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    event("response.completed", {"response": envelope})
    if meter:
        meter.done(usage)


def finish_text(event, text_item, output):
    index, item_id, text = text_item
    event("response.output_text.done", {"item_id": item_id, "output_index": index, "content_index": 0, "text": text})
    part = {"type": "output_text", "text": text, "annotations": []}
    event("response.content_part.done", {"item_id": item_id, "output_index": index, "content_index": 0, "part": part})
    item = {"type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [part]}
    output.append((index, item))
    event("response.output_item.done", {"output_index": index, "item": item})


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "omarchy-local-ai-gateway/1"
    timeout = WRITE_DEADLINE  # each socket operation; a client that stops reading cannot block a write for longer

    def log_message(self, fmt, *args):  # quiet by default; errors go through log()
        pass

    # Every request gets REQUEST_DEADLINE from when the gateway starts waiting for it until its headers, and its body
    # when it has one, are in; past that the connection is shut, so a slow sender frees its thread whatever the key.
    def handle_one_request(self):
        self.deadline = threading.Timer(REQUEST_DEADLINE, self.expire)
        self.deadline.daemon = True
        self.deadline.start()
        try:
            super().handle_one_request()
        finally:
            self.deadline.cancel()

    def expire(self):
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def refuse(self):
        self.close_connection = True  # an unread body must not be taken for the next request
        self.send_json(401, {"error": {"type": "authentication_error", "message": "invalid or missing API key"}})

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def authorized(self):
        key = required_key()
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "").strip()
        return given == key

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.deadline.cancel()
        return json.loads(raw) if raw else {}

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            self.close_connection = True  # no key: one answer, then the connection is closed
            try:
                up = Upstream("GET", "/v1/models")
                up.raw()
                self.send_json(200 if up.status == 200 else 503, {"status": "ok" if up.status == 200 else "engine unavailable"})
            except Exception as error:
                self.send_json(503, {"status": f"engine unavailable: {error}"})
            return
        if not self.authorized():
            return self.refuse()
        self.deadline.cancel()
        if path in ("/v1/models", "/models"):
            return self.passthrough("GET", "/v1/models", None)
        self.send_json(404, {"error": {"type": "not_found", "message": path}})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if not self.authorized():
            return self.refuse()
        try:
            body = self.read_body()
        except json.JSONDecodeError:
            return self.send_json(400, {"error": {"type": "invalid_request_error", "message": "body is not JSON"}})
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                if isinstance(body.get("messages"), list):
                    body["messages"] = hoist_system(body["messages"])
                if isinstance(body.get("tools"), list):
                    body["tools"] = scrub_schema(body["tools"])
                if MODEL_ALIAS:
                    body["model"] = MODEL_ALIAS
                hide = False
                if USAGE_LOG and body.get("stream"):  # ask the engine for its token counts; the client sees them only if it asked
                    options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
                    hide = not options.get("include_usage")
                    body["stream_options"] = dict(options, include_usage=True)
                return self.passthrough("POST", "/v1/chat/completions", body, hide)
            if path in ("/v1/messages", "/messages"):
                return self.handle_messages(body)
            if path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                text = json.dumps(body.get("messages", [])) + json.dumps(body.get("system", ""))
                return self.send_json(200, {"input_tokens": max(1, len(text) // 4)})
            if path in ("/v1/responses", "/responses"):
                return self.handle_responses(body)
            if path in ("/v1/completions", "/completions"):
                return self.passthrough("POST", "/v1/completions", body)
        except ValueError as error:
            return self.send_json(400, {"error": {"type": "invalid_request_error", "message": str(error)}})
        except (ConnectionError, OSError, http.client.HTTPException) as error:
            log(f"upstream error on {path}: {error}")
            return self.send_json(502, {"error": {"type": "api_error", "message": f"engine unavailable: {error}"}})
        self.send_json(404, {"error": {"type": "not_found", "message": path}})

    def passthrough(self, method, path, body, hide_usage=False):
        meter = Meter("chat", (body or {}).get("model") or MODEL_ALIAS) if path.endswith("/chat/completions") else None
        up = Upstream(method, path, body)  # returns once the engine has answered the headers: after the whole answer when not streamed
        if body and body.get("stream") and up.status == 200:
            self.start_sse()
            try:
                # forward the engine's SSE line by line, metering the deltas as they pass; the usage-only
                # chunk this gateway asked for is dropped when the client did not ask for it
                pending = b""
                while True:
                    chunk = up.response.read1(65536) if hasattr(up.response, "read1") else up.response.read(65536)
                    if not chunk:
                        break
                    pending += chunk
                    out = b""
                    while b"\n" in pending:
                        raw_line, pending = pending.split(b"\n", 1)
                        line = raw_line.strip()
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            try:
                                obj = json.loads(line[5:])
                            except json.JSONDecodeError:
                                obj = None
                            if isinstance(obj, dict):
                                if meter:
                                    meter.chunk(obj)
                                if hide_usage and obj.get("usage") and not obj.get("choices"):
                                    continue
                        out += raw_line + b"\n"
                    if out:
                        self.wfile.write(out)
                        self.wfile.flush()
                if pending:
                    self.wfile.write(pending)
                    self.wfile.flush()
            finally:
                up.conn.close()
            if meter:
                meter.done()
            return
        raw = up.raw()
        if meter and up.status == 200:
            try:
                meter.done(json.loads(raw).get("usage"))
            except (ValueError, AttributeError):
                meter.done()
        self.send_response(up.status)
        self.send_header("Content-Type", up.response.getheader("Content-Type") or "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def handle_messages(self, body):
        model = body.get("model") or MODEL_ALIAS
        chat = anthropic_to_chat(body)
        meter = Meter("messages", MODEL_ALIAS or model)
        if chat.get("stream"):
            chat["stream_options"] = {"include_usage": True}
            up = Upstream("POST", "/v1/chat/completions", chat)
            if up.status != 200:
                return self.send_json(up.status, {"error": {"type": "api_error", "message": up.raw().decode(errors="replace")[:500]}})
            self.start_sse()
            anthropic_stream(self, up, model, meter)
            return
        up = Upstream("POST", "/v1/chat/completions", chat)
        if up.status != 200:
            return self.send_json(up.status, {"error": {"type": "api_error", "message": up.raw().decode(errors="replace")[:500]}})
        resp = up.json()
        meter.done(resp.get("usage"))
        self.send_json(200, chat_to_anthropic(resp, model))

    def handle_responses(self, body):
        model = body.get("model") or MODEL_ALIAS
        chat = responses_to_chat(body)
        meter = Meter("responses", MODEL_ALIAS or model)
        if chat.get("stream"):
            chat["stream_options"] = {"include_usage": True}
            up = Upstream("POST", "/v1/chat/completions", chat)
            if up.status != 200:
                return self.send_json(up.status, {"error": {"type": "server_error", "message": up.raw().decode(errors="replace")[:500]}})
            self.start_sse()
            responses_stream(self, up, model, body, meter)
            return
        up = Upstream("POST", "/v1/chat/completions", chat)
        if up.status != 200:
            return self.send_json(up.status, {"error": {"type": "server_error", "message": up.raw().decode(errors="replace")[:500]}})
        resp = up.json()
        meter.done(resp.get("usage"))
        self.send_json(200, chat_to_responses(resp, model, body))


class Server(ThreadingHTTPServer):
    """A thread per connection, at most MAX_CONNECTIONS at once; one more is closed without a thread."""
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], OSError):  # a connection shut at its deadline or by its client
            super().handle_error(request, client_address)


def main():
    server = Server(("0.0.0.0", PORT), Handler)
    log(f"gateway listening on :{PORT}, engine {UPSTREAM}, key {'required' if required_key() else 'not set'}, usage log {USAGE_LOG or 'off'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
