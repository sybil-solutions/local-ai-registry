"""mlx_lm.server for Inkling 2 MLX weights, with the routes and parsing the lab needs.

Usage: python3 inkling2_mlx_serve.py <mlx_lm.server arguments>

mlx-lm (ml-explore/mlx-lm#2086) loads Inkling 2 but does not know its response format, so this adds it:

  <|message_model|><|content_thinking|>REASONING<|end_message|>
  <|message_model|><|content_text|>CONTENT<|end_message|>
  <|message_model|>NAME<|content_invoke_tool_json|>{"name": NAME, "args": {...}}<|end_message|>
  <|content_model_end_sampling|>   (EOS)

Thinking goes to `reasoning`, text to `content`, and each invoke_tool_json message becomes an OpenAI tool call.
INKLING_FAST (see apply_fast_paths) fuses the decode step's small elementwise chains: about +15-20% decode on an
M4 Max, with prefill unchanged. Also adds POST /v1/token/encode ({"text"} -> {"tokens", "length"}), which lab.py
uses to count decoded tokens, lists only the served model on /v1/models, and applies the model's
generation_config sampling (temperature) to requests that leave it out. Everything else is mlx_lm.server unchanged.
"""
import functools
import io
import json
import sys
from pathlib import Path

from mlx_lm import server, utils

SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")
MODEL, THINK, TEXT, TOOL, END, EOS = (
    "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>",
    "<|end_message|>", "<|content_model_end_sampling|>",
)
RESPONSE_FORMAT = {
    "initial_state": "normal",
    "transitions": {
        "normal": [(MODEL, "header"), (THINK, "reasoning"), (TEXT, "normal"), (TOOL, "tool"), (END, "normal"),
                   (EOS, "stop")],
        "header": [(THINK, "reasoning"), (TEXT, "normal"), (TOOL, "tool")],  # header text (a tool's name) is dropped
        "reasoning": [(END, "normal"), (EOS, "stop")],
        "tool": [(END, "normal")],
    },
}


def parse_tool_call(text, tools=None):
    call = json.loads(text)
    return {"name": call["name"], "arguments": call.get("args") or call.get("arguments") or {}}


def with_inkling_format(tokenizer):
    """Give a TokenizerWrapper Inkling 2's response format and tool parser."""
    tokenizer._response_format = RESPONSE_FORMAT
    tokenizer._tool_parser = parse_tool_call
    tokenizer._tool_call_start = TOOL
    tokenizer._tool_call_start_tokens = tuple(tokenizer.encode(TOOL, add_special_tokens=False))
    tokenizer._tool_call_end = None  # the response format closes the call on END
    tokenizer._tool_call_end_tokens = None
    return tokenizer


def apply_fast_paths(which):
    """Fuse Inkling 2's small elementwise chains with mx.compile, to cut Metal dispatches per token.

    `which` is a set of names (INKLING_FAST="gates,mask,moe,conv,attn,gates3", or "all"). Every patch runs only on
    decode steps (one new token per sequence); prefill keeps mlx-lm's reference code. gates, mask, moe, conv and attn
    compute the same ops in the same dtypes as the reference, fused into fewer Metal kernels (greedy decode is
    bit-identical). gates3 batches the q/k/v message gates of a looped layer and concatenates the forget and message
    projections: same math, different accumulation order (KL 1.5e-4 nats on one step, 30x under the run-to-run
    numerics of a different prefill step size).
    """
    import math
    from functools import partial

    import mlx.core as mx

    from mlx_lm.models.inkling2 import layers as L

    if "all" in which:
        which = {"gates", "mask", "moe", "conv", "attn", "gates3"}

    def decode_only(cls, name, fast_fn, seq_len):
        """Install fast_fn for decode steps (one new token); prefill keeps mlx-lm's reference code, bit for bit."""
        orig = getattr(cls, name)

        def dispatch(self, *a, **k):
            return fast_fn(self, *a, **k) if seq_len(*a) == 1 else orig(self, *a, **k)

        setattr(cls, name, dispatch)

    if "gates" in which:
        @partial(mx.compile, shapeless=True)
        def gate_fn(h, m, wd, wf, wm):
            s = mx.concatenate([mx.fast.rms_norm(h, None, 1e-6), mx.fast.rms_norm(m, None, 1e-6)], axis=-1)
            s = (s.astype(wd.dtype) @ wd.T).astype(mx.float32)
            f = mx.tanh(s @ wf.T)
            g = mx.tanh(s @ wm.T)
            return (1 - f) * h + g * m

        def gate_call(self, hidden_states, message):
            return gate_fn(hidden_states, message, self.down_proj.weight, self.forget_gate_proj.weight,
                           self.message_gate_proj.weight)

        decode_only(L.MessageGate, "__call__", gate_call, lambda h, m: h.shape[-2])

    if "mask" in which:
        @partial(mx.compile, shapeless=True)
        def decode_mask(qpos, kpos, bias_row, window, dtype_probe):
            distance = qpos.reshape(1) - kpos.reshape(-1)
            R = bias_row.shape[-1]
            bias = mx.take(bias_row, mx.clip(distance, 0, R - 1), axis=-1)
            bias = mx.where(distance < R, bias, mx.zeros_like(bias))
            visible = (distance >= 0) & (kpos.reshape(-1) >= 0)
            visible = visible & ((window <= 0) | (distance < window))
            return mx.where(visible, bias, -mx.inf).astype(dtype_probe.dtype)

        orig_attention = L.Attention._attention

        def attention(self, query, key, value, position_bias, sinks, cache):
            B, _, Lq, _ = query.shape
            if not (Lq == 1 and B == 1):
                return orig_attention(self, query, key, value, position_bias, sinks, cache)
            S = key.shape[2]
            end = cache.offset if cache is not None else Lq
            qpos = L._positions(end - Lq, Lq)
            kpos = L._key_positions(cache, S, Lq, end)
            mask = decode_mask(qpos, kpos, position_bias[0, 0], mx.array(self.sliding_window or 0),
                               mx.zeros((1,), query.dtype))
            out = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scaling,
                                                       mask=mask[None, :, None, :],
                                                       sinks=sinks[0, 0].astype(query.dtype))
            return out.transpose(0, 2, 1, 3)

        L.Attention._attention = attention

    if "moe" in which:
        @partial(mx.compile, shapeless=True)
        def combine(expert_states, routing_weights, probe):
            return (expert_states.astype(mx.float32) * routing_weights[..., None]).sum(axis=-2).astype(probe.dtype)

        def moe_call(self, hidden_states):
            if self.sharding_group is not None:
                return orig_moe(self, hidden_states)
            selected, weights = self.gate(hidden_states)
            return combine(self.experts(hidden_states, selected), weights, hidden_states[..., :1])

        orig_moe = L.SparseMoeBlock.__call__
        decode_only(L.SparseMoeBlock, "__call__", moe_call, lambda h: h.shape[-2])

    if "conv" in which:
        @partial(mx.compile, shapeless=True)
        def conv_out(out, residual, inv_sqrt, probe):
            return ((out.astype(mx.float32) + residual) * inv_sqrt).astype(probe.dtype)

        def short_conv(self, hidden_states, conv_state, conv_mask):
            K = self.kernel_size
            residual = hidden_states.astype(mx.float32)
            window = L._conv_input(hidden_states, conv_state, conv_mask, K)
            out = self.conv1d(window.astype(self.conv1d.weight.dtype))
            return conv_out(out, residual, mx.array(1 / math.sqrt(K + 1), mx.float32), hidden_states[..., :1]), window

        decode_only(L.ShortConvolution, "__call__", short_conv, lambda h, *_: h.shape[1])

    if "attn" in which:
        @partial(mx.compile, shapeless=True)
        def gated_out(attn_output, gate):
            return (attn_output.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype)

        @partial(mx.compile, shapeless=True)
        def value_norm(v):
            return mx.fast.rms_norm(v.astype(mx.float32), None, 1e-6).astype(v.dtype)

        def attend(self, hidden_states, query_states, key_states, value_states, relative_states, cache, tau=None):
            B, Lq, _ = hidden_states.shape
            gate = self.gate_proj(hidden_states)
            sinks = self.sink_proj(hidden_states)
            query_states = query_states.reshape(B, Lq, self.num_heads, -1)
            key_states = key_states.reshape(B, Lq, self.num_key_value_heads, -1)
            value_states = value_states.reshape(B, Lq, self.num_key_value_heads, -1)
            relative_states = relative_states.reshape(B, Lq, self.num_heads, -1)
            query_states = self.q_l2_clip(query_states)
            key_states = self.k_l2_clip(key_states)
            value_states = value_norm(value_states) if self.eps == 1e-6 else L.unweighted_rms_norm(
                value_states.astype(mx.float32), self.eps).astype(value_states.dtype)
            relative_states = self.rel_l2_clip(relative_states)
            if tau is not None:
                query_states = (query_states.astype(mx.float32) * tau).astype(query_states.dtype)
                relative_states = (relative_states.astype(mx.float32) * tau).astype(relative_states.dtype)
            position_bias = self.rel_logits_proj(relative_states)
            query_states = query_states.transpose(0, 2, 1, 3)
            key_states = key_states.transpose(0, 2, 1, 3)
            value_states = value_states.transpose(0, 2, 1, 3)
            if cache is not None:
                key_states, value_states = cache.update_and_fetch(key_states, value_states)
            attn_output = self._attention(query_states, key_states, value_states, position_bias, sinks, cache)
            attn_output = attn_output.reshape(B, Lq, -1)
            return self.o_proj(gated_out(attn_output, gate))

        decode_only(L.Attention, "_attend", attend, lambda h, *_, **__: h.shape[1])

    if "gates3" in which:
        # The q, k and v message gates of a looped layer as one batched set of kernels; forget and message
        # projections concatenated into one matmul.
        @mx.compile
        def gates3_fn(h, m, wd, wfm):
            s = mx.concatenate([mx.fast.rms_norm(h, None, 1e-6), mx.fast.rms_norm(m, None, 1e-6)], axis=-1)
            s = (s.astype(wd.dtype)[..., None, :] @ wd.swapaxes(-1, -2)).astype(mx.float32)
            fg = mx.tanh(s @ wfm.swapaxes(-1, -2))[..., 0, :]
            D = h.shape[-1]
            return (1 - fg[..., :D]) * h + fg[..., D:] * m

        def mix(self, states, messages):
            g3 = self.__dict__.get("_g3")
            if g3 is None:
                gs = (self.q_message_gate, self.k_message_gate, self.v_message_gate)
                wd = mx.stack([g.down_proj.weight for g in gs])
                wfm = mx.stack([mx.concatenate([g.forget_gate_proj.weight, g.message_gate_proj.weight]) for g in gs])
                mx.eval(wd, wfm)
                g3 = (wd, wfm)
                object.__setattr__(self, "_g3", g3)
            out = gates3_fn(mx.stack(states, axis=-2), messages[..., :3, :], *g3)
            return tuple(out[..., i, :] for i in range(3))

        decode_only(L.DecoderLayer, "_mix_messages", mix, lambda states, m: states[0].shape[-2])


def install(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = Path(model_path) / "generation_config.json"
    defaults = {k: v for k, v in json.loads(config.read_text()).items() if k in SAMPLING_KEYS} if config.exists() else {}

    load_tokenizer = utils.load_tokenizer

    def patched_load_tokenizer(*a, **kw):
        return with_inkling_format(load_tokenizer(*a, **kw))

    utils.load_tokenizer = patched_load_tokenizer

    class LabHandler(server.APIHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/v1/models", "/models"):
                return super().do_GET()
            out = json.dumps({"object": "list", "data": [{"id": model_path, "object": "model", "created": self.created}]}).encode()
            self._set_completion_headers(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            if self.path == "/v1/token/encode":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                ids = tokenizer.encode(body.get("text") or "", add_special_tokens=False)
                out = json.dumps({"tokens": ids, "length": len(ids)}).encode()
                self._set_completion_headers(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if defaults and self.path in ("/v1/chat/completions", "/chat/completions", "/v1/completions"):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                try:
                    data = json.loads(body)
                    if isinstance(data, dict):
                        body = json.dumps({**defaults, **data}).encode()
                except ValueError:
                    pass
                self.headers.replace_header("Content-Length", str(len(body)))
                self.rfile = io.BytesIO(body)
            return super().do_POST()

    server._run_http_server = functools.partial(server._run_http_server, handler_class=LabHandler)


if __name__ == "__main__":
    import os

    fast = {s for s in os.environ.get("INKLING_FAST", "").split(",") if s}
    if fast:
        apply_fast_paths(fast)
    install(sys.argv[sys.argv.index("--model") + 1])
    sys.argv[0] = "mlx_lm.server"
    server.main()
