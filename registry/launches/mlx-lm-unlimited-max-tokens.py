#!/usr/bin/env python3
"""mlx_lm.server 0.31.3 with `--max-tokens -1` meaning unlimited, as mlx_lm's own generator defines it, a KV bound and a
prompt limit.

Stock mlx_lm.server rejects max_tokens -1 (validation min 0) and its BatchGenerator would stop a -1 request after one
token. This launcher patches the server in memory only:
  1. validation accepts exactly -1 (anything below -1 is still rejected), and -1 with a draft model is rejected;
  2. a -1 request is served by the existing sequential path (stream_generate -> generate_step, where -1 is infinite);
  3. `--max-kv-size N` bounds each attention layer's KV cache to N tokens, the prompt and the answer together.
     make_prompt_cache ignores max_kv_size for a model with its own make_cache (LFM2: a growing KVCache per attention
     layer), so the sequential path uses _bounded_prompt_cache, which turns each KVCache into a RotatingKVCache of N
     tokens (keep=4), as make_prompt_cache does for other models. Once prompt plus answer passes N, the oldest tokens
     after the first 4 are dropped, so the rest of that answer no longer sees the start of the prompt;
  4. `--max-prompt-tokens P` answers a request whose tokenized prompt is longer than P with HTTP 400, before any prefill
     or generation (stock answers every error raised there with 404, and still does for everything else).
With either flag set, every request is served on the sequential path, where the bounded cache is made and the prompt is
measured. Without them nothing changes. The patch does not bound the prompt cache the server keeps between requests
(stock --prompt-cache-size does) and does not change the sampler code; it does move unseeded requests from the batch
path to the sequential one when a bound is set. HTTP API: a request may send max_tokens -1.

It refuses to run unless the imported mlx_lm package is byte-identical to the mlx-lm 0.31.3 wheel from PyPI
(sha256 758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8) and the patched server matches its hash.
Nothing on disk is modified.

  python3 mlx_lm_unlimited_serve.py --model <repo id> --max-tokens -1 --max-kv-size 32768 --max-prompt-tokens 24576 \
      --prompt-cache-size 1 --host 127.0.0.1 --port 8080
  python3 mlx_lm_unlimited_serve.py --verify-only     # all checks, without importing mlx
"""
import hashlib, importlib.util, os, sys, types

PRISTINE_PACKAGE = "8089a1a4434f6e9ec1be6502d7034bb9dd8774c34954924ff7d68bbdfd2822a4"  # every file of mlx_lm in the wheel
PRISTINE_SERVER = "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b"
PATCHED_SERVER = "9158b98e4aa52193ed0d8558eea912bfabe5f2939eec67e68f5e10e8e50d7f17"
EDITS = [
    ("""    def _is_batchable(self, args):
        return self.model_provider.is_batchable and args.seed is None
""", """    def _is_batchable(self, args):
        # max_tokens -1 (unlimited) is served sequentially: stream_generate/generate_step treat -1 as infinite,
        # while BatchGenerator stops at num_tokens >= max_tokens; with a KV bound or a prompt limit every request is
        # served sequentially, where the bounded cache is made and the prompt is measured
        return (self.model_provider.is_batchable and args.seed is None and args.max_tokens != -1
                and self.cli_args.max_kv_size is None and self.cli_args.max_prompt_tokens is None)
"""),
    ("""        self._validate("max_tokens", int, min_val=0)
""", """        self._validate("max_tokens", int, min_val=-1)  # -1 = unlimited (generate_step's own convention)
        if self.max_tokens == -1 and (
            self.requested_draft_model != "default_model"
            or self.response_generator.cli_args.draft_model is not None
        ):
            raise ValueError("max_tokens -1 (unlimited) is not supported with a draft model")
"""),
    ("""from .models.cache import (
    LRUPromptCache,
    make_prompt_cache,
)
""", """from .models.cache import (
    KVCache,
    LRUPromptCache,
    RotatingKVCache,
    make_prompt_cache,
)
"""),
    ("""

def get_system_fingerprint():
""", """

def _bounded_prompt_cache(model, max_kv_size):
    # make_prompt_cache ignores max_kv_size when the model has its own make_cache (LFM2: a growing KVCache per attention
    # layer). With a bound, each KVCache becomes a RotatingKVCache of max_kv_size tokens (keep=4, as make_prompt_cache
    # uses for models without make_cache); fixed-size caches such as ArraysCache are kept.
    cache = make_prompt_cache(model, max_kv_size)
    if max_kv_size is None:
        return cache
    return [RotatingKVCache(max_size=max_kv_size, keep=4) if type(c) is KVCache else c for c in cache]


class PromptTooLong(ValueError):
    pass  # a prompt longer than --max-prompt-tokens: answered with HTTP 400 before any prefill or generation


def _check_prompt_length(n_tokens, max_prompt_tokens):
    if max_prompt_tokens is not None and n_tokens > max_prompt_tokens:
        raise PromptTooLong(
            f"the prompt is {n_tokens} tokens; this server accepts at most {max_prompt_tokens} (--max-prompt-tokens)"
        )


def _error_status(e):
    # an over-long prompt is the client's to fix (400); every other error keeps the stock status
    return 400 if isinstance(e, PromptTooLong) else 404


def get_system_fingerprint():
"""),
    ("""            # Prepare the prompt and state machine
            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
""", """            # Prepare the prompt and state machine
            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
            _check_prompt_length(len(prompt), self.cli_args.max_prompt_tokens)  # before the context, so the client gets 400
"""),
    ("""        except Exception as e:
            self._set_completion_headers(404)
""", """        except Exception as e:
            self._set_completion_headers(_error_status(e))
"""),
    ("""                cache = make_prompt_cache(self.model_provider.model)
""", """                cache = _bounded_prompt_cache(self.model_provider.model, self.cli_args.max_kv_size)
"""),
    ("""    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
    )
""", """    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Bound each attention layer's KV cache, prompt and answer together, to this many tokens (RotatingKVCache)",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Answer a request whose prompt is longer than this many tokens with HTTP 400, before any generation",
    )
"""),
]


def package_digest(pkg):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(pkg):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, pkg).encode() + b"\0" + hashlib.sha256(open(p, "rb").read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def patched_server_source():
    spec = importlib.util.find_spec("mlx_lm")  # locates the package without executing it
    pkg = os.path.dirname(spec.origin)
    got = package_digest(pkg)
    if got != PRISTINE_PACKAGE:
        sys.exit(f"mlx_lm at {pkg} is not the pristine 0.31.3 wheel (package digest {got})")
    path = os.path.join(pkg, "server.py")
    text = open(path, encoding="utf-8").read()
    assert hashlib.sha256(text.encode()).hexdigest() == PRISTINE_SERVER
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit("patch context not found exactly once")
        text = text.replace(old, new)
    if hashlib.sha256(text.encode()).hexdigest() != PATCHED_SERVER:
        sys.exit("patched server does not match its pinned hash")
    return pkg, path, text


def main():
    pkg, path, text = patched_server_source()
    if sys.argv[1:] == ["--verify-only"]:
        print(f"ok: pristine mlx_lm 0.31.3 at {pkg}; patched server sha256 {PATCHED_SERVER}")
        return 0
    import mlx_lm  # the verified package
    mod = types.ModuleType("mlx_lm.server")
    mod.__file__, mod.__package__ = path, "mlx_lm"
    mod.__spec__ = importlib.util.spec_from_file_location("mlx_lm.server", path)
    sys.modules["mlx_lm.server"] = mod
    exec(compile(text, path, "exec"), mod.__dict__)
    sys.argv[0] = "mlx_lm.server"
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
