"""mlx_vlm.server for Inkling 2 Mini, with the routes the lab needs.

Usage: python3 inkling_mlx_serve.py <mlx_vlm.server arguments>

Adds POST /v1/token/encode ({"text"} -> {"tokens", "length"}) from the model's own tokenizer, which lab.py uses to
count decoded tokens (mlx_vlm.server has no tokenize route); lists only the served model on /v1/models, named as
--model gives it (mlx-vlm lists every model in the Hugging Face cache, and a client that picks the first one makes
the server swap models); and applies the model's generation_config sampling to requests that leave those fields
out. With --text-only (this script's own flag), drops the vision tower after loading and refuses image input, which
frees 0.64 GB of Metal memory for the KV cache. Everything else is mlx_vlm.server unchanged.
"""

import json
import sys
from pathlib import Path


class ServedModelOnly:
    """ASGI middleware: GET /v1/models lists the served model only."""

    def __init__(self, app, model_id):
        self.app = app
        self.body = json.dumps({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}).encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "GET" or scope["path"] not in ("/v1/models", "/models"):
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(self.body)).encode())]})
        await send({"type": "http.response.body", "body": self.body})


def resolve(model):
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def add_lab_routes(model_id, model_dir):
    from mlx_vlm.server import app
    from tokenizers import Tokenizer

    # the tokenizer file itself, so nothing in the repo's config runs
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    app.add_middleware(ServedModelOnly, model_id=model_id)

    @app.post("/v1/token/encode")
    async def encode(body: dict):
        ids = tokenizer.encode(body.get("text") or "", add_special_tokens=False).ids
        return {"tokens": ids, "length": len(ids)}


SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")
GENERATION_PATHS = {"/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses", "/v1/messages", "/messages"}


class SamplingDefaults:
    """ASGI middleware: fill in the model's sampling settings a request body leaves out; never override one it sets."""

    def __init__(self, app, defaults):
        self.app, self.defaults = app, defaults

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in GENERATION_PATHS:
            return await self.app(scope, receive, send)
        body, more = b"", True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                body = json.dumps({**self.defaults, **data}).encode()
        except ValueError:
            pass
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        headers = [(k, v) for k, v in scope["headers"] if k != b"content-length"] + [(b"content-length", str(len(body)).encode())]
        return await self.app({**scope, "headers": headers}, replay, send)


def add_sampling_defaults(model_dir):
    config = model_dir / "generation_config.json"
    defaults = {k: v for k, v in json.loads(config.read_text()).items() if k in SAMPLING_KEYS} if config.exists() else {}
    if defaults:
        from mlx_vlm.server import app

        app.add_middleware(SamplingDefaults, defaults=defaults)


def drop_vision_tower():
    """Wrap the server's model loader so the loaded model keeps no vision weights."""
    import gc

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.server import generation

    class NoVision(nn.Module):
        def __call__(self, *args, **kwargs):
            raise ValueError("this server runs Inkling text-only (--text-only); image input is not supported")

    load = generation.load

    def load_text_only(*args, **kwargs):
        model, processor = load(*args, **kwargs)
        if "vision_tower" in model:
            model.vision_tower = NoVision()
            gc.collect()
            mx.clear_cache()
        return model, processor

    generation.load = load_text_only


if __name__ == "__main__":
    if "--text-only" in sys.argv:
        sys.argv.remove("--text-only")
        drop_vision_tower()
    model_id = sys.argv[sys.argv.index("--model") + 1]
    model_dir = resolve(model_id)
    add_lab_routes(model_id, model_dir)
    add_sampling_defaults(model_dir)
    from mlx_vlm.server.cli import main

    sys.argv[0] = "mlx_vlm.server"
    main()
