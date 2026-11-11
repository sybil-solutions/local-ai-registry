"""mlx_vlm.server for LFM3 MLX weights, with the routes the lab needs (mlx-vlm 0.7.12).

Usage: python3 lfm3_mlx_serve.py <mlx_vlm.server arguments>

mlx-lm has no LFM3 support yet (ml-explore/mlx-lm#2016), so LFM3 runs on mlx-vlm's server, which loads it as a text
model. This adds POST /v1/token/encode ({"text"} -> {"tokens", "length"}) from the model's own tokenizer, which
lab.py uses to count decoded tokens; lists only the served model on /v1/models, named as --model gives it (mlx-vlm
lists every model in the Hugging Face cache, and a client that picks the first one makes the server swap models);
and applies the model's generation_config sampling (temperature, top_k, top_p, repetition_penalty) to requests that
leave those fields out, as vLLM and SGLang do by default. mlx-vlm otherwise samples with top_k 0 and no repetition
penalty. Everything else is mlx_vlm.server unchanged.
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


def add_lab_routes(model_path):
    from mlx_vlm.server import app
    from tokenizers import Tokenizer

    # the tokenizer file itself: AutoTokenizer would read the repo's config, whose auto_map asks to run the repo's code
    tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
    app.add_middleware(ServedModelOnly, model_id=model_path)

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


def add_sampling_defaults(model_path):
    config = Path(model_path) / "generation_config.json"
    defaults = {k: v for k, v in json.loads(config.read_text()).items() if k in SAMPLING_KEYS} if config.exists() else {}
    if defaults:
        from mlx_vlm.server import app

        app.add_middleware(SamplingDefaults, defaults=defaults)


if __name__ == "__main__":
    model_path = sys.argv[sys.argv.index("--model") + 1]
    add_lab_routes(model_path)
    add_sampling_defaults(model_path)
    from mlx_vlm.server.cli import main

    sys.argv[0] = "mlx_vlm.server"
    main()
