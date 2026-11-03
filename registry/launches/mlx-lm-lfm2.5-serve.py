"""mlx_lm.server for LFM2.5 MLX weights, with the routes the lab needs (mlx-lm 0.32.2).

Usage: python3 lfm25_mlx_serve.py <mlx_lm.server arguments>

Adds POST /v1/token/encode ({"text"} -> {"tokens", "length"}) from the model's own tokenizer, which lab.py uses to
count decoded tokens (mlx_lm.server has no tokenize route). Lists only the served model on /v1/models, named as
--model gives it: mlx_lm.server lists every MLX model in the Hugging Face cache first and resolves the --model path to
an absolute one, so a client that picks the first entry makes the server load another model. Applies the model's
generation_config sampling (temperature, top_k, top_p, repetition_penalty) to requests that leave those fields out,
as vLLM and SGLang do by default. Everything else is mlx_lm.server unchanged.
"""
import functools
import io
import json
import sys
from pathlib import Path

from mlx_lm import server

SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")


def install(model_path):
    """Swap mlx_lm.server's request handler for one with the token route and the model's sampling defaults."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = Path(model_path) / "generation_config.json"
    defaults = {k: v for k, v in json.loads(config.read_text()).items() if k in SAMPLING_KEYS} if config.exists() else {}

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
                self.rfile = io.BytesIO(body)  # HTTP/1.0: one request per connection, nothing follows the body
            return super().do_POST()

    # server.run does not pass its handler_class on, so the handler goes in one level down
    server._run_http_server = functools.partial(server._run_http_server, handler_class=LabHandler)


if __name__ == "__main__":
    install(sys.argv[sys.argv.index("--model") + 1])
    sys.argv[0] = "mlx_lm.server"
    server.main()
