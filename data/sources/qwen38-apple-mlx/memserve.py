"""Run the recipe launcher in-process and log MLX GPU memory once a second.

Measurement wrapper only; the recipe launches the launcher directly.
MEMLOG=<path.jsonl> is required (values in GiB). LAUNCHER=<path> is the launcher (registry/launches/).
Touch <MEMLOG>.reset to reset the peak counter.
"""

import importlib.util
import json
import os
import sys
import threading
import time

import mlx.core as mx


def monitor(path):
    reset = path + ".reset"
    with open(path, "a", buffering=1) as log:
        while True:
            if os.path.exists(reset):
                os.remove(reset)
                mx.reset_peak_memory()
            log.write(json.dumps({
                "t": time.time(),
                "active_gb": mx.get_active_memory() / 2**30,
                "peak_gb": mx.get_peak_memory() / 2**30,
                "cache_gb": mx.get_cache_memory() / 2**30,
            }) + "\n")
            time.sleep(1)


sys.dont_write_bytecode = True  # keep registry/launches/ free of __pycache__
spec = importlib.util.spec_from_file_location("qwen38_mlx_serve", os.environ["LAUNCHER"])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.install()

threading.Thread(target=monitor, args=(os.environ["MEMLOG"],), daemon=True).start()

from mlx_vlm.server.cli import main  # noqa: E402

sys.argv[0] = "mlx_vlm.server"
main()
