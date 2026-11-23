"""The recipe's launcher run as __main__, plus a once-per-second MLX memory log (t/active_gb/peak_gb/cache_gb, GiB).

Measurement wrapper only; the recipe launches the launcher directly. LAUNCHER=<path> is the frozen launcher
(registry/launches/mlx-lm-lfm2.5-serve.py), MEMLOG=<path.jsonl> the log. Touch <MEMLOG>.reset to reset the peak.
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
            log.write(json.dumps({"t": time.time(), "active_gb": mx.get_active_memory() / 2**30,
                                  "peak_gb": mx.get_peak_memory() / 2**30, "cache_gb": mx.get_cache_memory() / 2**30}) + "\n")
            time.sleep(1)


sys.dont_write_bytecode = True  # keep registry/launches/ free of __pycache__
spec = importlib.util.spec_from_file_location("lfm25_mlx_serve", os.environ["LAUNCHER"])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.install(sys.argv[sys.argv.index("--model") + 1])

threading.Thread(target=monitor, args=(os.environ["MEMLOG"],), daemon=True).start()

sys.argv[0] = "mlx_lm.server"
launcher.server.main()
