"""The recorded run's exact server (launcher as __main__) plus the repo's memserve.py memory monitor.

Identical server behavior to the recorded run's launch (~/ai-work/m4max/run_4bit.sh running
run/qwen38_mlx_serve.py directly): launcher.install() + add_lab_routes() + add_sampling_defaults()
+ mlx_vlm.server.cli main(). Added on top, and only on top: memserve.py's once-per-second
MLX memory log (t/active_gb/peak_gb/cache_gb) appended to $MEMLOG.

$LAUNCHER = the frozen launcher (registry/launches/mlx-vlm-qwen3.8-long-prefill.py,
sha256 f8cd617060c26a5618d75a314931717779a72848f443c18a645e1db2d165f94a, which equals
launch_config_sha256 in the recorded run).
"""
import importlib.util
import json
import os
import sys
import threading
import time

import mlx.core as mx

sys.dont_write_bytecode = True  # keep registry/launches/ free of __pycache__

spec = importlib.util.spec_from_file_location("qwen38_mlx_serve", os.environ["LAUNCHER"])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.install()

model_path = sys.argv[sys.argv.index("--model") + 1]
launcher.add_lab_routes(model_path)
launcher.add_sampling_defaults(model_path)


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


threading.Thread(target=monitor, args=(os.environ["MEMLOG"],), daemon=True).start()

from mlx_vlm.server.cli import main  # noqa: E402

sys.argv[0] = "mlx_vlm.server"
main()