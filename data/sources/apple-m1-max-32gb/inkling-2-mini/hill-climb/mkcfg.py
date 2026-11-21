# mkcfg.py NAME [llama-server args...] -> configs/NAME.json for bench.py (launch via launch.sh, stop via stop.sh).
import json, sys
W = "/tmp/inkling-skill/work"
name, args = sys.argv[1], sys.argv[2:]
json.dump({"name": name, "base_url": "http://127.0.0.1:8090", "model": "inkling",
           "launch": ["bash", f"{W}/launch.sh", name, *args], "stop": ["bash", f"{W}/stop.sh"],
           "health_timeout_s": 300, "concurrency": 2, "prefill_lengths": [4096, 16384],
           "temperature": 1.0, "require_ac": True,
           "memory": {"cap_gib": 24, "probe": ["bash", f"{W}/mem.sh"], "interval_s": 2}},
          open(f"{W}/configs/{name}.json", "w"), indent=1)
