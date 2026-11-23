"""lab.py's speed gate, reproduced: tokens/s over the first 30 s of a streamed answer (temperature 0.8)."""
import json, sys, time, urllib.request
sys.path.insert(0, "../../../lab")
import lab
tps, total, secs, *_ = lab.stream_rate("http://127.0.0.1:8080", "models/" + sys.argv[1], "Write a detailed 600-word story about a lighthouse keeper.")
print(json.dumps({"label": sys.argv[2], "tps": round(tps, 1), "tokens": total, "seconds": round(secs, 1)}))
