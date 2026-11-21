# eqprobe.py TAG: greedy, thinking off, 1500-token code rewrite; saves output to eq-TAG.txt and prints speed/acceptance.
import json, urllib.request, sys
code = open("/tmp/inkling-skill/skill/skills/inference-optimisation/scripts/bench.py").read()[:6000]
body = {"model": "inkling", "messages": [{"role": "user", "content": "Repeat the following code exactly, but rename the function `log` to `logline` everywhere. Output only the code.\n\n" + code}],
        "max_tokens": 1500, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"}}
d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})))
m = d["choices"][0]["message"]; tm = d["timings"]
open(f"/tmp/inkling-skill/work/eq-{sys.argv[1]}.txt", "w").write((m.get("reasoning_content") or "") + "\n=====\n" + (m.get("content") or ""))
print(sys.argv[1], d["usage"]["completion_tokens"], "tok", round(tm["predicted_per_second"], 2), "tok/s draft", tm.get("draft_n"), "acc", tm.get("draft_n_accepted"), "finish", d["choices"][0]["finish_reason"])
