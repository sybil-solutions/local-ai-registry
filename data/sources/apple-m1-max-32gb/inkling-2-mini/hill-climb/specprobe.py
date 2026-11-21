# specprobe.py: rewrite-heavy and novel prompts, greedy, thinking off; prints decode tok/s and draft acceptance.
import json, urllib.request, sys
code = open("/tmp/inkling-skill/skill/skills/inference-optimisation/scripts/bench.py").read()[:6000]
for name, prompt in [("rewrite", "Repeat the following code exactly, but rename the function `log` to `logline` everywhere. Output only the code.\n\n" + code),
                     ("novel", "Write a short story (about 400 words) about a lighthouse keeper.")]:
    body = {"model": "inkling", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})))
    tm = d.get("timings", {}); m = d["choices"][0]["message"]
    print(sys.argv[1] if len(sys.argv) > 1 else "", name, d["usage"]["completion_tokens"], "tok", round(tm.get("predicted_per_second", 0), 1), "tok/s",
          "draft", tm.get("draft_n"), "acc", tm.get("draft_n_accepted"), "| reasoning", len(m.get("reasoning_content") or ""), "| content", repr((m.get("content") or "")[:80]))
