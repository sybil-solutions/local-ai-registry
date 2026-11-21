"""Quick local versions of the registry's chat / reasoning / tools gates (non-streaming + streaming)."""
import json, sys, urllib.request
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
M = json.load(urllib.request.urlopen(B + "/v1/models"))["data"][0]["id"]
def post(body):
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps({"model": M, **body}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
r = post({"messages": [{"role": "user", "content": "Name three primary colors, comma separated."}], "max_tokens": 4096})
c = r["choices"][0]; print("CHAT", c["finish_reason"], repr(c["message"].get("content"))[:200], "| reasoning chars", len(c["message"].get("reasoning") or ""))
r = post({"messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 4096})
c = r["choices"][0]; print("REASON", c["finish_reason"], repr(c["message"].get("content"))[:200], "| reasoning:", repr((c["message"].get("reasoning") or "")[:120]))
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
r = post({"messages": msgs, "tools": tools, "max_tokens": 4096})
m = r["choices"][0]["message"]; print("TOOL1", r["choices"][0]["finish_reason"], m.get("tool_calls"), repr(m.get("content"))[:100])
if m.get("tool_calls"):
    tc = m["tool_calls"][0]
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}}]},
             {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"temp_c": 14, "conditions": "light rain", "code": "WX-4417"})}]
    r = post({"messages": msgs, "tools": tools, "max_tokens": 4096})
    print("TOOL2", r["choices"][0]["finish_reason"], repr(r["choices"][0]["message"].get("content"))[:300])
