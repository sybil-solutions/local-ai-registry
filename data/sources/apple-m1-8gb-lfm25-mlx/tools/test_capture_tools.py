#!/usr/bin/env python3
"""The capture readers read the committed gzipped evidence, and an empty or incomplete capture fails.

  python3 -m unittest test_capture_tools      (from this directory; standard library only)

Each case builds a small synthetic observer.py capture (a /health GET and one streamed chat completion) in a scratch
directory, plain and gzipped, and runs audit_capture.py and check_run.py as programs and count_stream_tokens.py's reader.
"""
import gzip, json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import count_stream_tokens  # noqa: E402  (its tokenizer import happens only in main)


def request(body):
    b = json.dumps(body).encode()
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8081\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b)


def streamed(pieces):
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
    events += [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in pieces]
    events += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: " + str(len(body.encode())).encode()
            + b"\r\n\r\n" + body.encode())


HEALTH = (b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8081\r\n\r\n", b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}")
PIECES = [{"reasoning": "Think. "}, {"content": "Hello"}, {"content": " there."}]


def capture(d, chat_body, gz):
    os.makedirs(d, exist_ok=True)
    for name, (req, resp) in {"conn-0000": HEALTH, "conn-0001": (request(chat_body), streamed(PIECES))}.items():
        for ext, data in ((".req", req), (".resp", resp)):
            if gz:
                with gzip.open(os.path.join(d, name + ext + ".gz"), "wb") as f:
                    f.write(data)
            else:
                with open(os.path.join(d, name + ext), "wb") as f:
                    f.write(data)
    return d


def run(tool, *args):
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, tool), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines(), r.stderr


CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.8, "stream": True}


class CaptureTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_gzipped_capture_reads_like_the_plain_one(self):
        plain, gz = capture(f"{self.d}/plain", CHAT, False), capture(f"{self.d}/gz", CHAT, True)
        for tool, args in (("audit_capture.py", ()), ("check_run.py", ("--expect", "1"))):
            rp, op, _ = run(tool, plain, *args)
            rg, og, _ = run(tool, gz, *args)
            self.assertEqual((rp, rg), (0, 0), tool)
            self.assertEqual(op, og, tool)  # the same rows and verdict from either form
            self.assertEqual(json.loads(og[-1])["verdict"], "pass", tool)
        self.assertEqual(json.loads(run("audit_capture.py", gz)[1][-1])["connections"], 2)

    def test_empty_capture_fails(self):
        os.makedirs(f"{self.d}/empty")
        rc, out, _ = run("audit_capture.py", f"{self.d}/empty")
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(out[-1])["verdict"], "fail")
        self.assertIn("empty capture", json.loads(out[-1])["problems"][0])
        rc, out, _ = run("check_run.py", f"{self.d}/empty", "--expect", "1")
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(out[-1])["chat_requests"], 0)

    def test_incomplete_or_doubled_capture_fails(self):
        d = capture(f"{self.d}/half", CHAT, True)
        os.remove(f"{d}/conn-0001.resp.gz")
        for tool in ("audit_capture.py", "check_run.py"):
            rc, _, err = run(tool, d)
            self.assertNotEqual(rc, 0, tool)
            self.assertIn("without its response", err, tool)
        d = capture(f"{self.d}/both", CHAT, True)
        capture(d, CHAT, False)
        for tool in ("audit_capture.py", "check_run.py"):
            rc, _, err = run(tool, d)
            self.assertNotEqual(rc, 0, tool)
            self.assertIn("both plain and gzipped", err, tool)

    def test_sampler_keys_fail_unless_allowed_and_limits_never(self):
        d = capture(f"{self.d}/seeded", {**CHAT, "seed": 7}, True)
        self.assertEqual(run("audit_capture.py", d)[0], 1)
        rc, out, _ = run("audit_capture.py", d, "--allow-sampler", "seed")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out[-1])["allowed_sampler_keys"], ["seed"])
        rc, _, err = run("audit_capture.py", d, "--allow-sampler", "max_tokens")
        self.assertNotEqual(rc, 0)
        self.assertIn("sampler-state keys only", err)
        capped = capture(f"{self.d}/capped", {**CHAT, "max_tokens": 64, "seed": 7}, True)
        rc, out, _ = run("audit_capture.py", capped, "--allow-sampler", "seed")
        self.assertEqual(rc, 1)
        self.assertIn("max_tokens", json.dumps(json.loads(out[-1])["problems"]))

    def test_an_unanswered_or_cut_stream_is_listed_and_fails_check_run(self):
        # a connection closed before any answer (empty response), and a stream the client closed mid-answer
        d = capture(f"{self.d}/cut", CHAT, True)
        full = streamed(PIECES)
        cut = full.replace(b"Content-Length: " + str(len(full.split(b"\r\n\r\n", 1)[1])).encode(), b"Transfer-Encoding: chunked")
        head, body = cut.split(b"\r\n\r\n", 1)
        chunked = head + b"\r\n\r\n" + b"%x\r\n" % len(body[:120]) + body[:120] + b"\r\n"  # no closing chunk: cut off
        for name, (req, resp) in {"conn-0002": (request(CHAT), b""), "conn-0003": (request(CHAT), chunked)}.items():
            for ext, data in ((".req", req), (".resp", resp)):
                with gzip.open(os.path.join(d, name + ext + ".gz"), "wb") as f:
                    f.write(data)
        rc, out, err = run("audit_capture.py", d)
        self.assertEqual(rc, 0, err)  # the requests carry no limit: the audit passes, and names both connections
        self.assertEqual(json.loads(out[-1])["unanswered_or_partial"], ["conn-0002", "conn-0003"])
        rows = {json.loads(l)["conn"]: json.loads(l) for l in out[:-1]}
        self.assertTrue(rows["conn-0002"]["response"].startswith("none"))
        self.assertTrue(rows["conn-0003"]["response"].startswith("partial"))
        capped = capture(f"{self.d}/cutcapped", {**CHAT, "max_tokens": 9}, True)
        with gzip.open(os.path.join(capped, "conn-0002.req.gz"), "wb") as f:
            f.write(request({**CHAT, "max_tokens": 9}))
        with gzip.open(os.path.join(capped, "conn-0002.resp.gz"), "wb") as f:
            f.write(b"")
        self.assertEqual(run("audit_capture.py", capped)[0], 1)  # a limit fails even on an unanswered request
        rc, out, _ = run("check_run.py", d, "--expect", "3")
        self.assertEqual(rc, 1)  # whether an answer ended on its own is check_run's verdict: these did not
        self.assertEqual(json.loads(out[-1])["failing"], ["conn-0002", "conn-0003"])
        status, text, events, texted, finish, partial = count_stream_tokens.stream(count_stream_tokens.response(d, "conn-0003"))
        self.assertEqual((status, partial, finish), (200, True, []))  # counted up to the cut, never as a finished answer
        self.assertLessEqual(len(text), len("Think. Hello there."))

    def test_token_counter_reads_gzipped_streams_and_counts_events(self):
        d = capture(f"{self.d}/gz", CHAT, True)
        status, text, events, texted, finish, _ = count_stream_tokens.stream(count_stream_tokens.response(d, "conn-0001"))
        self.assertEqual((status, text, events, texted, finish), (200, "Think. Hello there.", 5, 3, ["stop"]))
        with self.assertRaises(SystemExit):
            count_stream_tokens.response(d, "conn-0099")
        capture(d, CHAT, False)
        with self.assertRaises(SystemExit):
            count_stream_tokens.response(d, "conn-0001")


if __name__ == "__main__":
    unittest.main()
