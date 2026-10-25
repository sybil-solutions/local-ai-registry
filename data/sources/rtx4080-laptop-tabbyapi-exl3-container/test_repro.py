#!/usr/bin/env python3
"""CPU-only fixtures for repro.sh with a fake runtime (Linux: repro.sh fences helpers through /proc).

  python3 data/sources/rtx4080-laptop-tabbyapi-exl3-container/test_repro.py

A stub `podman` and `nvidia-smi` go first on PATH; the stub's "container" is a fake TabbyAPI that answers
the unchanged stock lab (lab/lab.py from this checkout, run for real) correctly, or with a chosen fault.
Nothing touches a GPU, a real container or the network. Each test runs repro.sh in a scratch copy of the
checkout and checks its exit code, that its own container is positively gone, that a foreign container is
never touched, and that no helper or server process is left behind.
"""
import http.server, json, os, shutil, signal, socket, socketserver, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REL = os.path.relpath(HERE, ROOT)
SERVED = "Qwen3.5-9B-EXL3-6hb-4bpw"
DIGEST = "sha256:0f83e6198dc3be2561652d8df2525a7d1a69733e12fc8e25cbc5bc319a8a3ad1"


# ----------------------------------------------------------------------------- the fake TabbyAPI
def serve(port, faults):
    streams = [0]

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            print("fake-server", format % args, flush=True)

        def send(self, code, body):
            b = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if "never_healthy" in faults:
                return self.send(503, {"detail": "loading"})
            self.send(200, {"object": "list", "data": [{"id": SERVED}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if self.path == "/v1/token/encode":
                return self.send(200, {"length": max(1, len(req.get("text", "")) // 4)})
            msgs = req.get("messages") or []
            text = "\n".join(m.get("content") or "" for m in msgs)
            if req.get("stream"):
                streams[0] += 1
                which = "speed" if streams[0] == 1 else "repeat"
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers()
                for i in range(60):
                    d = {"choices": [{"index": 0, "delta": {"content": f"word{i} "}, "finish_reason": None}]}
                    self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n"); self.wfile.flush(); time.sleep(0.005)
                fin = "length" if f"{which}_length" in faults else "stop"
                self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": fin}]}).encode() + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()
                if "late_eof" in faults:
                    time.sleep(0.5)  # [DONE] has reached the client; the stream closes half a second later
                self.close_connection = True
                return
            msg, kind = {"role": "assistant", "content": ""}, "chat"
            if req.get("tools") and msgs[-1]["role"] == "user":
                kind, msg["tool_calls"] = "tool", [{"id": "call_1", "type": "function", "index": 0,
                                                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]
            elif msgs[-1]["role"] == "tool":
                kind, msg["content"] = "toolresult", "The weather in Paris is 17°C and overcast."
            elif "17 * 23" in text:
                kind, msg["content"], msg["reasoning_content"] = "reasoning", "391", "17 times 23 is 391."
            elif "access code" in text:
                if "hang_context" in faults:
                    time.sleep(3600)
                kind, msg["content"] = "context", "58213"
            else:
                msg["content"] = "Red, Yellow, Blue"
            fin = "tool_calls" if kind == "tool" else "stop"
            if f"{kind}_length" in faults:
                fin = "length"
            self.send(200, {"choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                            "usage": {"prompt_tokens": max(1, len(text) // 4), "completion_tokens": 5}})

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    S(("127.0.0.1", port), H).serve_forever()


# ----------------------------------------------------------------------------- stub runtime
PODMAN = r'''#!/usr/bin/env bash
# fake podman. State: $FAKE_STATE/c/<id>/{labels,pid,name}. Every call is appended to $FAKE_STATE/calls.
S=$FAKE_STATE/c; mkdir -p "$S"
while [[ "${1:-}" == --root || "${1:-}" == --runroot || "${1:-}" == --tmpdir ]]; do shift 2; done
echo "$*" >> "$FAKE_STATE/calls"
label_of() { sed -n 's/^sybil.run=//p' "$S/$1/labels" 2>/dev/null; }
case ",${FAKE_HANG:-}," in *",$1,"*) [ -e "$FAKE_STATE/started" ] && { trap '' TERM; exec sleep 100000; } ;; esac
alive() { [ -f "$S/$1/pid" ] && kill -0 "$(cat "$S/$1/pid")" 2>/dev/null; }
stop() { [ -f "$S/$1/pid" ] && { kill -9 -- "-$(cat "$S/$1/pid")" 2>/dev/null; kill -9 "$(cat "$S/$1/pid")" 2>/dev/null; rm -f "$S/$1/pid"; }; }
case "$1" in
  image)
    case "$2" in
      exists)
        if [ -n "${FAKE_EXISTS_ORPHAN:-}" ]; then  # runs into the stage's own 60 s deadline: the direct child
          (trap '' TERM; exec sleep "$FAKE_EXISTS_ORPHAN") &  # dies on TERM, a grandchild ignores it
          exec sleep 600
        fi
        [ -n "${FAKE_IMAGE_ABSENT:-}" ] && [ ! -e "$FAKE_STATE/pulled" ] && exit 1; exit 0 ;;
      inspect) echo "$FAKE_DIGEST" ;;
    esac ;;
  pull)
    [ -n "${FAKE_PULL_FAIL:-}" ] && exit 1
    if [ -n "${FAKE_PULL_HANG:-}" ]; then  # a slow pull, deaf to TERM, with a second process in its group
      trap '' TERM; sleep "$FAKE_PULL_HANG" & exec sleep "$FAKE_PULL_HANG"
    fi
    touch "$FAKE_STATE/pulled"; exit 0 ;;
  run)
    shift; cid=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'); labels=(); port=
    while [ $# -gt 0 ]; do
      case "$1" in
        --cidfile) cidfile=$2; shift 2 ;;
        --label) labels+=("$2"); shift 2 ;;
        --name) name=$2; shift 2 ;;
        -p) port=$(echo "$2" | cut -d: -f2); shift 2 ;;
        -v|-e|--device|--shm-size|--entrypoint) shift 2 ;;
        *) shift ;;
      esac
    done
    [ -n "${FAKE_RUN_FAIL:-}" ] && exit 125
    touch "$FAKE_STATE/started"
    mkdir -p "$S/$cid"; printf '%s\n' "${labels[@]}" > "$S/$cid/labels"; echo "$name" > "$S/$cid/name"
    setsid "$FAKE_PY" "$FAKE_TEST" --serve "$port" "${FAKE_FAULTS:-}" > "$S/$cid/out.log" 2>&1 < /dev/null &
    echo $! > "$S/$cid/pid"
    echo "$cid" > "$cidfile"; echo "$cid" ;;
  inspect)
    if [ "$2" = --format ] || [ "$2" = -f ]; then fmt=$3; cid=$4; else fmt=; cid=$2; fi
    [ -d "$S/$cid" ] || exit 125
    case "$fmt" in
      *State.Running*) alive "$cid" && echo true || echo false ;;
      *Labels*) echo "$cid ${FAKE_LABEL_OVERRIDE:-$(label_of "$cid")}" ;;
      *) echo '[{"Id": "'"$cid"'"}]' ;;
    esac ;;
  logs)
    cid=${!#}; [ -d "$S/$cid" ] || exit 125
    if [ -n "${FAKE_LOGS_UNTIL:-}" ]; then  # a slow log save that the deadline lands in; it ignores TERM
      trap '' TERM; date +%s.%N >> "$FAKE_STATE/logs-start"
      while [ "$(date +%s)" -lt "$FAKE_LOGS_UNTIL" ]; do sleep 0.2; done
      date +%s.%N >> "$FAKE_STATE/logs-end"
    fi
    case "${FAKE_LOGS:-ok}" in fail) exit 1 ;; empty) exit 0 ;; *) cat "$S/$cid/out.log"; echo "fake log end" ;; esac ;;
  kill) cid=${!#}; [ -d "$S/$cid" ] || exit 125; stop "$cid" ;;
  rm)
    cid=${!#}; [ -d "$S/$cid" ] || exit 1
    case "${FAKE_RM:-ok}" in
      always_fail) exit 2 ;;
      fail_once) [ -e "$FAKE_STATE/rm-failed" ] || { touch "$FAKE_STATE/rm-failed"; exit 2; } ;;
    esac
    stop "$cid"; rm -rf "$S/$cid" ;;
  container) [ -d "$S/${3}" ] && exit 0 || exit 1 ;;
  ps)
    want=$(echo "$*" | sed -n 's/.*label=sybil.run=\([^ ]*\).*/\1/p')
    for d in "$S"/*/; do [ -d "$d" ] || continue; c=$(basename "$d"); [ "$(label_of "$c")" = "$want" ] && echo "$c"; done; exit 0 ;;
esac
exit 0
'''

NVSMI = r'''#!/usr/bin/env bash
echo "nvidia-smi $*" >> "$FAKE_STATE/calls"
[ -n "${FAKE_NVSMI_HANG_AFTER_RUN:-}" ] && [ -e "$FAKE_STATE/started" ] && { trap '' TERM; exec sleep 100000; }
case "$*" in
  *query-compute-apps*) [ -n "${FAKE_COMPUTE_FAIL:-}" ] && exit 1; [ -n "${FAKE_GPU_APPS:-}" ] && echo "$FAKE_GPU_APPS"; exit 0 ;;
  "--query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
    [ -n "${FAKE_UTIL_HANG:-}" ] && { trap '' TERM; exec sleep 100000; }
    [ -n "${FAKE_UTIL_FAIL:-}" ] && exit 1
    [ -n "${FAKE_BUSY_AFTER_PULL:-}" ] && [ -e "$FAKE_STATE/pulled" ] && { echo "99, 9000"; exit 0; }
    printf '%s\n' "${FAKE_UTIL:-3, 52}"; exit 0 ;;
  "--query-gpu=name --format=csv,noheader") echo "${FAKE_GPU_NAME:-NVIDIA GeForce RTX 4080 Laptop GPU}" ;;
  "--query-gpu=driver_version --format=csv,noheader") echo 610.57.04 ;;
  *) echo "fake nvidia-smi $*" ;;
esac
'''


DATE = r'''#!/usr/bin/env bash
# fake date: passes through, except that with FAKE_LATE_SIGNAL (e.g. TERM or TERM,INT) it signals the
# controller's own bash once, from the first `date` after cleanup.ok exists: inside the final verdict stamp.
if [ -n "${FAKE_LATE_SIGNAL:-}" ] && ls "$STORE"/run-*/cleanup.ok > /dev/null 2>&1 && [ ! -e "$FAKE_STATE/late-sent" ]; then
  : > "$FAKE_STATE/late-sent"
  p=$PPID
  while [ "${p:-1}" -gt 1 ]; do  # the repro.sh bash whose parent is the outer timeout
    if [[ "$(tr '\0' ' ' < /proc/$p/cmdline)" == "bash "*repro.sh* ]]; then
      pp=$(awk '{print $4}' /proc/$p/stat)
      [[ "$(tr '\0' ' ' < /proc/$pp/cmdline)" == timeout* ]] && break
    fi
    p=$(awk '{print $4}' /proc/$p/stat)
  done
  echo "late signals ${FAKE_LATE_SIGNAL} -> $p" >> "$FAKE_STATE/late-sent"
  for sig in ${FAKE_LATE_SIGNAL//,/ }; do kill -s "$sig" "$p"; done
fi
exec "$FAKE_REAL_DATE" "$@"
'''


PGREP = r"""#!/usr/bin/env bash
# fake pgrep: the real one, unless FAKE_PGREP_ERROR (from the start) or FAKE_PGREP_ERROR_AFTER_RUN (once the
# container exists) names an exit status to return instead, as a failing or missing pgrep would
code=${FAKE_PGREP_ERROR:-}
[ -z "$code" ] && [ -n "${FAKE_PGREP_ERROR_AFTER_RUN:-}" ] && [ -e "$FAKE_STATE/started" ] && code=$FAKE_PGREP_ERROR_AFTER_RUN
[ -n "$code" ] && exit "$code"
exec "$FAKE_REAL_PGREP" "$@"
"""


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Repro(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="sybil-repro-test-")
        self.repo = os.path.join(self.t, "repo")
        for d in ("lab", "registry"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(self.repo, d), ignore=shutil.ignore_patterns("runs", "__pycache__"))
        os.makedirs(os.path.join(self.repo, "lab", "runs"))
        shutil.copytree(HERE, os.path.join(self.repo, REL), ignore=shutil.ignore_patterns("__pycache__"))
        # fake weights, and the copy's manifest replaced by theirs (the shipped manifest is checked in test_manifest)
        self.model = os.path.join(self.t, "model"); os.makedirs(self.model)
        import hashlib
        lines = []
        for name, data in (("config.json", b"{}"), ("model.safetensors", b"x" * 1000)):
            open(os.path.join(self.model, name), "wb").write(data)
            lines.append(f"sha256 {hashlib.sha256(data).hexdigest()} {len(data)} {name}")
        open(os.path.join(self.repo, REL, "weights.manifest"), "w").write("\n".join(lines) + "\n")
        self.libdir = os.path.join(self.t, "lib"); os.makedirs(self.libdir)
        for real, links in (("libcuda.so.610.57.04", ["libcuda.so.1", "libcuda.so"]), ("libnvidia-ml.so.610.57.04", ["libnvidia-ml.so.1"]),
                            ("libnvidia-ptxjitcompiler.so.610.57.04", ["libnvidia-ptxjitcompiler.so.1"]),
                            ("libnvidia-nvvm.so.610.57.04", ["libnvidia-nvvm.so.4"]), ("libnvidia-gpucomp.so.610.57.04", [])):
            open(os.path.join(self.libdir, real), "w").write("fake")
            for l in links:
                os.symlink(real, os.path.join(self.libdir, l))
        self.bin = os.path.join(self.t, "bin"); os.makedirs(self.bin)
        for name, text in (("podman", PODMAN), ("nvidia-smi", NVSMI), ("date", DATE), ("pgrep", PGREP)):
            p = os.path.join(self.bin, name); open(p, "w").write(text); os.chmod(p, 0o755)
        self.state = os.path.join(self.t, "state"); os.makedirs(os.path.join(self.state, "c"))
        self.store = os.path.join(self.t, "store")
        self.home = os.path.join(self.t, "home"); os.makedirs(self.home)
        path = ":".join(d for d in os.environ["PATH"].split(":") if ".local/bin" not in d and "mise" not in d)  # no real parish-media
        self.env = {**os.environ, "HOME": self.home, "PATH": self.bin + ":" + path, "FAKE_STATE": self.state, "FAKE_PY": sys.executable,
                    "FAKE_TEST": os.path.join(self.repo, REL, "test_repro.py"), "FAKE_DIGEST": DIGEST, "FAKE_REAL_DATE": shutil.which("date"), "FAKE_REAL_PGREP": shutil.which("pgrep"), "MODEL_DIR": self.model, "STORE": self.store,
                    "LIBDIR": self.libdir, "SERVER_PORT": str(free_port()), "REC_PORT": str(free_port()), "PYTHON": sys.executable,
                    "HEALTH_TIMEOUT": "60", "LAB_TIMEOUT": "300", "REPEAT_TIMEOUT": "120", "REPRO_TIMEOUT": "600"}
        self.env.pop("SYBIL_REPRO_BOUNDED", None)
        self.foreign = "f" * 64  # someone else's container in the same store: must never be touched
        os.makedirs(os.path.join(self.state, "c", self.foreign))
        open(os.path.join(self.state, "c", self.foreign, "labels"), "w").write("sybil.run=SOMEONE-ELSE\n")

    def tearDown(self):
        for pat in [self.t] + ([f"^sleep {self.hang}$"] if getattr(self, "hang", None) else []):
            for line in subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split():
                if int(line) != os.getpid():
                    try:
                        os.kill(int(line), signal.SIGKILL)
                    except Exception:
                        pass
        for d in os.listdir(os.path.join(self.state, "c")):  # never leave a fake server behind, even on failure
            pid = os.path.join(self.state, "c", d, "pid")
            if os.path.exists(pid):
                try:
                    os.killpg(int(open(pid).read()), signal.SIGKILL)
                except Exception:
                    pass
        shutil.rmtree(self.t, ignore_errors=True)

    def run_repro(self, timeout=900, **env):
        t0 = time.monotonic()
        out = os.path.join(self.t, f"repro-out-{time.monotonic_ns()}.txt")
        with open(out, "w") as f:  # a file, not a pipe: a leaked descriptor cannot hang the harness (assertClean finds the leak)
            # SIGINT back to default, as from a user's shell: a job started with `&` by a non-interactive shell
            # inherits SIGINT ignored, and bash cannot trap a signal that was ignored when it started
            r = subprocess.run(["bash", os.path.join(self.repo, REL, "repro.sh")], env={**self.env, **env}, stdin=subprocess.DEVNULL,
                               stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        self.elapsed = time.monotonic() - t0
        for _ in range(50):  # the run's logger (tee) may still be writing its last lines
            if not subprocess.run(["pgrep", "-f", f"tee -a {self.store}/"], capture_output=True).stdout:
                break
            time.sleep(0.1)
        self.out = open(out).read()
        return r.returncode

    def ours(self):
        return [d for d in os.listdir(os.path.join(self.state, "c")) if d != self.foreign]

    def calls(self):
        p = os.path.join(self.state, "calls")
        return open(p).read() if os.path.exists(p) else ""

    def run_dir(self):
        runs = sorted(d for d in os.listdir(self.store) if d.startswith("run-"))
        return os.path.join(self.store, runs[-1])

    def assertClean(self):
        self.assertEqual(self.ours(), [], "this run's container must be positively gone")
        self.assertTrue(os.path.isdir(os.path.join(self.state, "c", self.foreign)), "foreign container removed")
        self.assertNotIn(self.foreign, self.calls(), "foreign container was touched")
        left = subprocess.run(["pgrep", "-af", self.t], capture_output=True, text=True).stdout
        self.assertEqual([l for l in left.splitlines() if "pgrep" not in l], [], "helper or server process left behind")

    def tree(self):
        import hashlib
        out = {}
        for top in ("lab", "registry", REL):
            for dp, dn, fn in os.walk(os.path.join(self.repo, top)):  # __pycache__ included: bytecode is a write too
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, self.repo)] = hashlib.sha256(open(q, "rb").read()).hexdigest()
        return out

    def test_qualified(self):
        before = self.tree()
        rc = self.run_repro()
        self.assertEqual(rc, 0, self.out[-3000:])
        self.assertEqual(self.tree(), before, "the checkout was written")  # the lab ran in STORE/run-<id>/lab-root
        R0 = self.run_dir()
        self.assertTrue(os.path.exists(os.path.join(R0, "stock-recipe.json")))  # the lab's rewritten recipe, kept in the run dir
        self.assertTrue(any(f.startswith("rtx-4080-laptop-12gb.") and f.endswith(".json") for f in os.listdir(R0)))
        R = self.run_dir()
        pc = json.load(open(os.path.join(R, "postconditions.json")))
        self.assertTrue(pc["pass"]); self.assertEqual(len([c for c in pc["chat_records"] if "n" in c]), 7)
        self.assertTrue(os.path.getsize(os.path.join(R, "server.log")) > 0)
        self.assertEqual(open(os.path.join(R, "cleanup.ok")).read().strip(), "1")
        self.assertLess(self.calls().index("logs"), self.calls().rindex("rm "))  # log saved before removal
        self.assertClean()

    def assertGateTextRightButLength(self, fault):  # Sol r1 falsifier: correct text, finish length -> stock lab passes, repro fails
        rc = self.run_repro(FAKE_FAULTS=fault)
        self.assertEqual(rc, 8, self.out[-3000:])
        self.assertEqual(open(os.path.join(self.run_dir(), "lab.exit")).read().strip(), "0")  # the stock lab alone passes
        self.assertClean()

    def test_reasoning_length(self):
        self.assertGateTextRightButLength("reasoning_length")

    def test_tool_call_length(self):
        self.assertGateTextRightButLength("tool_length")

    def test_context_length(self):
        self.assertGateTextRightButLength("context_length")

    def test_speed_length(self):
        self.assertGateTextRightButLength("speed_length")

    def test_repeat_length(self):
        self.assertGateTextRightButLength("repeat_length")

    def test_log_failure_fails_run(self):
        rc = self.run_repro(FAKE_LOGS="fail")
        self.assertEqual(rc, 9, self.out[-2000:]); self.assertIn("rm ", self.calls())  # removal still attempted
        self.assertClean()

    def test_empty_log_fails_run(self):
        self.assertEqual(self.run_repro(FAKE_LOGS="empty"), 9, self.out[-2000:]); self.assertClean()

    def test_rm_fails_once_escalates(self):
        rc = self.run_repro(FAKE_RM="fail_once")
        self.assertEqual(rc, 0, self.out[-2000:]); self.assertIn("escalating", self.out); self.assertIn("kill ", self.calls())
        self.assertClean()

    def test_rm_always_fails(self):
        rc = self.run_repro(FAKE_RM="always_fail")
        self.assertEqual(rc, 9, self.out[-2000:]); self.assertIn("NOT removed", self.out)
        self.assertEqual(len(self.ours()), 1)  # still there, and the run says so

    def test_unowned_label_never_touched(self):
        rc = self.run_repro(FAKE_LABEL_OVERRIDE="SOMEONE-ELSE")
        self.assertEqual(rc, 5, self.out[-2000:])
        cid = self.ours()[0]
        after = self.calls().split("run -d", 1)[1].splitlines()
        touched = [l for l in after if l.split()[:1] in (["rm"], ["kill"], ["logs"], ["stop"]) and cid in l]
        self.assertEqual(touched, [], "a container that is not verifiably this run's was logged, stopped or removed")
        self.assertIn("left untouched", self.out)

    def test_gpu_busy(self):
        self.assertEqual(self.run_repro(FAKE_GPU_APPS="4242, python, 900 MiB"), 3); self.assertNotIn("run -d", self.calls()); self.assertClean()

    def test_wrong_gpu(self):
        self.assertEqual(self.run_repro(FAKE_GPU_NAME="NVIDIA GeForce RTX 4070"), 3); self.assertNotIn("run -d", self.calls())

    def test_missing_driver_library(self):
        os.remove(os.path.join(self.libdir, "libnvidia-gpucomp.so.610.57.04"))
        self.assertEqual(self.run_repro(), 4); self.assertNotIn("run -d", self.calls())

    def test_weights_mismatch(self):
        open(os.path.join(self.model, "config.json"), "w").write("{ }")
        self.assertEqual(self.run_repro(), 4); self.assertNotIn("run -d", self.calls())

    def test_lock_held(self):
        os.makedirs(self.store, exist_ok=True)
        holder = subprocess.Popen(["flock", os.path.join(self.store, "repro.lock"), "sleep", "60"])
        try:
            time.sleep(0.5)
            self.assertEqual(self.run_repro(), 7); self.assertNotIn("run -d", self.calls())
        finally:
            holder.kill(); holder.wait()

    def test_never_healthy_bounded(self):
        rc = self.run_repro(FAKE_FAULTS="never_healthy", HEALTH_TIMEOUT="8")
        self.assertEqual(rc, 6, self.out[-2000:]); self.assertClean()

    def test_lab_timeout_bounded(self):
        rc = self.run_repro(FAKE_FAULTS="hang_context", LAB_TIMEOUT="15")
        self.assertEqual(rc, 1, self.out[-2000:]); self.assertClean()

    def test_overall_timeout_cleans_up(self):
        rc = self.run_repro(FAKE_FAULTS="hang_context", REPRO_TIMEOUT="25")
        self.assertIn(rc, (124, 143), self.out[-2000:]); self.assertIn("cleanup verdict", self.out); self.assertClean()
        self.assertLess(self.elapsed, 25 + 60)  # the lab stage is stopped at once, not when its own bound runs out

    # ---- Tier 2 r1 F5: the overall deadline lands in a long stage that does not depend on the recorder (a pull
    # that ignores TERM and has a second process in its group); the stage must be stopped and cleanup finish
    def test_deadline_in_slow_pull(self):
        import random, re
        self.hang = str(7000000 + random.randrange(1000000))
        rc = self.run_repro(timeout=200, FAKE_IMAGE_ABSENT="1", FAKE_PULL_HANG=self.hang, REPRO_TIMEOUT="20")
        self.assertIn(rc, (124, 143), self.out[-2000:])
        self.assertIn("stopping the running stage", self.out)
        self.assertTrue(re.search(r"cleanup verdict: ok=1 after \d+s of 540s", self.out), self.out[-2000:])
        self.assertLess(self.elapsed, 20 + 60)  # TERM, the pull's 10 s grace, then KILL of its group
        left = subprocess.run(["pgrep", "-f", f"^sleep {self.hang}$"], capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [], "the stage or its grandchild survived")
        self.assertNotIn("run -d", self.calls()); self.assertClean()

    # ---- Codex: a stage reaches its own deadline; its direct child dies on TERM, a grandchild ignores it, and
    # `timeout` exits 124 before its -k KILL. The grandchild must not survive, and cleanup must say so truthfully.
    def test_stage_own_deadline_grandchild_ignores_term(self):
        import random, re
        self.hang = str(7000000 + random.randrange(1000000))
        rc = self.run_repro(timeout=300, FAKE_EXISTS_ORPHAN=self.hang, FAKE_PULL_FAIL="1")
        self.assertEqual(rc, 4, self.out[-2000:])  # image check timed out, the pull failed
        self.assertIn("outlived its timeout", self.out)
        self.assertTrue(re.search(r"cleanup verdict: ok=1 after \d+s", self.out), self.out[-2000:])
        left = subprocess.run(["pgrep", "-f", f"^sleep {self.hang}$"], capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [], "the grandchild survived the stage")
        self.assertClean()

    # ---- Codex r2: a pgrep error is not "empty". From the start: the first stage fails closed (exit 9) and
    # cleanup reports ok=0. Once the container exists: the same, and the container is still removed.
    def assertPgrepErrorFailsClosed(self, **env):
        import re
        rc = self.run_repro(**env)
        self.assertEqual(rc, 9, self.out[-2000:])
        self.assertIn("cannot be confirmed empty", self.out)
        self.assertTrue(re.search(r"cleanup verdict: ok=0 after \d+s", self.out), self.out[-2000:])
        self.assertClean()

    def test_pgrep_error_from_start(self):
        self.assertPgrepErrorFailsClosed(FAKE_PGREP_ERROR="2")

    def test_pgrep_missing_from_start(self):
        self.assertPgrepErrorFailsClosed(FAKE_PGREP_ERROR="127")

    def test_pgrep_error_once_container_exists(self):
        self.assertPgrepErrorFailsClosed(FAKE_PGREP_ERROR_AFTER_RUN="2")
        self.assertIn("run -d", self.calls())
        self.assertIn("removed (positively gone)", self.out)

    # ---- Tier 2 r1 F6: [DONE] reaches the client half a second before the stream closes
    def test_late_eof_after_done(self):
        rc = self.run_repro(FAKE_FAULTS="late_eof")
        self.assertEqual(rc, 0, self.out[-3000:])
        rows = [json.loads(l) for l in open(os.path.join(self.run_dir(), "recorder.jsonl"))]  # every line parses
        self.assertEqual(len([r for r in rows if r["method"] == "POST" and r["path"] == "/v1/chat/completions"]), 7)
        self.assertClean()

    # ---- idle / no-gaming policy (Sol r2 finding 2): never start a container on a busy or unknown GPU
    def assertBusy(self, needle, **env):
        rc = self.run_repro(**env)
        self.assertEqual(rc, 3, self.out[-2000:]); self.assertIn(needle, self.out)
        self.assertNotIn("run -d", self.calls()); self.assertClean()

    def test_graphics_busy_compute_empty(self):
        self.assertBusy("GPU busy: 99% utilization", FAKE_UTIL="99, 3000")

    def test_busy_after_pull(self):
        self.assertBusy("launch: GPU not idle", FAKE_IMAGE_ABSENT="1", FAKE_BUSY_AFTER_PULL="1")
        self.assertIn("pull ", self.calls())

    def test_util_malformed(self):
        self.assertBusy("malformed", FAKE_UTIL="[N/A], 52")

    def test_util_two_lines(self):
        self.assertBusy("malformed", FAKE_UTIL="3, 52\n4, 60")

    def test_util_timeout_bounded(self):
        self.assertBusy("failed or timed out", FAKE_UTIL_HANG="1")
        self.assertLess(self.elapsed, 60)

    def test_util_query_fails(self):
        self.assertBusy("failed or timed out", FAKE_UTIL_FAIL="1")

    def test_compute_query_fails(self):
        self.assertBusy("compute query failed", FAKE_COMPUTE_FAIL="1")

    def power(self, ac):
        d = os.path.join(self.t, "power")
        for name, ty, online in (("BAT0", "Battery", None), ("AC", "Mains", ac)):
            os.makedirs(os.path.join(d, name), exist_ok=True)
            open(os.path.join(d, name, "type"), "w").write(ty + "\n")
            if online is not None:
                open(os.path.join(d, name, "online"), "w").write(online + "\n")
        return d

    def test_on_battery(self):
        self.assertBusy("not on AC power", SYBIL_TEST_POWER_DIR=self.power("0"))

    def test_power_override_never_qualifies(self):
        rc = self.run_repro(SYBIL_TEST_POWER_DIR=self.power("1"))
        self.assertEqual(rc, 10, self.out[-2000:]); self.assertIn("cannot qualify", self.out); self.assertClean()

    def parish_media(self, out, code):
        d = os.path.join(self.t, "pm"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "parish-media"); open(p, "w").write(f"#!/bin/sh\ncat <<'EOF'\n{out}\nEOF\nexit {code}\n"); os.chmod(p, 0o755)
        return d + ":" + self.env["PATH"]

    def test_parish_media_not_ready(self):
        self.assertBusy("parish-media guard not ready", PATH=self.parish_media(json.dumps({"ac_online": True, "not_ready": "GPU is in use"}), 0))

    def test_parish_media_command_fails(self):
        self.assertBusy("parish-media status failed", PATH=self.parish_media(json.dumps({"ac_online": True, "not_ready": None}), 1))

    def test_parish_media_ready(self):
        rc = self.run_repro(PATH=self.parish_media(json.dumps({"ac_online": True, "not_ready": None}), 0))
        self.assertEqual(rc, 0, self.out[-2000:]); self.assertClean()

    # ---- cleanup reserve (Sol r2 finding 1): every cleanup step hangs to its bound; cleanup still reaches its verdict
    def test_worst_case_cleanup_reaches_verdict(self):
        rc = self.run_repro(timeout=900, FAKE_FAULTS="hang_context", REPRO_TIMEOUT="20",
                            FAKE_HANG="logs,rm,kill,container", FAKE_NVSMI_HANG_AFTER_RUN="1")
        self.assertEqual(rc, 124, self.out[-3000:])  # TERMed by the overall bound, never KILLed (137)
        import re
        m = re.search(r"cleanup verdict: ok=0 after (\d+)s of (\d+)s", self.out)
        self.assertTrue(m, self.out[-3000:])
        used, reserve = int(m.group(1)), int(m.group(2))
        self.assertGreater(used, 100)  # the hung steps really ran to their bounds
        self.assertLess(used, reserve)
        self.assertIn("NOT removed", self.out)
        self.assertLess(self.elapsed, 20 + reserve)

    # ---- Sol r3: the overall deadline lands inside an already running cleanup (in save_logs)
    def deadline_in_cleanup(self, **env):
        import re
        bound = 20  # cleanup starts about 10 s in; the slow log save runs until start + 25 s, inside its own 30 s bound
        t0 = time.time()
        until = int(t0) + bound + 5
        rc = self.run_repro(REPRO_TIMEOUT=str(bound), FAKE_LOGS_UNTIL=str(until), **env)
        self.assertEqual(rc, 124, self.out[-3000:])  # the bound fired; the run never reached KILL (137)
        start = float(open(os.path.join(self.state, "logs-start")).read().split()[0])
        end = float(open(os.path.join(self.state, "logs-end")).read().split()[0])
        self.assertLess(start, t0 + bound - 1); self.assertLess(t0 + bound + 1, end)  # the deadline landed inside save_logs
        R = self.run_dir()
        log = open(os.path.join(R, "repro.log")).read()
        m = re.search(r"signal 15 during cleanup at ([0-9.]+): deferred", log)
        self.assertTrue(m, log[-3000:])
        self.assertGreaterEqual(float(m.group(1)), start)  # bash runs the handler once the interrupted step returns
        self.assertNotIn("during cleanup", open(os.path.join(R, "server.log")).read())  # the raw server log stays raw
        after = self.out[self.out.index("cleanup (reserve"):]
        self.assertIn("server log saved", after)
        self.assertIn("removed (positively gone)", after)  # removal was attempted and confirmed after the signal
        v = re.search(r"cleanup verdict: ok=(\d) after (\d+)s of (\d+)s; exit (\d+)", after)
        self.assertTrue(v, after[-2000:])
        self.assertEqual(v.group(1), "1"); self.assertLess(int(v.group(2)), int(v.group(3)))
        self.assertNotEqual(v.group(4), "0")  # a run that hit its deadline never reports success
        self.assertIn(f"rm -f -t 10 ", self.calls()[self.calls().index("logs --timestamps"):])
        self.assertClean()  # container positively gone, recorder and fake server gone, foreign container untouched
        return v.group(4)

    def test_deadline_during_cleanup_after_success(self):
        self.assertEqual(self.deadline_in_cleanup(), "143")

    def test_deadline_during_cleanup_after_failure(self):
        self.assertEqual(self.deadline_in_cleanup(FAKE_FAULTS="reasoning_length"), "8")

    # ---- Sol r4: a signal inside the final verdict stamp, after cleanup has read its status, must still fail the run
    def late_signal(self, sigs, **env):
        rc = self.run_repro(FAKE_LATE_SIGNAL=sigs, **env)
        sent = open(os.path.join(self.state, "late-sent")).read()
        self.assertIn(f"late signals {sigs} -> ", sent)
        log = open(os.path.join(self.run_dir(), "repro.log")).read()
        tail = log[log.index("cleanup (reserve"):]  # the record goes straight to the file, the verdict through tee: either order
        self.assertIn("cleanup verdict:", tail)
        self.assertIn("during cleanup at", tail)
        self.assertIn("removed (positively gone)", tail)
        self.assertClean()
        return rc, tail

    def test_late_term_after_pass(self):
        rc, tail = self.late_signal("TERM")
        self.assertEqual(rc, 143, tail)  # the verdict line may still read "exit 0"; the process status is what counts

    def test_late_int_after_pass(self):
        rc, tail = self.late_signal("INT")
        self.assertEqual(rc, 130, tail)

    def test_late_term_then_int(self):
        rc, tail = self.late_signal("TERM,INT")
        self.assertEqual(rc, 143, tail)  # the first signal sets the status
        self.assertIn("signal 15 during cleanup", tail); self.assertIn("signal 2 during cleanup", tail)

    def test_late_term_keeps_prior_failure(self):
        rc, tail = self.late_signal("TERM", FAKE_FAULTS="reasoning_length")
        self.assertEqual(rc, 8, tail)

    def test_container_start_failure(self):
        self.assertEqual(self.run_repro(FAKE_RUN_FAIL="1"), 5); self.assertClean()


class StageGroup(unittest.TestCase):
    """The exact stage / empty_group / stop_stage functions from repro.sh, run in a bare bash harness."""
    # an unkillable group, simulated: pgrep keeps reporting it, and kill is a no-op so the harness never signals a
    # group id it does not own (tearDown removes the test's own sleep)
    UNKILLABLE = 'pgrep() { return 0; }; kill() { echo "kill suppressed: $*"; }'
    def setUp(self):
        import random
        self.n = str(7000000 + random.randrange(1000000))
        s = open(os.path.join(HERE, "repro.sh")).read()
        self.funcs = s[s.index("proc_start() {"):s.index("# A container id is this run's only")]

    def tearDown(self):
        for pid in subprocess.run(["pgrep", "-f", f"^sleep {self.n}$"], capture_output=True, text=True).stdout.split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass

    def bash(self, body, pre=""):  # stage commands send their output to /dev/null: a survivor must not hold these pipes
        harness = ('STAGE=; STAGE_T=\nstamp() { echo "STAMP $*"; }\nfail() { echo "FAIL $1 STAGE=[$STAGE]"; exit "$1"; }\n'
                   + self.funcs + pre + "\n" + body)
        return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=60)

    def left(self):
        time.sleep(0.5)  # an orphan reparented to init is reaped promptly once it is KILLed
        return subprocess.run(["pgrep", "-f", f"^sleep {self.n}$"], capture_output=True, text=True).stdout.split()

    def test_own_deadline_grandchild_ignores_term(self):  # Codex's reproduction: stage 1 1, rc 124
        r = self.bash(f"""stage 1 1 bash -c "exec > /dev/null 2>&1; (trap '' TERM; exec sleep {self.n}) & exec sleep 100"; echo "rc=$? STAGE=[$STAGE] STAGE_T=[$STAGE_T]"
stop_stage; echo "cleanup_rc=$?" """)
        self.assertIn("rc=124 STAGE=[] STAGE_T=[]", r.stdout, r.stdout + r.stderr)
        self.assertIn("outlived its timeout", r.stdout)
        self.assertIn("cleanup_rc=0", r.stdout)
        self.assertEqual(self.left(), [], "the grandchild survived")

    def test_finished_stage_leaves_a_child(self):
        r = self.bash(f'stage 10 1 bash -c "exec > /dev/null 2>&1; sleep {self.n} & exit 0"; echo "rc=$? STAGE=[$STAGE]"')
        self.assertIn("rc=0 STAGE=[]", r.stdout, r.stdout + r.stderr)
        self.assertEqual(self.left(), [], "the background child survived")

    def test_clean_stage_needs_no_kill(self):
        r = self.bash('stage 10 1 true; echo "rc=$? STAGE=[$STAGE]"')
        self.assertIn("rc=0 STAGE=[]", r.stdout); self.assertNotIn("outlived", r.stdout)

    def test_group_that_cannot_be_emptied_fails_closed(self):  # pgrep keeps reporting the group: an unkillable member
        r = self.bash(f'stage 1 1 bash -c "exec > /dev/null 2>&1; (trap \'\' TERM; exec sleep {self.n}) & exec sleep 100"; echo "returned rc=$?"',
                      pre=self.UNKILLABLE)
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r"FAIL 9 STAGE=\[\d+\]")  # the identity is kept for cleanup
        self.assertNotIn("returned rc=", r.stdout)

    QUERY_ERROR = 'pgrep() { return 2; }; kill() { echo "kill sent: $*"; }'
    MISSING = 'pgrep() { return 127; }; kill() { echo "kill sent: $*"; }'

    def assertFailsClosed(self, r):
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r"FAIL 9 STAGE=\[\d+\]")  # the identity is kept for cleanup
        self.assertIn("cannot be confirmed empty", r.stdout)
        self.assertNotIn("returned rc=", r.stdout)
        self.assertNotIn("kill sent: -KILL", r.stdout)  # no KILL to a group that cannot be seen

    def test_query_error_on_own_deadline_fails_closed(self):  # Codex r2: pgrep exit 2 after stage 1 1
        self.assertFailsClosed(self.bash(
            f'stage 1 1 bash -c "exec > /dev/null 2>&1; (trap \'\' TERM; exec sleep {self.n}) & exec sleep 100"; echo "returned rc=$?"',
            pre=self.QUERY_ERROR))

    def test_query_error_after_a_clean_stage_fails_closed(self):
        self.assertFailsClosed(self.bash('stage 10 1 true; echo "returned rc=$?"', pre=self.QUERY_ERROR))

    def test_missing_pgrep_fails_closed(self):
        self.assertFailsClosed(self.bash('stage 10 1 true; echo "returned rc=$?"', pre=self.MISSING))

    def test_query_error_in_stop_stage(self):
        r = self.bash('sleep 0.1 & STAGE=$!; STAGE_T=x; wait; stop_stage; echo "stop_rc=$?"', pre=self.QUERY_ERROR)
        self.assertIn("stop_rc=1", r.stdout, r.stdout + r.stderr)
        self.assertIn("cannot be confirmed empty", r.stdout)
        self.assertNotIn("kill sent: -KILL", r.stdout)

    def test_stop_stage_reports_a_group_it_cannot_empty(self):
        r = self.bash('sleep 0.1 & STAGE=$!; STAGE_T=x; wait; stop_stage; echo "stop_rc=$?"', pre=self.UNKILLABLE)
        self.assertIn("could not be confirmed empty", r.stdout); self.assertIn("stop_rc=1", r.stdout)


class RecorderShutdown(unittest.TestCase):
    """A client that stops at [DONE] and a recorder stopped at once: the row must still be written, whole."""
    def test_term_right_after_done_keeps_the_row(self):
        up_port, rec_port = free_port(), free_port()
        d = tempfile.mkdtemp(prefix="sybil-repro-test-rec-")
        up = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "--serve", str(up_port), "late_eof"])
        rec = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "finish_recorder.py"), str(rec_port), str(up_port), os.path.join(d, "rec")])
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", rec_port), 0.2).close(); socket.create_connection(("127.0.0.1", up_port), 0.2).close(); break
                except OSError:
                    time.sleep(0.05)
            body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "story"}]}).encode()
            s = socket.create_connection(("127.0.0.1", rec_port), 5)
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
            buf = b""
            while b"data: [DONE]" not in buf:
                chunk = s.recv(65536)
                self.assertTrue(chunk, "stream ended before [DONE]")
                buf += chunk
            rec.send_signal(signal.SIGTERM)  # at once, while the upstream still holds the stream open
            self.assertEqual(rec.wait(10), 0)
            s.close()
            rows = [json.loads(l) for l in open(os.path.join(d, "rec.jsonl"))]
            chat = [r for r in rows if r["path"] == "/v1/chat/completions"]
            self.assertEqual(len(chat), 1, rows)
            self.assertEqual(chat[0]["finish_reasons"], ["stop"])
        finally:
            for p in (rec, up):
                if p.poll() is None:
                    p.kill(); p.wait()
            shutil.rmtree(d, ignore_errors=True)


class WeightsContract(unittest.TestCase):
    """verify_weights.py: exactly the manifest's files; a real .cache directory is the one extra allowed."""
    def setUp(self):
        import hashlib
        self.d = tempfile.mkdtemp(prefix="sybil-repro-test-w-")
        self.m = os.path.join(self.d, "model"); os.makedirs(self.m)
        lines = []
        for name, data in (("config.json", b"{}"), ("model.safetensors", b"x" * 100)):
            open(os.path.join(self.m, name), "wb").write(data)
            lines.append(f"sha256 {hashlib.sha256(data).hexdigest()} {len(data)} {name}")
        self.manifest = os.path.join(self.d, "manifest"); open(self.manifest, "w").write("\n".join(lines) + "\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def check(self):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "verify_weights.py"), self.manifest, self.m], capture_output=True, text=True).returncode

    def test_exact(self):
        self.assertEqual(self.check(), 0)

    def test_hf_cache_directory_allowed(self):
        os.makedirs(os.path.join(self.m, ".cache", "huggingface", "download"))
        self.assertEqual(self.check(), 0)

    def test_cache_as_file_rejected(self):
        open(os.path.join(self.m, ".cache"), "w").write("x")
        self.assertEqual(self.check(), 1)

    def test_cache_symlink_rejected(self):
        os.makedirs(os.path.join(self.d, "elsewhere")); os.symlink(os.path.join(self.d, "elsewhere"), os.path.join(self.m, ".cache"))
        self.assertEqual(self.check(), 1)

    def test_other_dot_directory_rejected(self):
        os.makedirs(os.path.join(self.m, ".git"))
        self.assertEqual(self.check(), 1)

    def test_extra_file_rejected(self):
        open(os.path.join(self.m, "extra.bin"), "w").write("x")
        self.assertEqual(self.check(), 1)

    def test_changed_file_rejected(self):
        open(os.path.join(self.m, "config.json"), "w").write("{ }")
        self.assertEqual(self.check(), 1)


class Manifest(unittest.TestCase):
    def test_manifest_pins_the_measured_weights(self):
        rows = [l.split() for l in open(os.path.join(HERE, "weights.manifest"))]
        self.assertEqual(len(rows), 13)
        d = {r[3]: (r[0], r[1], int(r[2])) for r in rows}
        self.assertEqual(d["model.safetensors"], ("sha256", "c43536c08cb54147a097937868f0791f637f98d1505c79475aa9b9b05589a74b", 7307469398))
        self.assertEqual(d["tokenizer.json"][1], "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42")

    def test_script_pins(self):
        s = open(os.path.join(HERE, "repro.sh")).read()
        for pin in ("tabbyapi-exl3@sha256:0f83e6198dc3be2561652d8df2525a7d1a69733e12fc8e25cbc5bc319a8a3ad1",
                    "Qwen3.5-9B-exl3@e98453aa143a24fab9eb14b718daed02e2fa6eef",
                    "72198e58f86afcf47cd0b7aae9d2c23dfc5f2af25440b260e2bca6a816e07097",
                    "d3efe710e28fe94a93a81a86bf52f74f95ab6740800890d78b6a4f9db7ee9d0c",
                    "b7d36ea47989bd9d5283d2d5799851827480aba53b11aacc9fe10dd403245cca"):
            self.assertIn(pin, s)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve(int(sys.argv[2]), set(filter(None, (sys.argv[3] if len(sys.argv) > 3 else "").split(","))))
    else:
        unittest.main(verbosity=2)
