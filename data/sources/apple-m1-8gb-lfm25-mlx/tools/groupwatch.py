#!/usr/bin/env python3
"""Run a task-owned server under a supervisor that can always take down the whole tree it started, and nothing else.

  groupwatch.py run  --state DIR [--max-seconds N] [--min-free-pct P] [--max-swap-mb M] [--trigger-file F] -- CMD ...
  groupwatch.py stop --state DIR

`run` starts CMD in a new session (own process group, pgid = its pid). Every second it records CMD and its
descendants, found by walking parent links from CMD, as (pid, start time) identities. A descendant that leaves the
group (setsid) stays recorded. Ownership is never inferred from argv or from group membership alone.
Teardown happens on: lifetime cap, macOS memory guard (memory_pressure free % below P, or swap used above M MB),
the trigger file, or SIGTERM/SIGINT/SIGHUP to the supervisor (`stop`).
Teardown, per round (SIGTERM, wait up to 15 s, then SIGKILL, wait up to 5 s):
  - signal the whole group with killpg only if every process currently in that group is a recorded identity;
    otherwise skip killpg and log the unknown members;
  - signal each recorded identity that is still alive, rechecking its start time right before the signal.
Then verify that no recorded identity survives, and that its own child has exited (its Popen handle, not ps, says so;
that child is signalled through the handle too, recorded or not). Zombies count as dead; the supervisor reaps its own
child. A process table counts only when ps exits 0, every line parses, and the table holds the supervisor itself: any
other answer raises, so nothing is ever read as gone because a query failed. With the memory guard armed, three
memory readings in a row that fail (memory_pressure or sysctl exit nonzero, or print nothing usable) tear it down.
`stop` signals the supervisor only after its pid, start time and exact argv (as ps showed them when `run` began) match.
Log: DIR/groupwatch.log. Exit of `run`: CMD's exit code, or 3 when the guard, trigger, cap or a stop tore it down; a
teardown that leaves a recorded process alive, or fails, exits 4 instead (CMD's own nonzero code is kept). If the
supervision itself fails after CMD started (ps timing out, any other error), it tears down what it owns and exits 5, or
4 if that teardown is not clean; if even teardown cannot run, it still kills its own direct child. Logging is best
effort: a log line that cannot be written (a full disk) never stops supervision, teardown or the fallback, and it turns
an exit that would have been 0 into 5. The signal handlers are installed before CMD starts.
One supervisor per state directory: `run` holds an exclusive lock on DIR/groupwatch.lock from before CMD starts until it
returns, and refuses (exit 1, nothing started) while another holds it. Each run has its own token (its pid and random
hex), kept in supervisor.id and at the head of every log line it writes, so lines of runs that shared DIR stay apart.
Exit of `stop`: 0 only when that supervisor's own final EXIT/ABORT line (its token) says clean=True and no FAIL line of
its own follows it; a supervisor.id without a token cannot be bound and is not success.
"""
import fcntl, os, re, secrets, signal, subprocess, sys, time

UNCLEAN = 4  # exit code when teardown left a recorded process alive or failed
MONITOR_FAILED = 5  # exit code when supervision failed after the command started (or its log could not be written)

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

class Telemetry(RuntimeError):
    pass  # a process or memory query that failed or answered something unusable


def ps_table():
    """pid -> (ppid, pgid, lstart, args). Zombies (stat Z) are dead for our purposes and left out. Raises Telemetry unless
    ps exits 0, every line parses, and the table holds this process: an empty or partial answer is never "all gone"."""
    r = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart=,args="], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise Telemetry(f"ps exited {r.returncode}: {(r.stderr or '').strip()[:200]}")
    t = {}
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        p = line.split(None, 4)
        w = p[4].split(None, 5) if len(p) == 5 else []  # lstart is 5 words ("Fri Oct 23 16:41:51 2026"), then args
        if len(w) < 5 or not all(x.lstrip("-").isdigit() for x in p[:3]):
            raise Telemetry(f"ps printed a line it should not have: {line[:200]!r}")
        if p[3].startswith("Z"):
            continue
        t[int(p[0])] = (int(p[1]), int(p[2]), " ".join(w[:5]), w[5] if len(w) > 5 else "")
    if os.getpid() not in t:
        raise Telemetry(f"ps listed {len(t)} processes, and not this one")
    return t

def descendants(root, table):
    kids, seen, stack = {}, set(), [root]
    for pid, v in table.items():
        kids.setdefault(v[0], []).append(pid)
    while stack:
        p = stack.pop()
        if p in seen or p not in table:
            continue
        seen.add(p)
        stack += kids.get(p, [])
    return seen

def mac_memory():
    """(free %, swap used MB); either is None when its query fails, exits nonzero or prints nothing usable."""
    free = swap = None
    try:
        r = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10)
        m = re.search(r"free percentage: (\d+)%", r.stdout) if r.returncode == 0 else None
        free = int(m.group(1)) if m else None
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5)
        m = re.search(r"used = ([\d.]+)M", r.stdout) if r.returncode == 0 else None
        swap = float(m.group(1)) if m else None
    except Exception:
        pass
    return free, swap

MEMORY_MISSES = 3  # an armed guard that gets no reading this many polls in a row tears the run down

class Supervisor:
    def __init__(self, state):
        self.state, self.known, self.proc = state, {}, None  # known: pid -> lstart
        self.log_failed = False  # a log line could not be written
        self.lock = open(os.path.join(state, "groupwatch.lock"), "a")
        try:  # released by close(), or by the OS when this process ends however it ends
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise SystemExit(f"groupwatch run: {state} is in use by another supervisor; nothing started")
        self.token = f"{os.getpid()}-{secrets.token_hex(8)}"  # this run, distinct from any other that used DIR
        self.log = open(os.path.join(state, "groupwatch.log"), "a", buffering=1)

    def close(self):
        for f in (self.log, self.lock):
            try:
                f.close()
            except Exception:
                pass

    def say(self, msg):
        self.log.write(f"{now()} [{self.token}] {msg}\n")

    def note(self, msg):
        """say(), best effort: a failed write is remembered, never raised, so no cleanup step waits on the log."""
        try:
            self.say(msg)
        except Exception:
            self.log_failed = True

    def record(self, root):
        table = ps_table()
        for pid in descendants(root, table):
            self.known.setdefault(pid, table[pid][2])
        return table

    def child_alive(self):
        """Our own direct child, by its Popen handle (waitpid): positive knowledge, whatever ps says."""
        return self.proc is not None and self.proc.poll() is None  # poll() also reaps it, so it never lingers as a zombie

    def owned_alive(self, table=None):
        child = self.child_alive()
        table = ps_table() if table is None else table
        alive = {pid for pid, st in self.known.items() if pid in table and table[pid][2] == st}
        return alive | ({self.proc.pid} if child else set())

    def signal_owned(self, pid, sig):
        t = ps_table()
        if pid in t and t[pid][2] == self.known.get(pid):  # identity rechecked right before the signal
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def teardown(self, root, pgid, reason):
        self.record(root)
        self.note(f"TEARDOWN ({reason}): pgid {pgid}, recorded {sorted(self.known)}")
        for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            table = ps_table()
            live = self.owned_alive(table)
            if not live:
                break
            members = {pid for pid, v in table.items() if v[1] == pgid}
            foreign = {pid for pid in members if self.known.get(pid) != table[pid][2]}
            if members and not foreign:
                self.note(f"send {sig.name} to group {pgid} (members {sorted(members)}, all recorded)")
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
            elif foreign:
                self.note(f"skip killpg {pgid}: unrecorded members {sorted(foreign)}")
            self.note(f"send {sig.name} to recorded pids {sorted(live)}")
            for pid in live:
                self.signal_owned(pid, sig)
            if self.child_alive():  # recorded or not, our own child is signalled through its handle
                try:
                    self.proc.send_signal(sig)
                except ProcessLookupError:
                    pass
            t = time.monotonic()
            while time.monotonic() - t < wait and self.owned_alive():
                time.sleep(0.2)
        left = self.owned_alive()
        self.note(f"after teardown alive: {sorted(left) if left else 'none'}")
        return not left

def run(argv):
    a = {"--max-seconds": None, "--min-free-pct": None, "--max-swap-mb": None, "--trigger-file": None, "--state": None}
    if "--" not in argv:
        sys.exit("groupwatch run: put -- before the command")
    i = 0
    while argv[i] != "--":
        if argv[i] not in a or i + 1 >= len(argv) or argv[i + 1] == "--":
            sys.exit(f"groupwatch run: unknown option or missing value: {argv[i]}")
        a[argv[i]] = argv[i + 1]
        i += 2
    cmd, state = argv[i + 1:], a["--state"]
    if not state or not cmd:
        sys.exit("groupwatch run: --state DIR and a command are required")
    os.makedirs(state, exist_ok=True)
    sv = Supervisor(state)  # takes the state directory's lock, or exits before anything starts
    try:
        return run_locked(sv, a, cmd)
    finally:
        sv.close()

def run_locked(sv, a, cmd):
    state = a["--state"]
    me = ps_table()[os.getpid()]
    with open(os.path.join(state, "supervisor.id"), "w") as f:
        f.write(f"{os.getpid()}\n{me[2]}\n{me[3]}\n{sv.token}\n")
    stop = {"why": ""}  # handlers first: a stop that arrives while CMD starts is a stop, not a dead supervisor
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda n, f: stop.update(why=f"supervisor got {signal.Signals(n).name}"))
    proc = subprocess.Popen(cmd, start_new_session=True)  # the exec'd command gets default handlers back
    sv.proc = proc
    pgid = proc.pid  # a new session: the child leads its own group
    try:  # from here on, any failure of the supervision still tears down what it started
        rc = supervise(sv, proc, a, cmd, stop)
    except BaseException as e:  # BaseException too: nothing may leave run() with the command left running
        ok = safe_teardown(sv, proc.pid, pgid, f"monitor failed ({type(e).__name__})")  # cleanup first, then the log
        sv.note(f"monitor failed: {type(e).__name__}: {e}")
        sv.note(f"ABORT: monitor failed ({type(e).__name__}); clean={ok}")
        if not ok:
            return unclean(sv, MONITOR_FAILED)
        sv.note(f"FAIL: supervision failed; exit {MONITOR_FAILED}")
        return MONITOR_FAILED
    if rc == 0 and sv.log_failed:  # the run cannot vouch for itself without its log
        return MONITOR_FAILED
    return rc

def supervise(sv, proc, a, cmd, stop):
    state = a["--state"]
    pgid = os.getpgid(proc.pid)
    with open(os.path.join(state, "child.pid"), "w") as f:
        f.write(f"{proc.pid}\n")
    sv.note(f"options: " + " ".join(f"{k}={v}" for k, v in a.items() if k != "--state" and v is not None))
    sv.note(f"supervisor pid {os.getpid()} started child pid {proc.pid} pgid {pgid}: {' '.join(cmd)}")
    t0, last_mem, misses = time.monotonic(), 0.0, 0
    while proc.poll() is None and not stop["why"]:
        sv.record(proc.pid)
        if a["--max-seconds"] and time.monotonic() - t0 > float(a["--max-seconds"]):
            stop["why"] = f"lifetime cap {a['--max-seconds']} s"
        if a["--trigger-file"] and os.path.exists(str(a["--trigger-file"])):
            stop["why"] = "trigger file"
        if (a["--min-free-pct"] or a["--max-swap-mb"]) and time.monotonic() - last_mem >= 5:
            last_mem = time.monotonic()
            free, swap = mac_memory()
            sv.note(f"mem free_pct={free} swap_used_mb={swap}")
            missing = (a["--min-free-pct"] and free is None) or (a["--max-swap-mb"] and swap is None)
            misses = misses + 1 if missing else 0
            if misses >= MEMORY_MISSES:
                stop["why"] = f"memory guard: no reading in {misses} polls in a row"
            if a["--min-free-pct"] and free is not None and free < int(a["--min-free-pct"]):
                stop["why"] = f"memory guard: free {free}% < {a['--min-free-pct']}%"
            if a["--max-swap-mb"] and swap is not None and swap > float(a["--max-swap-mb"]):
                stop["why"] = f"memory guard: swap {swap} MB > {a['--max-swap-mb']} MB"
        time.sleep(1)
    if stop["why"]:
        ok = safe_teardown(sv, proc.pid, pgid, stop["why"])
        sv.note(f"ABORT: {stop['why']}; clean={ok}")
        return 3 if ok else unclean(sv, 3)
    rc = proc.wait()
    ok = safe_teardown(sv, proc.pid, pgid, f"command exited {rc}; sweeping recorded leftovers")
    sv.note(f"EXIT: command exited {rc}; clean={ok}")
    if ok:
        return rc
    return rc if rc != 0 else unclean(sv, rc)  # a failing command keeps its own code; a "successful" one does not pass

def safe_teardown(sv, root, pgid, reason):
    """teardown(), with any error in it counted as not clean (fail closed). If teardown cannot run, the supervisor still
    kills its own direct child (its Popen handle cannot name a reused pid); descendants are then unverified."""
    try:
        return sv.teardown(root, pgid, reason)
    except BaseException as e:
        killed = False
        try:  # the fallback before any logging
            if sv.proc is not None and sv.proc.poll() is None:
                sv.proc.kill()
                sv.proc.wait()
                killed = True
        except ProcessLookupError:
            pass
        sv.note(f"teardown failed: {type(e).__name__}: {e}")
        if killed:
            sv.note(f"killed direct child {sv.proc.pid}; recorded descendants not verified")
        return False

def unclean(sv, was):
    sv.note(f"FAIL: teardown not clean; exit {UNCLEAN} (would have been {was})")
    return UNCLEAN

def stop(argv):
    state = argv[argv.index("--state") + 1]
    with open(os.path.join(state, "supervisor.id")) as f:
        rows = f.read().splitlines()
    pid, lstart = int(rows[0]), rows[1]
    argv = rows[2] if len(rows) > 2 else None
    token = rows[3] if len(rows) > 3 else None
    t = ps_table()
    # identity = pid + start time + the exact argv ps showed when `run` began (older id files: argv must name groupwatch run)
    ok_argv = (t[pid][3] == argv) if (pid in t and argv is not None) else (pid in t and "groupwatch.py run" in t[pid][3])
    if pid not in t or t[pid][2] != lstart or not ok_argv:
        print(f"refusing: pid {pid} is not the recorded supervisor (gone, reused, or different argv)")
        return 1
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        t = ps_table()
        if pid not in t or t[pid][2] != lstart:
            return verdict(state, pid, token)
        time.sleep(0.3)
    print("supervisor still running")
    return 1

def verdict(state, pid, token):
    """0 only if this run's own final EXIT/ABORT line (by its token, after its own start line) says clean=True and no FAIL
    line of its own follows it. Lines of any other run that shared the log are never read as this one's."""
    if not token:
        print(f"groupwatch stop: supervisor.id for {pid} has no run token, so its log lines cannot be told from another run's")
        return UNCLEAN
    with open(os.path.join(state, "groupwatch.log")) as f:
        own = [l for l in f.read().splitlines() if l.split(" ", 2)[1:2] == [f"[{token}]"]]
    starts = [i for i, l in enumerate(own) if f" supervisor pid {pid} started child " in l]
    mine = own[starts[-1] + 1:] if starts else []
    ends = [i for i, l in enumerate(mine) if " EXIT: " in l or " ABORT: " in l]
    final = [mine[i] for i in ends]
    failed = bool(ends) and any(" FAIL: " in l for l in mine[ends[-1] + 1:])
    if final and final[-1].endswith("; clean=True") and not failed:
        print(final[-1])
        return 0
    print(final[-1] if final else f"supervisor {pid} left no final line in its log")
    print(f"groupwatch stop: supervisor {pid} ended without a clean teardown")
    return UNCLEAN

if __name__ == "__main__":
    sys.exit({"run": run, "stop": stop}[sys.argv[1]](sys.argv[2:]))
