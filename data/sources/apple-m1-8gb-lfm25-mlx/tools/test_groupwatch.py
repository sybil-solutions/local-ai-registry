#!/usr/bin/env python3
"""groupwatch.py fails closed: a teardown that is not clean never exits 0, from `run` or from `stop`.

  python3 -m unittest test_groupwatch      (from this directory; needs ps with lstart, as on macOS)

In-process cases run groupwatch.run() on a real short command with Supervisor.teardown replaced (the AST-tail case of the
review: the command exits 0 and teardown reports False). Program cases run groupwatch.py itself and signal it, and run
`stop` against a stand-in supervisor whose log the test writes.
"""
import contextlib, errno, io, os, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import groupwatch  # noqa: E402

GW = os.path.join(HERE, "groupwatch.py")


def close(sv):
    """Release a Supervisor's lock and log (an older groupwatch.py has no close(): its log only)."""
    (getattr(sv, "close", None) or sv.log.close)()


class InProcess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "gw")

    def run_gw(self, cmd, teardown, *opts, record=None, say=None):
        made, init = [], groupwatch.Supervisor.__init__
        def track(sv, state):
            init(sv, state)
            made.append(sv)
        self.made = made
        with patch.object(groupwatch.Supervisor, "teardown", teardown), patch.object(groupwatch.Supervisor, "__init__", track), \
                patch.object(groupwatch.Supervisor, "record", record or groupwatch.Supervisor.record), \
                patch.object(groupwatch.Supervisor, "say", say or groupwatch.Supervisor.say), \
                patch.object(groupwatch.signal, "signal"):
            rc = groupwatch.run(["--state", self.state, *opts, "--", *cmd])
        for sv in made:  # the stand-in teardown killed nothing: end the child here, and close the log
            if sv.proc is not None and sv.proc.poll() is None:
                sv.proc.kill()
            if sv.proc is not None:
                sv.proc.wait()
            close(sv)
        log = Path(self.state, "groupwatch.log").read_text()
        return rc, log

    def test_successful_command_with_unclean_teardown_is_not_success(self):
        rc, log = self.run_gw(["true"], lambda *a: False)
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertIn("EXIT: command exited 0; clean=False", log)
        self.assertIn("FAIL: teardown not clean; exit 4 (would have been 0)", log)

    def test_failing_command_keeps_its_own_code_when_unclean(self):
        rc, _ = self.run_gw(["sh", "-c", "exit 7"], lambda *a: False)
        self.assertEqual(rc, 7)

    def test_a_teardown_that_raises_counts_as_unclean(self):
        def boom(*a):
            raise OSError("ps failed")
        rc, log = self.run_gw(["true"], boom)
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertIn("teardown failed: OSError: ps failed", log)
        self.assertIn("clean=False", log)

    def test_clean_runs_keep_their_codes(self):
        self.assertEqual(self.run_gw(["true"], lambda *a: True)[0], 0)
        self.assertEqual(self.run_gw(["sh", "-c", "exit 5"], lambda *a: True)[0], 5)

    def test_a_telemetry_failure_after_launch_still_tears_down_and_fails(self):
        # ps_table() runs ps with timeout=10: a TimeoutExpired from record() must not escape with the child left running
        calls = []
        def timeout(sv, root):
            raise subprocess.TimeoutExpired("ps", 10)
        def teardown(sv, root, pgid, reason):
            calls.append(reason)
            sv.proc.kill(); sv.proc.wait()  # what a real teardown does to the child
            return True
        rc, log = self.run_gw(["sleep", "30"], teardown, record=timeout)
        self.assertEqual(rc, groupwatch.MONITOR_FAILED)
        self.assertEqual(calls, ["monitor failed (TimeoutExpired)"])
        self.assertIn("monitor failed: TimeoutExpired", log)
        self.assertIn("ABORT: monitor failed (TimeoutExpired); clean=True", log)
        self.assertIn("FAIL: supervision failed; exit 5", log)

    def test_a_telemetry_failure_with_a_failing_teardown_kills_the_child_and_exits_4(self):
        def timeout(sv, root):
            raise subprocess.TimeoutExpired("ps", 10)
        def broken(*a):
            raise subprocess.TimeoutExpired("ps", 10)
        rc, log = self.run_gw(["sleep", "30"], broken, record=timeout)
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertIsNotNone(self.made[0].proc.returncode)  # the direct child was killed by the fallback, not left running
        self.assertIn("killed direct child", log)
        self.assertIn("ABORT: monitor failed (TimeoutExpired); clean=False", log)

    # a full disk: every log write fails (ENOSPC). Cleanup must not wait on the log, and the result is never 0

    @staticmethod
    def enospc(sv, msg):
        raise OSError(errno.ENOSPC, "No space left on device")

    def test_a_full_disk_never_turns_into_success(self):
        calls = []
        def teardown(sv, root, pgid, reason):
            calls.append(reason)
            return True
        rc, _ = self.run_gw(["true"], teardown, say=self.enospc)
        self.assertEqual(rc, groupwatch.MONITOR_FAILED)  # the command exited 0, but nothing could be logged
        self.assertEqual(len(calls), 1)  # its leftovers were still swept
        self.assertTrue(self.made[0].log_failed)

    def test_a_full_disk_and_a_telemetry_failure_still_tear_down(self):
        calls = []
        def timeout(sv, root):
            raise subprocess.TimeoutExpired("ps", 10)
        def teardown(sv, root, pgid, reason):
            calls.append(reason)
            sv.proc.kill(); sv.proc.wait()
            return True
        rc, _ = self.run_gw(["sleep", "30"], teardown, record=timeout, say=self.enospc)
        self.assertEqual(rc, groupwatch.MONITOR_FAILED)
        self.assertEqual(calls, ["monitor failed (TimeoutExpired)"])

    def test_a_full_disk_and_a_failing_teardown_still_kill_the_child(self):
        trigger = os.path.join(self.tmp.name, "stop-now")
        Path(trigger).write_text("")
        def broken(*a):
            raise subprocess.TimeoutExpired("ps", 10)
        rc, _ = self.run_gw(["sleep", "30"], broken, "--trigger-file", trigger, say=self.enospc)
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertIsNotNone(self.made[0].proc.returncode)  # killed by the fallback, before any log write

    def test_safe_teardown_runs_its_fallback_when_the_log_fails(self):
        os.makedirs(self.state)
        sv = groupwatch.Supervisor(self.state)
        self.addCleanup(close, sv)
        sv.proc = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: sv.proc.poll() is None and sv.proc.kill())
        def broken(*a):
            raise RuntimeError("teardown broke")
        with patch.object(groupwatch.Supervisor, "say", self.enospc), patch.object(groupwatch.Supervisor, "teardown", broken):
            self.assertIs(groupwatch.safe_teardown(sv, sv.proc.pid, sv.proc.pid, "test"), False)
        self.assertIsNotNone(sv.proc.returncode)

    def test_an_interrupt_after_launch_still_tears_down(self):
        calls = []
        def interrupted(sv, root):
            raise KeyboardInterrupt
        def teardown(sv, root, pgid, reason):
            calls.append(reason)
            sv.proc.kill(); sv.proc.wait()
            return True
        rc, _ = self.run_gw(["sleep", "30"], teardown, record=interrupted)
        self.assertEqual(rc, groupwatch.MONITOR_FAILED)
        self.assertEqual(calls, ["monitor failed (KeyboardInterrupt)"])

    def test_a_second_supervisor_on_the_same_state_starts_nothing(self):
        os.makedirs(self.state)
        first = groupwatch.Supervisor(self.state)  # holds the state directory's lock
        self.addCleanup(close, first)
        marker = os.path.join(self.tmp.name, "started")
        with self.assertRaises(SystemExit) as e:
            groupwatch.run(["--state", self.state, "--", "touch", marker])
        self.assertIn("in use by another supervisor", str(e.exception.code))
        self.assertFalse(os.path.exists(marker))
        close(first)
        rc, _ = self.run_gw(["touch", marker], lambda *a: True)  # free again once the first is done
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(marker))

    def test_an_abort_exits_3_when_clean_and_4_when_not(self):
        trigger = os.path.join(self.tmp.name, "stop-now")
        Path(trigger).write_text("")
        rc, log = self.run_gw(["sleep", "30"], lambda *a: True, "--trigger-file", trigger)
        self.assertEqual(rc, 3)
        self.assertIn("ABORT: trigger file; clean=True", log)
        rc, log = self.run_gw(["sleep", "30"], lambda *a: False, "--trigger-file", trigger)
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertIn("ABORT: trigger file; clean=False", log)


REAL_RUN = subprocess.run


def failing_ps(after=0, result=None):
    """subprocess.run as groupwatch calls it, except that ps answers `result` (exit 1, nothing printed) from call `after`."""
    calls = {"ps": 0}
    def fake(argv, *a, **k):
        if argv[0] == "ps":
            calls["ps"] += 1
            if calls["ps"] > after:
                return result or subprocess.CompletedProcess(argv, 1, "", "ps failed")
        return REAL_RUN(argv, *a, **k)
    return fake


class Telemetry(unittest.TestCase):
    """A process query that fails, or answers nothing usable, must never read as "every process is gone"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "gw")

    def ps_answers(self, rc, out):
        with patch.object(groupwatch.subprocess, "run", failing_ps(result=subprocess.CompletedProcess(["ps"], rc, out, "x"))):
            return groupwatch.ps_table()

    def test_ps_table_refuses_a_failed_or_unusable_answer(self):
        mine = REAL_RUN(["ps", "-o", "pid=,ppid=,pgid=,stat=,lstart=,args=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        self.assertIn(os.getpid(), groupwatch.ps_table())  # the real one, for contrast
        for rc, out, why in ((1, "", "nonzero exit, nothing printed"), (1, mine, "nonzero exit, output printed"),
                             (0, "", "exit 0, nothing printed"), (0, "garbage\n" + mine, "a line that does not parse"),
                             (0, "1 0 1 Ss Mon Oct 26 00:00:00 2026 /sbin/launchd\n", "a table without this process")):
            with self.assertRaises(getattr(groupwatch, "Telemetry", Exception), msg=why):  # an older groupwatch has no Telemetry
                self.ps_answers(rc, out)

    def test_memory_readings_from_a_failed_query_are_missing_not_numbers(self):
        out = "System-wide memory free percentage: 50%\n"
        with patch.object(groupwatch.subprocess, "run", lambda argv, *a, **k: subprocess.CompletedProcess(argv, 1, out + "used = 9.00M", "")):
            self.assertEqual(groupwatch.mac_memory(), (None, None))

    def supervisor(self):
        os.makedirs(self.state)
        sv = groupwatch.Supervisor(self.state)
        self.addCleanup(close, sv)
        sv.proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(lambda: sv.proc.poll() is None and sv.proc.kill())
        return sv

    def test_teardown_with_ps_failing_is_not_clean_and_kills_the_child(self):
        sv = self.supervisor()
        with patch.object(groupwatch.subprocess, "run", failing_ps()):
            ok = groupwatch.safe_teardown(sv, sv.proc.pid, sv.proc.pid, "test")
        self.assertIs(ok, False)
        self.assertIsNotNone(sv.proc.poll())  # the fallback killed it through its handle

    def test_teardown_never_calls_a_live_unrecorded_child_gone(self):
        sv = self.supervisor()  # its child was never recorded (as when every record() before teardown failed)
        with patch.object(groupwatch.Supervisor, "record", lambda sv, root: groupwatch.ps_table()):
            ok = sv.teardown(sv.proc.pid, sv.proc.pid, "test")
        self.assertIsNotNone(sv.proc.poll())  # signalled through its handle
        self.assertIs(ok, True)  # clean only because its exit is known

    def test_run_with_ps_failing_after_launch_fails_closed_and_stop_agrees(self):
        with patch.object(groupwatch.subprocess, "run", failing_ps(after=1)), patch.object(groupwatch.signal, "signal"):
            started = time.monotonic()
            rc = groupwatch.run(["--state", self.state, "--", "sleep", "20"])
        self.assertEqual(rc, groupwatch.UNCLEAN)
        self.assertLess(time.monotonic() - started, 15)  # torn down at once, not left to run out
        log = Path(self.state, "groupwatch.log").read_text()
        self.assertIn("ps exited 1", log)
        self.assertNotIn("clean=True", log)
        rows = Path(self.state, "supervisor.id").read_text().splitlines()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(groupwatch.verdict(self.state, int(rows[0]), rows[3]), 0)

    def test_an_armed_memory_guard_with_no_readings_tears_down(self):
        with patch.object(groupwatch, "mac_memory", lambda: (None, None)), patch.object(groupwatch.signal, "signal"):
            started = time.monotonic()
            rc = groupwatch.run(["--state", self.state, "--min-free-pct", "10", "--", "sleep", "25"])
        self.assertEqual(rc, 3)
        self.assertLess(time.monotonic() - started, 22)
        self.assertIn("ABORT: memory guard: no reading in 3 polls in a row; clean=True", Path(self.state, "groupwatch.log").read_text())

    def test_stop_with_ps_failing_is_not_success(self):
        os.makedirs(self.state)
        Path(self.state, "supervisor.id").write_text(f"{os.getpid()}\nx\nx\nt\n")
        with patch.object(groupwatch.subprocess, "run", failing_ps()), contextlib.redirect_stdout(io.StringIO()):
            try:
                rc = groupwatch.stop(["--state", self.state])
            except getattr(groupwatch, "Telemetry", Exception):
                rc = "raised"
        self.assertNotEqual(rc, 0)


class Program(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "gw")

    def start(self, *cmd):
        p = subprocess.Popen([sys.executable, "-I", GW, "run", "--state", self.state, "--", *cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: p.poll() is None and p.kill())
        for _ in range(100):  # until the supervisor has recorded its child
            if Path(self.state, "child.pid").exists() and "started child" in Path(self.state, "groupwatch.log").read_text():
                break
            time.sleep(0.1)
        time.sleep(1.2)
        return p

    def test_sigterm_tears_down_cleanly_and_exits_3(self):
        p = self.start("sleep", "60")
        child = int(Path(self.state, "child.pid").read_text())
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=60), 3)
        self.assertTrue(Path(self.state, "groupwatch.log").read_text().rstrip().endswith("ABORT: supervisor got SIGTERM; clean=True"))
        self.assertNotIn(child, groupwatch.ps_table())

    def test_stop_reports_a_clean_teardown_with_0(self):
        self.start("sleep", "60")
        r = subprocess.run([sys.executable, "-I", GW, "stop", "--state", self.state], capture_output=True, text=True, timeout=90)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertTrue(r.stdout.strip().endswith("clean=True"))

    def test_a_second_program_on_a_running_state_is_refused(self):
        self.start("sleep", "60")
        marker = os.path.join(self.tmp.name, "started")
        r = subprocess.run([sys.executable, "-I", GW, "run", "--state", self.state, "--", "touch", marker],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("in use by another supervisor", r.stderr)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(subprocess.run([sys.executable, "-I", GW, "stop", "--state", self.state], capture_output=True,
                                        timeout=90).returncode, 0)

    TOKEN = "4242-standin"

    def stand_in(self, final_lines, earlier=(), later=(), token=TOKEN):
        """A process that `stop` accepts as the recorded supervisor (pid, start time, argv, run token), with a log this
        test writes: `earlier` and `later` lines are other runs' (other tokens), around this run's own lines."""
        os.makedirs(self.state)
        p = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)", "groupwatch-stand-in"])
        self.addCleanup(lambda: p.poll() is None and p.kill())
        me = None
        for _ in range(50):
            me = groupwatch.ps_table().get(p.pid)
            if me and "groupwatch-stand-in" in me[3]:
                break
            time.sleep(0.1)
        assert me, "the stand-in supervisor did not start"
        Path(self.state, "supervisor.id").write_text(f"{p.pid}\n{me[2]}\n{me[3]}\n" + (f"{token}\n" if token else ""))
        own = [f"2026-10-25T00:00:00 supervisor pid {p.pid} started child pid 1 pgid 1: sleep 60", *final_lines]
        own = [f"{l.split(' ', 1)[0]} [{self.TOKEN}] {l.split(' ', 1)[1]}" for l in own]
        lines = [*earlier, *own[:1], *later, *own[1:]]  # another run's lines may fall between this run's
        Path(self.state, "groupwatch.log").write_text("".join(l + "\n" for l in lines))
        return p

    def stop(self):
        return subprocess.run([sys.executable, "-I", GW, "stop", "--state", self.state], capture_output=True, text=True, timeout=90)

    def test_stop_is_not_success_when_the_supervisor_logged_clean_false(self):
        self.stand_in(["2026-10-25T00:00:01 ABORT: supervisor got SIGTERM; clean=False"])
        r = self.stop()
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("without a clean teardown", r.stdout)

    def test_stop_is_not_success_when_the_supervisor_vanished_without_a_final_line(self):
        # an earlier run in the same state directory ended clean; that line must not vouch for this supervisor
        self.stand_in([], earlier=["2026-10-24T00:00:00 [1-old] supervisor pid 1 started child pid 2 pgid 2: x",
                                   "2026-10-24T00:00:09 [1-old] EXIT: command exited 0; clean=True"])
        r = self.stop()
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("left no final line", r.stdout)

    def test_stop_is_not_success_when_a_fail_line_follows_a_clean_abort(self):
        self.stand_in(["2026-10-25T00:00:01 monitor failed: TimeoutExpired: ps",
                       "2026-10-25T00:00:02 ABORT: monitor failed (TimeoutExpired); clean=True",
                       "2026-10-25T00:00:02 FAIL: supervision failed; exit 5"])
        r = self.stop()
        self.assertNotEqual(r.returncode, 0, r.stdout)

    def test_stop_never_takes_another_supervisors_final_line(self):
        # supervisor A starts, supervisor B starts in the same state directory and ends clean; A has no final line
        self.stand_in([], later=["2026-10-25T00:00:01 [4343-other] supervisor pid 4343 started child pid 7 pgid 7: x",
                                 "2026-10-25T00:00:02 [4343-other] EXIT: command exited 0; clean=True"])
        r = self.stop()
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("left no final line", r.stdout)

    def test_stop_takes_its_own_final_line_between_another_runs_lines(self):
        # A's clean final line, with B's unclean lines around it: A's verdict is A's
        self.stand_in(["2026-10-25T00:00:03 ABORT: supervisor got SIGTERM; clean=True"],
                      later=["2026-10-25T00:00:01 [4343-other] supervisor pid 4343 started child pid 7 pgid 7: x",
                             "2026-10-25T00:00:02 [4343-other] EXIT: command exited 0; clean=False",
                             "2026-10-25T00:00:02 [4343-other] FAIL: teardown not clean; exit 4 (would have been 0)"])
        r = self.stop()
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_stop_without_a_run_token_is_not_success(self):
        self.stand_in(["2026-10-25T00:00:01 ABORT: supervisor got SIGTERM; clean=True"], token=None)
        r = self.stop()
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("no run token", r.stdout)

    def test_stop_accepts_this_supervisors_clean_final_line(self):
        self.stand_in(["2026-10-25T00:00:01 ABORT: supervisor got SIGTERM; clean=True"])
        r = self.stop()
        self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == "__main__":
    unittest.main()
