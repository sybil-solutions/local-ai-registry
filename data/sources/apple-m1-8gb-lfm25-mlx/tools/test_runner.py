#!/usr/bin/env python3
"""macmini-run3.sh reads only files the committed layout has, and refuses macOS below 15 before any setup step runs.

  python3 -m unittest test_runner      (from this directory; macOS, since the runner calls sw_vers, pmset and launchctl)

The guard cases build a task directory the way the README says (tools/ and runtime/ copied from this checkout), with a
stub sdk-setup.sh that only leaves a marker, and a stub sw_vers first on PATH.
"""
import os, re, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE)  # data/sources/apple-m1-8gb-lfm25-mlx
RUNNER = os.path.join(HERE, "macmini-run3.sh")
GENERATED = {"sdk-setup.sh", "sdk-start.argv", "labtree", "runs/", ".regression.XXXXXX"}  # made per task, documented


class LaunchFlags(unittest.TestCase):
    def test_the_regression_parses_the_launch_flags(self):
        # runtime/test_unlimited_max_tokens.py parses LAUNCH_FLAGS with each server copy's own argparse block; they must be
        # the committed launch's server flags (after the launcher script), or the regression proves another configuration
        import ast, json
        launch = json.loads(Path(SOURCE, "..", "..", "..", "registry", "launches", "mlx-lm-unlimited-rt312-lfm2.5-2.6b-4bit-32k-m1.json").read_text())
        src = Path(SOURCE, "runtime", "test_unlimited_max_tokens.py").read_text()
        flags = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and n.targets[0].id == "LAUNCH_FLAGS")
        cmd = launch["command"]
        self.assertEqual(cmd[cmd.index("rt312/mlx_lm_unlimited_serve.py") + 1:], flags)
        self.assertEqual(launch["ctx"], int(flags[flags.index("--max-kv-size") + 1]))  # ctx is the window the server holds


class Layout(unittest.TestCase):
    def test_every_file_the_runner_reads_is_committed_or_generated(self):
        refs = sorted(set(re.findall(r"\$D/([A-Za-z0-9_./-]+)", Path(RUNNER).read_text())))
        self.assertTrue(refs)
        missing = [r for r in refs if r not in GENERATED and not os.path.exists(os.path.join(SOURCE, r))]
        self.assertEqual(missing, [])
        self.assertTrue(all(r.startswith(("tools/", "runtime/")) for r in refs if r not in GENERATED))


@unittest.skipUnless(sys.platform == "darwin", "the runner is macOS-only")
class MacOSGuard(unittest.TestCase):
    def task(self, version):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        shutil.copytree(HERE, os.path.join(d, "tools"), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(SOURCE, "runtime"), os.path.join(d, "runtime"))
        Path(d, "sdk-setup.sh").write_text(f"touch {d}/SETUP_RAN\n")
        stub = os.path.join(d, "stub")
        os.makedirs(stub)
        Path(stub, "sw_vers").write_text(f"#!/bin/sh\necho '{version}'\n")
        os.chmod(os.path.join(stub, "sw_vers"), 0o755)
        r = subprocess.run(["bash", os.path.join(d, "tools", "macmini-run3.sh"), "setup", "t"], capture_output=True, text=True,
                           env={**os.environ, "PATH": f"{stub}:/usr/bin:/bin:/usr/sbin:/sbin"}, timeout=600)
        return d, r

    def test_older_macos_stops_before_any_setup_step(self):
        for version in ("14.6", "11.7.10", "not a version"):
            d, r = self.task(version)
            self.assertEqual(r.returncode, 3, (version, r.stderr))
            self.assertIn("ABORT", r.stderr, version)
            self.assertFalse(os.path.exists(os.path.join(d, "SETUP_RAN")), version)
            self.assertFalse(os.path.exists(os.path.join(d, "runs")), version)  # nothing made at all

    def test_macos_15_and_later_proceed(self):
        for version in ("15.0", "26.4.1"):
            d, r = self.task(version)
            self.assertIn(f"macOS {version}: at least 15, ok", r.stdout, version)
            self.assertTrue(os.path.exists(os.path.join(d, "SETUP_RAN")), (version, r.stderr[-500:]))


if __name__ == "__main__":
    unittest.main()
