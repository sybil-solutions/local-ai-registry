#!/usr/bin/env python3
"""Run test_unlimited_max_tokens.py and then report whether any mlx module was imported (it must not be: CPU-only).
  run_regression_cpu.py PRISTINE_SERVER_PY PATCHED_SERVER_PY   -> prints the test output, "exit <code>", "mlx modules loaded: ..." """
import os, runpy, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["test_unlimited_max_tokens.py"] + sys.argv[1:3]
code = 0
try:
    runpy.run_path(os.path.join(here, "test_unlimited_max_tokens.py"), run_name="__main__")
except SystemExit as e:
    code = e.code
print("exit", code)
print("mlx modules loaded:", sorted(m for m in sys.modules if m == "mlx" or m.startswith(("mlx.", "mlx_lm"))) or "none")
sys.exit(code)
