"""Plain-python fallback runner for the tests in this directory.

No venv in this repo carries pytest (checked 2026-09: train-applesilicon,
train-mww-applesilicon, preflight, tts-service), so the suite doubles as
plain `python tests/test_x.py` runs: every test is a module-level
`test_*()` function holding plain asserts, which pytest would also collect
if a venv ever gained the dependency. `run()` executes them all in sorted
name order and exits non-zero on any failure.
"""

import sys
import traceback


def run(module) -> None:
    tests = [v for k, v in sorted(vars(module).items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except BaseException:                                    # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{module.__name__}: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
