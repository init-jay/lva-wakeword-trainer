"""Plain-python fallback runner for the tests in this directory.

`make test` runs the suite with pytest from the tests project's own env
(tests/pyproject.toml); on a machine without uv, every file still runs as
plain `python tests/test_x.py`: each test is a module-level `test_*()`
function holding plain asserts, which is also exactly what pytest collects
unchanged. `run()` executes them all in sorted name order and exits
non-zero on any failure.
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
