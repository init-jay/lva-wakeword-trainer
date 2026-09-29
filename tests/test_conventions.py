"""Suite hygiene: every test file actually runs, and none imports pytest.

`make test` collects the suite with pytest from the tests project's env
(tests/pyproject.toml); the plain-assert, pytest-free style is what keeps
the no-uv fallback - `python tests/test_x.py` via tests/_runner.py -
working. Two failure shapes make a test file a no-op without failing:
one that imports pytest dies on its import, and under the plain-python
fallback that SILENTLY SKIPS every file after it in alphabetical order:
on 2026-09-22 that was 7 of 10 files, including test_sweep.py (the guard
on the grid fix), because test_ledger.py had been written with fixtures
and capsys. The 12 tests it carried were well designed and had never
executed. And one with no `main()` runner imports fine and exits 0 having
run NOTHING: that was test_paths.py, whose seven layout guards never
executed until 7e6d343 repaired it - nothing detected it because the file
looked indistinguishable from a healthy one in review. These guards make
the convention an act, not a memory: the moment a file reaches for pytest
or loses its runner, the suite says so here instead of going green on
phantoms."""

import sys
from pathlib import Path


HERE = Path(__file__).parent


def test_no_test_file_imports_pytest():
    bad = [p.name for p in sorted(HERE.glob("test_*.py"))
           if p.name != Path(__file__).name
           and "import pytest" in p.read_text()]
    assert not bad, (
        "pytest import(s) - the no-uv fallback `python tests/test_x.py` "
        f"would die on the first of these and skip every file after it: {bad}")


def test_every_test_file_actually_runs():
    # The runner convention is `def main(): import _runner; _runner.run(...)`
    # under `if __name__ == "__main__"`. Without it, `python tests/test_x.py`
    # imports the module, runs zero tests, exits 0 - the suite goes green on a
    # file that guards nothing. Both markers, so a main() nobody calls is
    # refused too.
    bad = [p.name for p in sorted(HERE.glob("test_*.py"))
           if p.name != Path(__file__).name
           and ("def main(" not in p.read_text()
                or 'if __name__ == "__main__"' not in p.read_text())]
    assert not bad, (
        "test file(s) with no __main__ runner - the no-uv fallback "
        f"`python tests/test_x.py` would import and exit 0, running nothing: {bad}")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
