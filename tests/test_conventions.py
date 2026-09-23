"""Suite hygiene: no test file may import pytest.

The `make test` target runs each file with plain python (tests/_runner.py,
and no venv in this repo carries pytest). A file that imports pytest dies
on its import and stops the loop - and SILENTLY SKIPS every file after it
in alphabetical order: on 2026-09-22 that was 7 of 10 files, including
test_sweep.py (the guard on the B1 grid fix), because test_ledger.py had
been written with fixtures and capsys. The 12 tests it carried were well
designed and had never executed. This guard makes the convention an act,
not a memory: the moment a file reaches for pytest, the suite says so at
that file instead of skipping the rest.
"""

import sys
from pathlib import Path


def test_no_test_file_imports_pytest():
    here = Path(__file__).parent
    bad = [p.name for p in sorted(here.glob("test_*.py"))
           if p.name != Path(__file__).name
           and "import pytest" in p.read_text()]
    assert not bad, (
        "pytest import(s) - `make test` would stop at the first of these "
        f"and skip every file after it (tests/_runner.py): {bad}")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
