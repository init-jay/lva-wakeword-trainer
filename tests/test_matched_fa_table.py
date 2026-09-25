"""Pins for tools/matched_fa_table.py: one grid, a derived budget, a real override.

WHY THESE EXIST. The tool once re-declared the sweep grid (a copy of
eval/src/eval_model.py's SWEEP_GRID) and defaulted --budgets to the literal
"6,12" - counts that are only right for one adversarial set's size. A 298-clip
set derives 5 under the repo's 2% constraint, so the out-of-the-box reading sat
on the wrong side of the budget it exists to hold. Both drifts are pinned here
so they cannot come back.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

import eval_model as ev        # noqa: E402
import matched_fa_table as mft  # noqa: E402
import score_margins as sm     # noqa: E402


def _csv(root, model, n_adv):
    """A score_margins.py dump: header, one positive row, n_adv adversarial rows."""
    path = root / f"{model}.csv"
    with open(path, "w") as fh:
        fh.write("model,set,clip,peak\n")
        fh.write(f"{model},pos/alice,a1,0.900000\n")
        for i in range(n_adv):
            fh.write(f"{model},neg/extend,a{i},0.3{i:03d}\n")
    return path


def test_the_grid_is_the_imported_sweep_grid_not_a_local_copy():
    # Identity, not equality: a re-declared copy would compare equal but still
    # drift from the sweep - identity is what fails on a re-declaration.
    assert mft.GRID is ev.SWEEP_GRID, "must import ev.SWEEP_GRID, not re-declare it"
    assert mft.GRID == [round(0.05 * i, 2) for i in range(1, 20)]


def test_the_derived_default_budget_tracks_the_adversarial_set_size():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        derived = {}
        for n, model in ((50, "m_small"), (298, "m_large")):
            _, _, adv = mft.load(str(_csv(d, model, n)))
            assert len(adv) == n, f"fixture should carry {n} adversarial rows"
            derived[n] = mft.resolve_budgets(adv, None)
            assert derived[n] == [sm.default_budget(n)], (derived[n], n)
    # A literal default would not do this: the derived budget changes with set size.
    assert derived[50] != derived[298]


def test_explicit_budgets_override_the_derived_default():
    adv = [0.3] * 298  # derives 5; the override must win
    assert mft.resolve_budgets(adv, [6, 12]) == [6, 12]


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
