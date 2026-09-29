"""Guards for src/train/mww/manifest.py: the ESPHome manifest cutter.

choose_cutoff picks the LOWEST cutoff whose faph is within budget (most
sensitive operating point that meets it) and, since 2026-09-30, refuses a
CHOSEN cutoff above the degenerate-init threshold (DEFAULT_MAX_CUTOFF, 0.70).
The refusal exists because the faph budget alone cannot see the firehose
shape: a weak-ambient training INIT needs an abnormally high threshold to get
its ambient false-accepts to budget, so its lowest in-budget cutoff lands far
up the curve while its recall still looks respectable - 0.88 with 83% recall
- and the manifest shipped as if it were a candidate. The held-corpus retrains
pinned the variable: same corpus c2a0b135, the unseeded init chose 0.88 while
seeds 111/222 chose 0.25/0.29 (94-95% recall); across the measured manifests
the healthy inits sit at 0.09-0.62 and the degenerate ones at 0.73-0.96, and
0.70 is the threshold measured in that gap.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import train.mww.manifest as man  # noqa: E402

# ROC rows are (cutoff, frr, faph), the parse_roc order.

# The healthy reference (f4021da-c85a0dc0, 2026-09-30 05:20): zero FAs at 0.66,
# 0.187 at 0.59. Lowest in-budget cutoff is 0.59 - well inside the gate.
HEALTHY = [
    (0.59, 0.1252, 0.187),
    (0.66, 0.1377, 0.000),
    (0.80, 0.2500, 0.000),
]

# The firehose reference (f4021da-c2a0b135, unseeded init, same day): zero FAs
# only at 0.99, 0.187 at 0.88 - the operating point the old gate shipped.
FIREHOSE = [
    (0.88, 0.1699, 0.187),
    (0.84, 0.1574, 0.375),
    (0.99, 0.2970, 0.000),
]


def test_healthy_curve_is_chosen_unchanged():
    # The gate must not move a healthy operating point: the same 0.59 the
    # un-gated code chose.
    assert man.choose_cutoff(HEALTHY, 0.2) == (0.59, 0.1252, 0.187)


def test_firehose_curve_is_refused_by_default():
    # The old code returned (0.88, ...) here - a deployable-looking manifest
    # for a model that needs a 0.88 threshold to suppress ambient.
    try:
        man.choose_cutoff(FIREHOSE, 0.2)
    except SystemExit as e:
        msg = str(e)
        assert "0.88" in msg and "0.7" in msg      # the cutoff and the threshold
        assert "seed" in msg                        # the remediation names the variable
    else:
        raise AssertionError("firehose curve was accepted")


def test_threshold_sits_in_the_measured_gap():
    # 0.62 was the highest healthy cutoff measured; 0.73 the lowest
    # degenerate one. The default must separate them, not the other way
    # round - a regression here re-ships firehoses (or re-refuses healthy
    # candidates) silently.
    assert man.DEFAULT_MAX_CUTOFF > 0.62
    assert man.DEFAULT_MAX_CUTOFF < 0.73


def test_explicit_higher_threshold_records_it_deliberately():
    # The escape hatch: the operator can record a degenerate init on purpose
    # (it is still a model; sometimes the measurement is the point).
    assert man.choose_cutoff(FIREHOSE, 0.2, max_cutoff=0.95) == (0.88, 0.1699, 0.187)


def test_synthetic_frr1_row_still_discarded():
    # The pre-existing guard: microwakeword appends the (faph 0, frr 1)
    # plotting terminator when no cutoff reaches the floor - it must not win.
    rows = FIREHOSE + [(1.0, 1.0, 0.0)]
    try:
        man.choose_cutoff(rows, 0.2)
    except SystemExit as e:
        assert "0.88" in str(e)                     # refused as firehose, not by the frr-1 guard
    else:
        raise AssertionError("firehose curve was accepted")


def test_all_frr1_rows_refused_as_detects_nothing():
    try:
        man.choose_cutoff([(1.0, 1.0, 0.0)], 0.2)
    except SystemExit as e:
        assert "detects nothing" in str(e)
    else:
        raise AssertionError("an all-frr-1 ROC was accepted")


def test_no_budget_met_still_refused_with_lowest_measured():
    try:
        man.choose_cutoff([(0.5, 0.5, 1.0)], 0.2)
    except SystemExit as e:
        assert "no cutoff achieves faph <= 0.2" in str(e)
    else:
        raise AssertionError("an over-budget ROC was accepted")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
