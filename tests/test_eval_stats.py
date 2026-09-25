"""Guards for the eval harness statistics: threshold_for_fa and
bootstrap_diff_ci (src/eval/src/compare_models.py), wilson_interval and
threshold_sweep (src/eval/src/eval_model.py).

These are the arithmetic behind the repo's two measurement rules: never
compare models at a fixed threshold (two runs of an identical config
measured 77% and 67% at 0.5), and never pool per-speaker or per-category
results - a verdict that a wrong formula could print as "better".
threshold_for_fa's contract is exact-count, and the bootstrap is what
gates the "best model" line on real evidence instead of a coin flip
(compare_models main()).

IMPORT: on the host, eval/ has no __init__.py - the eval IMAGE mounts
src/eval/src as the package `eval` (src/eval/docker-compose.yml), and both
compare_models.py and eval_model.py do `from eval import backends, ...`,
which only resolves in that layout. We replicate it here: register a bare
module named `eval` whose __path__ is src/eval/src, which is exactly what the
mount does at runtime. Nothing outside this process is touched.
"""

import random
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if "eval" not in sys.modules:
    _pkg = types.ModuleType("eval")
    _pkg.__path__ = [str(REPO_ROOT / "src" / "eval" / "src")]
    sys.modules["eval"] = _pkg

from eval.compare_models import bootstrap_diff_ci, threshold_for_fa  # noqa: E402
from eval.eval_model import SWEEP_GRID, threshold_sweep, wilson_interval  # noqa: E402


# ---------------------------------------------------------------------------
# threshold_for_fa
# ---------------------------------------------------------------------------

def test_threshold_admits_exactly_count_clips_at_or_above():
    # Distinct scores, so the contract "exactly count at-or-above" is
    # checkable without tie ambiguity: the threshold sits one epsilon
    # above the (count+1)-th highest score (0.3 here, not 0.4 - the 0.4
    # itself is one of the two admitted).
    adv = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    thr = threshold_for_fa(adv, 2)
    assert int((adv >= thr).sum()) == 2
    assert int((adv < thr).sum()) == 3
    assert abs(thr - 0.3) < 2e-6
    assert thr > 0.3


def test_threshold_is_0p0_when_count_reaches_or_exceeds_the_set_size():
    # count >= len admits everything and has no meaningful threshold, so
    # it reports 0.0 rather than slicing past the end of the sorted array.
    adv = np.array([0.3, 0.6, 0.9])
    assert threshold_for_fa(adv, 3) == 0.0      # count == len
    assert threshold_for_fa(adv, 5) == 0.0      # count > len


# ---------------------------------------------------------------------------
# bootstrap_diff_ci
# ---------------------------------------------------------------------------

def _ci(pos_w, adv_w, pos_s, adv_s, count, n_boot=200):
    return bootstrap_diff_ci(pos_w, adv_w, pos_s, adv_s, count,
                             random.Random(0), n_boot=n_boot)


def test_bootstrap_is_identical_for_a_fixed_seed():
    adv_w = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
    adv_s = np.array([0.1, 0.3, 0.6, 0.8, 0.95])
    pos_w = {"plain": np.array([0.8, 0.9, 0.85])}
    pos_s = {"plain": np.array([0.7, 0.8, 0.75])}
    # Two independent Random(0) streams: the draws must be the same noise,
    # which is what makes a rerun read the same intervals (module constant
    # N_BOOTSTRAP docstring).
    a = _ci(pos_w, adv_w, pos_s, adv_s, 2)
    b = _ci(pos_w, adv_w, pos_s, adv_s, 2)
    assert a == b


def test_identical_models_give_zero_centred_ci_including_zero():
    # Same adversarial scores for both models: the paired resamples give
    # the same operating point to both, so the point estimate is exactly
    # zero and the interval cannot exclude it.
    adv = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    pos = {"plain": np.array([0.8, 0.9, 0.85])}
    point, lo, hi = _ci({"plain": pos["plain"]}, adv,
                        {"plain": pos["plain"]}, adv, 2)
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_strictly_better_model_gives_clearly_positive_point_estimate():
    # Clearly separated at EVERY matched operating point: w's positives
    # score above ALL of w's adversarial scores (so its matched threshold
    # can never exclude them), while s's positives score below ALL of s's
    # (its matched threshold, derived from a with-replacement resample of
    # the adversarial set, always lands above them). Every bootstrap draw
    # therefore reads 1.0 vs 0.0 - no draw can drag the CI back to zero.
    adv_w = np.linspace(0.5, 0.7, 10)
    adv_s = np.linspace(0.2, 0.4, 10)
    pos_w = {"plain": np.full(4, 0.9)}
    pos_s = {"plain": np.full(4, 0.1)}
    point, lo, hi = _ci(pos_w, adv_w, pos_s, adv_s, 2)
    assert point > 0.5
    assert lo > 0.0 and hi > 0.0
    assert lo <= point <= hi
    # count >= set size is the guarded "not enough negatives" case.
    assert bootstrap_diff_ci(pos_w, adv_w, pos_s, adv_s, 10,
                             random.Random(0), n_boot=2) is None


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------

def test_wilson_interval_known_values_3_of_6():
    lo, hi = wilson_interval(3, 6)
    assert abs(lo - 0.188) < 0.01
    assert abs(hi - 0.812) < 0.01


def test_wilson_interval_zero_hits_clamps_lower_bound_to_zero():
    # The formula overshoots into negative territory by a rounding hair at
    # 0/n; the interval is on a proportion and must clamp to [0, 1].
    lo, hi = wilson_interval(0, 6)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def test_wilson_interval_all_hits_clamps_upper_bound_to_one():
    lo, hi = wilson_interval(6, 6)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_interval_zero_n_is_none():
    assert wilson_interval(0, 0) is None


# ---------------------------------------------------------------------------
# threshold_sweep (bug.md C3 step 2, 2026-09-22: the 40-eval manual job)
# ---------------------------------------------------------------------------

def test_threshold_sweep_re_thresholds_peaks_hand_computed():
    # Detection at t iff the peak is >= t: first_crossing fires on ANY frame
    # >= t, and the peak is the max over frames, so re-thresholding the peak
    # is a pure filter - no model re-run. Tiny hand-computed case:
    # 3 positives, 4 adversarial, two grid points.
    pos = [0.2, 0.6, 0.9]
    adv = [0.1, 0.5, 0.7, 0.95]
    s = threshold_sweep(pos, adv, [0.5, 0.9])
    assert s["thresholds"] == [0.5, 0.9]
    assert s["positives_rate"] == [2 / 3, 1 / 3]  # >=0.5: 2/3; >=0.9: 1/3
    assert s["adv_rate"] == [3 / 4, 1 / 4]       # >=0.5: 3/4; >=0.9: 1/4
    assert s["pos_n"] == 3 and s["adv_n"] == 4


def test_sweep_grid_is_19_points_and_cross_checks_the_recorded_threshold():
    # 0.05..0.95 step 0.05. 0.5 is on the grid on purpose: the sweep's own
    # 0.5 point must agree with the single-threshold reading the same run
    # printed, so the two can never drift apart silently.
    assert len(SWEEP_GRID) == 19
    assert SWEEP_GRID[0] == 0.05 and SWEEP_GRID[-1] == 0.95
    assert 0.5 in SWEEP_GRID


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
