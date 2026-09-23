"""Guards for train/ledger.py's summarise, in the spirit of bug.md round 2
(2026-09-22): the runs were sound, the tool that summarised them was not.

C1: group on the RESOLVED config, not the grid label. The 094e414 sweep ran
before the grid was threaded into the command, so four rows carried 25k
labels around runs that were actually filed at 50k (config.steps), and
grouping on the label printed an 80.9% "25k" mean that was really a
25k/50k mix - contradicting the hand-built 73.5% verdict in improvement.md.

C2: n counts (config-hash, seed) PAIRS, not records. Two records sharing the
tag's h-half (the config hash, train/provenance.py) and the seed are the
same run computed at two commits - not two draws from a distribution. Exact
duplicates collapse; duplicates whose evals differ are a determinism
regression and must be named, not averaged over.

C3 step 1: the columns carry the recorded eval threshold (@0.5, or `mixed`),
because the rates are one-threshold readings and CLAUDE.md forbids reading
them as a matched-FA comparison.

C3 step 2: when records carry eval_model.py's threshold_sweep (the
fixed-grid re-threshold of that run's own per-clip peaks), the table reads
a COMMON matched-FA budget: detection at at-most-B adversarial FA is a
point pick on each group's own step-function curve, never an
interpolation. B is the median across swept groups of each group's own
median recorded-threshold FA. Pre-sweep records (every existing one) keep
the @-threshold reading, and with no sweep on file at all the output is
byte-identical to the pre-sweep table - pinned against the real ledger's
golden output, since the append-only ledger holds both vintages.

No pytest (tests/_runner.py): this file was written with fixtures and
capsys and the `make test` target - plain python, one file at a time, no
pytest in any venv in this repo - died on its import and silently skipped
the six files after it in the suite. Converted 2026-09-22 to the house
convention: tempfile for the scratch ledger, redirect_stderr for the
warning capture, an early return for the fresh-checkout skip.
"""

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import ledger  # noqa: E402


@contextlib.contextmanager
def tmp_ledger(records):
    """Point ledger.ledger_path at a scratch file so summarise reads a
    synthetic ledger; the append-only real file is never touched (it is
    history, not a fixture)."""
    real = ledger.ledger_path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs.jsonl"
        _write(path, records)
        ledger.ledger_path = lambda wake_word: path
        try:
            yield path
        finally:
            ledger.ledger_path = real


@contextlib.contextmanager
def _err():
    """capsys.readouterr().err, without pytest: summarise's warnings go to
    `file=sys.stderr`, which print resolves at call time, so swapping the
    sys.stderr object catches them."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def rec(tag, steps_label, config_steps, seed, det, fa, threshold=0.5):
    return {
        "target": "oww",
        "tag": tag,
        "corpus_id": "f9c065b",
        "seed": seed,
        "grid": {"training-steps": steps_label},
        "config": {"steps": config_steps},
        "eval_block": {
            "threshold": threshold,
            "adversarial": {"rate": fa},
            "positives": {"rate": det},
        },
    }


def _sweep(thresholds, adv_rate, pos_rate, adv_n=298, pos_n=51):
    """An eval threshold_sweep block in the shape eval_model.py writes it:
    per-grid-point rates, adversarial axis = extend+hey_other only."""
    return {
        "thresholds": list(thresholds),
        "adv_rate": list(adv_rate),
        "positives_rate": list(pos_rate),
        "adv_n": adv_n,
        "pos_n": pos_n,
    }


def rec_sweep(tag, steps, seed, det, fa, curve, threshold=0.5):
    r = rec(tag, steps, steps, seed, det, fa, threshold=threshold)
    r["eval_block"]["threshold_sweep"] = curve
    return r


def test_same_label_different_config_splits_and_warns():
    # C1: two records whose grid labels agree but whose resolved configs do
    # not must land in SEPARATE groups, with a stderr warning that names the
    # offending tag, the label and the resolved value.
    with tmp_ledger([
        rec("aaa1111-cf9c065b-haaaaaa", 25000, 25000, 42, 0.70, 0.02),
        rec("bbb2222-cf9c065b-hbbbbbb", 25000, 50000, 43, 0.90, 0.04),
    ]), _err() as err:
        out = ledger.summarise("hey seeree")
    err = err.getvalue()
    assert "training-steps=25000" in out
    assert "training-steps=50000" in out
    assert out.count("n=1 ") == 2, "one record per group: %r" % out
    assert "bbb2222-cf9c065b-hbbbbbb" in err
    assert "25000" in err and "50000" in err
    # the agreeing record does not warn
    assert "aaa1111" not in err


def test_record_without_config_falls_back_to_label():
    # No resolved config on file (predates the config half): the label is
    # the only evidence, grouping keeps it, and there is nothing to warn
    # against.
    r = rec("aaa1111-cf9c065b-haaaaaa", 25000, 25000, 42, 0.70, 0.02)
    r["config"] = None
    with tmp_ledger([r]), _err() as err:
        out = ledger.summarise("hey seeree")
    assert "training-steps=25000" in out
    assert err.getvalue() == ""


def test_exact_duplicates_collapse_to_distinct_pairs():
    # C2: four records, two (config-hash, seed) pairs - the real 50k group
    # shape (h94736bd/1042 and h642a48d/1043, each at two commits, agreeing
    # to the last digit). n is 2, with the run count alongside, and the
    # statistics come from the two distinct pairs.
    with tmp_ledger([
        rec("aaa1111-dirty-cf9c065b-h94736bd", 50000, 50000, 1042,
            0.9019607843137255, 0.03691275167785235),
        rec("ccc3333-dirty-cf9c065b-h94736bd", 50000, 50000, 1042,
            0.9019607843137255, 0.03691275167785235),
        rec("ddd4444-cf9c065b-h642a48d", 50000, 50000, 1043,
            0.7843137254901961, 0.03691275167785235),
        rec("ccc3333-dirty-cf9c065b-h642a48d", 50000, 50000, 1043,
            0.7843137254901961, 0.03691275167785235),
    ]), _err() as err:
        out = ledger.summarise("hey seeree")
    assert "n=2 (4 runs)" in out
    assert "84.3% [78.4-90.2]" in out
    assert "adv FA@0.5" in out and "detection@0.5" in out
    assert err.getvalue() == ""


def test_divergent_duplicates_are_loud():
    # C2's other half: the same (config-hash, seed) with DIFFERENT eval
    # numbers is a determinism regression - name both tags and the
    # differing fields; do not collapse the disagreement into one number.
    with tmp_ledger([
        rec("aaa1111-cf9c065b-h94736bd", 50000, 50000, 1042, 0.9, 0.03),
        rec("ccc3333-cf9c065b-h94736bd", 50000, 50000, 1042, 0.67, 0.03),
    ]), _err() as err:
        out = ledger.summarise("hey seeree")
    err = err.getvalue()
    assert "DETERMINISM REGRESSION" in err
    assert "aaa1111-cf9c065b-h94736bd" in err
    assert "ccc3333-cf9c065b-h94736bd" in err
    assert "positives.rate" in err
    # both values stay in the table: the pair did NOT collapse, so both
    # show in the spread (the exact-duplicate path would have shown one)
    assert "78.5% [67.0-90.0]" in out
    # n still counts the one distinct pair, with the run count alongside
    assert "n=1 (2 runs)" in out


def test_mixed_thresholds_say_mixed():
    # C3 step 1: one-threshold readings labelled per group; a group whose
    # records were eval'd at different thresholds cannot pretend to be one
    # threshold and says so instead.
    with tmp_ledger([
        rec("aaa1111-cf9c065b-haaaaaa", 25000, 25000, 42, 0.7, 0.02,
            threshold=0.5),
        rec("bbb2222-cf9c065b-hbbbbbb", 25000, 25000, 43, 0.8, 0.01,
            threshold=0.4),
    ]):
        out = ledger.summarise("hey seeree")
    assert "adv FA@mixed" in out
    assert "detection@mixed" in out


def test_summary_carries_the_matched_fa_caveat():
    # C3 step 1: the standing caveat under the table - the numbers cannot
    # be read as a matched-FA comparison (CLAUDE.md).
    with tmp_ledger([
        rec("aaa1111-cf9c065b-haaaaaa", 25000, 25000, 42, 0.7, 0.02),
        rec("bbb2222-cf9c065b-hbbbbbb", 50000, 50000, 43, 0.8, 0.01),
    ]):
        out = ledger.summarise("hey seeree")
    assert "one-threshold readings, not a matched-FA comparison" in out
    assert "Never compare models" in out


def test_real_ledger_groups_on_resolved_steps():
    # Invariants, not snapshot numbers: the ledger is APPEND-only and growth
    # is the norm, so a pin that breaks on the next appended row teaches the
    # wrong lesson (the old version pinned n=2 / 73.5% on the 25k group, and
    # any row appended - the 2026-09-23 corpus_axes sweep was the first -
    # broke it). What must hold forever: C1's self-correction names the two
    # mislabeled 094e414 rows (25k label, 50k config); both resolved step
    # groups exist; the 50k group's numbers stay EXACTLY pinnable because
    # no sweep row has config.steps=50000 (checked 2026-09-23 against the
    # then-current file); a 25k group spanning more than one corpus_id warns
    # (the sweep filed config-equal rows across four corpora); the sweep
    # arms render by grid label.
    if not ledger.ledger_path("hey seeree").is_file():
        print("  skip: no ledger at "
              f"{ledger.ledger_path('hey seeree')} (fresh checkout)")
        return
    with _err() as err:
        out = ledger.summarise("hey seeree")
    err = err.getvalue()
    # C1's self-correction, ledger-growth-proof: both mislabeled rows named
    assert "094e414-cf9c065b-h4279b3d" in err
    assert "094e414-dirty-cf9c065b-h9c5902b" in err
    # Both resolved step groups exist
    assert any("training-steps=25000" in l for l in out.splitlines())
    line_50k = next(l for l in out.splitlines() if "training-steps=50000" in l)
    # the 50k group cannot gain a member from any corpus_axes row (they are
    # all config.steps=25000), so its numbers stay exact-pinnable
    assert "n=4 (6 runs)" in line_50k
    assert "86.3% [78.4-90.2]" in line_50k
    # Cross-corpus: grouped on training-steps alone, the 25k group spans the
    # historical corpus and the sweep's - exactly one warning, naming the
    # group and >= 2 distinct corpus ids (the per-corpus view is
    # compare_arms'; this is the summariser's loud flag, not a statistic)
    with _err() as err25:
        ledger.summarise("hey seeree", grid_keys=["training-steps"])
    warnings = [l for l in err25.getvalue().splitlines()
                if "corpus_id(s)" in l]
    m = re.match(r"  WARNING: group (\S+) (\S+) spans (\d+) corpus_id\(s\)",
                 warnings[0]) if warnings else None
    assert m and m.group(2) == "training-steps=25000" and int(m.group(3)) >= 2, \
        err25.getvalue()
    assert "silently averaged" in err25.getvalue()
    # a sweep arm renders by its grid label
    assert any("real-vtlp=ryan=30" in l for l in out.splitlines())


# ---------------------------------------------------------------------------
# C3 step 2: the matched-FA reading off the recorded threshold sweep
# ---------------------------------------------------------------------------

def test_matched_fa_at_most_budget_picks_the_right_point():
    # B = median across the two swept groups of each group's own FA at its
    # recorded threshold: median(2%, 3%) = 2.5%. At FA <= 2.5% the 25k
    # curve's eligible points are (2%, 85%) and (0%, 60%) -> 85%; the
    # (5%, 90%) point is EXCLUDED, not blended into the reading, and the
    # 50k group reads its own curve: only (1%, 72%) is eligible.
    c25 = _sweep([0.3, 0.5, 0.7], [0.05, 0.02, 0.0], [0.90, 0.85, 0.60])
    c50 = _sweep([0.3, 0.5, 0.7], [0.06, 0.03, 0.01], [0.88, 0.80, 0.72])
    with tmp_ledger([
        rec_sweep("aaa1111-cf9c065b-haaaaaa", 25000, 42, 0.85, 0.02, c25),
        rec_sweep("bbb2222-cf9c065b-hbbbbbb", 50000, 43, 0.80, 0.03, c50),
    ]):
        out = ledger.summarise("hey seeree")
    assert "matched-FA budget: adv FA <= 2.5%" in out
    line25 = next(l for l in out.splitlines() if "25000" in l)
    line50 = next(l for l in out.splitlines() if "50000" in l)
    assert "det@FA<=2.5%  85.0%" in line25
    assert "det@FA<=2.5%  72.0%" in line50
    # the derivation is printed, not implicit
    assert "median" in out and "never interpolated" in out
    # and the old @-threshold columns still sit beside it, labelled
    assert "adv FA@0.5" in line25 and "detection@0.5" in line25


def test_budget_below_best_fa_prints_marked_fallback():
    # B = median(2%, 5%) = 3.5%. The 50k curve never reaches 3.5% (best
    # point 5%): its value is that best point (70% at its own FA), marked
    # '*' with a footnote saying what it is - never an interpolated guess
    # at 3.5%.
    ca = _sweep([0.3, 0.5, 0.7], [0.05, 0.02, 0.0], [0.90, 0.85, 0.60])
    cb = _sweep([0.3, 0.5, 0.7], [0.09, 0.07, 0.05], [0.90, 0.80, 0.70])
    with tmp_ledger([
        rec_sweep("aaa1111-cf9c065b-haaaaaa", 25000, 42, 0.85, 0.02, ca),
        rec_sweep("bbb2222-cf9c065b-hbbbbbb", 50000, 43, 0.80, 0.05, cb),
    ]):
        out = ledger.summarise("hey seeree")
    assert "matched-FA budget: adv FA <= 3.5%" in out
    line50 = next(l for l in out.splitlines() if "50000" in l)
    assert "det@FA<=3.5%  70.0% *" in line50
    assert "no curve point at adv FA <= 3.5%" in out
    assert "not an interpolation" in out


def test_mixed_ledger_renders_both_readings_labelled():
    # One swept group beside one pre-sweep group: the pre-sweep row keeps
    # its @-threshold columns and gets an explicit '-' (never a borrowed
    # number), and the caveat names the two vintages so the table cannot
    # be read as one comparison.
    ca = _sweep([0.3, 0.5, 0.7], [0.05, 0.02, 0.0], [0.90, 0.85, 0.60])
    with tmp_ledger([
        rec_sweep("aaa1111-cf9c065b-haaaaaa", 25000, 42, 0.85, 0.02, ca),
        rec("bbb2222-cf9c065b-hbbbbbb", 50000, 50000, 43, 0.80, 0.03),
    ]):
        out = ledger.summarise("hey seeree")
    # with a single swept group, B is that group's own recorded-threshold FA
    line50 = next(l for l in out.splitlines() if "50000" in l)
    assert "adv FA@0.5" in line50 and "detection@0.5" in line50
    assert "-  (no sweep on file" in line50
    line25 = next(l for l in out.splitlines() if "25000" in l)
    assert "det@FA<=2.0%  85.0%" in line25
    assert "predate the sweep" in out


def test_real_ledger_swept_and_presweep_vintages_render():
    # The 2026-09-23 corpus_axes sweep made the real ledger hold BOTH
    # vintages - swept and pre-sweep rows side by side - so the old test's
    # "no sweep on file, byte-identical to the pre-sweep table" world is
    # gone. This test pins the STRUCTURAL facts instead: the matched-FA
    # budget header appears (swept groups exist), and the pre-sweep c6542f5
    # rows still render their @-threshold columns beside an explicit '-' cell
    # - a number no run produced must not be printed as one.
    # The byte-pin against golden files returns with re-captured goldens
    # (main agent, once the final retry row lands); this test deliberately
    # does not read tests/golden/*.
    if not ledger.ledger_path("hey seeree").is_file():
        print("  skip: no ledger at "
              f"{ledger.ledger_path('hey seeree')} (fresh checkout)")
        return
    with _err():
        out = ledger.summarise("hey seeree")
    # swept groups exist: the common budget header with its derivation
    assert "matched-FA budget: adv FA <=" in out
    # pre-sweep rows: their @-threshold columns are intact, and their
    # no-sweep group gets the explicit '-' cell, not a borrowed number
    assert "adv FA@0.5" in out
    assert "detection@0.5" in out
    assert "-  (no sweep on file" in out


def test_real_ledger_byte_pin_recaptured():
    # Byte-pin, re-captured DELIBERATELY 2026-09-23 after the real-vtlp sweep
    # (5 rows) + the seed-2042 retry row (1) grew the ledger past the 2026-09-22
    # pin - the append-only rule means growth is the norm and re-capture is an
    # act, not a drift (docstring of the test this replaces). Pinned state:
    # 14 records, three real-vtlp arms + the steps grid, all vintages rendered.
    # If a new record is ever appended, re-capture with the same deliberateness:
    #   python -m train.ledger --wake-word "hey seeree" > stdout.golden 2> stderr.golden
    if not ledger.ledger_path("hey seeree").is_file():
        print("  skip: no ledger at "
              f"{ledger.ledger_path('hey seeree')} (fresh checkout)")
        return
    golden = Path(__file__).parent / "golden"
    with _err() as err:
        out = ledger.summarise("hey seeree")
    assert out + "\n" == (golden / "ledger_hey_seeree_stdout.txt").read_text()
    assert err.getvalue() == (golden / "ledger_hey_seeree_stderr.txt").read_text()


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
