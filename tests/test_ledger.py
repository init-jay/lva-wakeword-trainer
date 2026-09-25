"""Guards for src/train/ledger.py's summarise: the runs were sound, the tool
that summarised them was not.

C1: group on the RESOLVED config, not the grid label. An early sweep ran
before the grid was threaded into the command, so some rows carried 25k
labels around runs that were actually filed at 50k (config.steps), and
grouping on the label printed a misleading "25k" mean that was really a
25k/50k mix - contradicting the hand-built verdict.

C2: n counts (config-hash, seed) PAIRS, not records. Two records sharing the
tag's h-half (the config hash, src/train/provenance.py) and the seed are the
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
byte-identical to the pre-sweep table - pinned in this file as a byte-exact
constant over a synthetic ledger: the real one is append-only and grows,
so a pin that followed it would need re-capturing on every appended run.

No pytest (tests/_runner.py): this file was written with fixtures and
capsys and the `make test` target - plain python, one file at a time, no
pytest in any venv in this repo - died on its import and silently skipped
the files after it in the suite. Converted to the house convention:
tempfile for the scratch ledger, redirect_stderr for the
warning capture, and a byte-exact constant for the render pin. Every test
runs against a synthetic ledger - the real one is history, not a fixture,
and a test that skipped on a fresh checkout would teach nothing there.
"""

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

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
        "corpus_id": "base",
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
        rec("aaa1111-base-haaaaaa", 25000, 25000, 42, 0.70, 0.02),
        rec("bbb2222-base-hbbbbbb", 25000, 50000, 43, 0.90, 0.04),
    ]), _err() as err:
        out = ledger.summarise("test word")
    err = err.getvalue()
    assert "training-steps=25000" in out
    assert "training-steps=50000" in out
    assert out.count("n=1 ") == 2, "one record per group: %r" % out
    assert "bbb2222-base-hbbbbbb" in err
    assert "25000" in err and "50000" in err
    # the agreeing record does not warn
    assert "aaa1111" not in err


def test_record_without_config_falls_back_to_label():
    # No resolved config on file (predates the config half): the label is
    # the only evidence, grouping keeps it, and there is nothing to warn
    # against.
    r = rec("aaa1111-base-haaaaaa", 25000, 25000, 42, 0.70, 0.02)
    r["config"] = None
    with tmp_ledger([r]), _err() as err:
        out = ledger.summarise("test word")
    assert "training-steps=25000" in out
    assert err.getvalue() == ""


def test_exact_duplicates_collapse_to_distinct_pairs():
    # C2: four records, two (config-hash, seed) pairs, each at two commits,
    # agreeing to the last digit. n is 2, with the run count alongside, and
    # the statistics come from the two distinct pairs.
    with tmp_ledger([
        rec("aaa1111-dirty-base-hconfa", 50000, 50000, 1042,
            0.9019607843137255, 0.03691275167785235),
        rec("ccc3333-dirty-base-hconfa", 50000, 50000, 1042,
            0.9019607843137255, 0.03691275167785235),
        rec("ddd4444-base-hconfb", 50000, 50000, 1043,
            0.7843137254901961, 0.03691275167785235),
        rec("ccc3333-dirty-base-hconfb", 50000, 50000, 1043,
            0.7843137254901961, 0.03691275167785235),
    ]), _err() as err:
        out = ledger.summarise("test word")
    assert "n=2 (4 runs)" in out
    assert "84.3% [78.4-90.2]" in out
    assert "adv FA@0.5" in out and "detection@0.5" in out
    assert err.getvalue() == ""


def test_divergent_duplicates_are_loud():
    # C2's other half: the same (config-hash, seed) with DIFFERENT eval
    # numbers is a determinism regression - name both tags and the
    # differing fields; do not collapse the disagreement into one number.
    with tmp_ledger([
        rec("aaa1111-base-hconfa", 50000, 50000, 1042, 0.9, 0.03),
        rec("ccc3333-base-hconfa", 50000, 50000, 1042, 0.67, 0.03),
    ]), _err() as err:
        out = ledger.summarise("test word")
    err = err.getvalue()
    assert "DETERMINISM REGRESSION" in err
    assert "aaa1111-base-hconfa" in err
    assert "ccc3333-base-hconfa" in err
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
        rec("aaa1111-base-haaaaaa", 25000, 25000, 42, 0.7, 0.02,
            threshold=0.5),
        rec("bbb2222-base-hbbbbbb", 25000, 25000, 43, 0.8, 0.01,
            threshold=0.4),
    ]):
        out = ledger.summarise("test word")
    assert "adv FA@mixed" in out
    assert "detection@mixed" in out


def test_summary_carries_the_matched_fa_caveat():
    # C3 step 1: the standing caveat under the table - the numbers cannot
    # be read as a matched-FA comparison (CLAUDE.md).
    with tmp_ledger([
        rec("aaa1111-base-haaaaaa", 25000, 25000, 42, 0.7, 0.02),
        rec("bbb2222-base-hbbbbbb", 50000, 50000, 43, 0.8, 0.01),
    ]):
        out = ledger.summarise("test word")
    assert "one-threshold readings, not a matched-FA comparison" in out
    assert "Never compare models" in out


def test_label_config_drift_groups_on_resolved_steps():
    # C1's self-correction, forced by construction instead of by history:
    # one row's grid label disagrees with its resolved config.steps and must
    # group under the config value, with a stderr warning that names the
    # offending tag (the 25k/50k mix that made the table contradict the
    # hand-built verdict); both resolved step groups render; a group whose
    # rows were trained on different frozen corpora warns exactly once,
    # naming its corpus ids; and a sweep arm that varies a second grid key
    # gets its own row labelled by its grid values.
    mislabeled = rec("aaa1111-base-h1111111", 25000, 50000, 11, 0.90, 0.04)
    other_corpus = rec("ccc3333-base-h3333333", 25000, 25000, 13, 0.75, 0.02)
    other_corpus["corpus_id"] = "a1b2c3d"
    sweep_arm = rec("eee5555-base-h5555555", 25000, 25000, 15, 0.80, 0.02)
    sweep_arm["grid"]["copies"] = 3
    with tmp_ledger([
        mislabeled,
        rec("bbb2222-base-h2222222", 25000, 25000, 12, 0.70, 0.02),
        other_corpus,
        rec("ddd4444-base-h4444444", 50000, 50000, 14, 0.85, 0.03),
        sweep_arm,
    ]), _err() as err:
        out = ledger.summarise("test word")
        # same synthetic ledger, grouped on the single key the cross-corpus
        # warning names: the warning must survive the narrower grouping
        with _err() as err25:
            ledger.summarise("test word", grid_keys=["training-steps"])
    err = err.getvalue()
    text25 = err25.getvalue()
    # (a) the disagreeing row groups under its config value, named on stderr
    assert "aaa1111-base-h1111111" in err
    assert "but the run's resolved config" in err
    assert "training-steps=25000" in err and "training-steps=50000" in err
    # (b) both resolved step groups render, and the mislabeled row landed in
    # the 50k group (each group reads n=2: had it stayed under its label, the
    # groups would read n=3 against n=1)
    line_50k = next(l for l in out.splitlines() if "training-steps=50000" in l)
    line_25k = next(l for l in out.splitlines()
                    if "copies=-" in l and "training-steps=25000" in l)
    assert "n=2" in line_50k
    assert "n=2" in line_25k
    # (d) the sweep arm is a row of its own, labelled by its grid values
    assert any("copies=3" in l and "training-steps=25000" in l
               for l in out.splitlines())
    # (c) exactly one cross-corpus warning, for the group that spans corpora
    # (here the plain 25k group); the arm group and the 50k group each sit
    # on one corpus and stay quiet. (The per-corpus view is compare_arms';
    # this is the summariser's loud flag, not a statistic.)
    warnings = [l for l in err.splitlines() if "corpus_id(s)" in l]
    assert len(warnings) == 1, f"expected one cross-corpus warning, got {warnings}"
    # Under the single-key grouping the same group must still warn exactly
    # once with >= 2 distinct corpus ids - pinned by the exact warning shape
    # so a rewording is a change, not a drift.
    warnings25 = [l for l in text25.splitlines()
                  if "corpus_id(s)" in l and "training-steps=25000" in l
                  and l.split("group ")[1].startswith("oww")]
    assert len(warnings25) == 1, f"expected one oww 25k cross-corpus warning, got {warnings25}"
    m = re.match(r"  WARNING: group (\S+) (\S+) spans (\d+) corpus_id\(s\)",
                 warnings25[0])
    assert m and m.group(2) == "training-steps=25000"
    assert int(m.group(3)) >= 2, warnings25[0]
    assert "silently averaged" in text25


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
        rec_sweep("aaa1111-base-haaaaaa", 25000, 42, 0.85, 0.02, c25),
        rec_sweep("bbb2222-base-hbbbbbb", 50000, 43, 0.80, 0.03, c50),
    ]):
        out = ledger.summarise("test word")
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
        rec_sweep("aaa1111-base-haaaaaa", 25000, 42, 0.85, 0.02, ca),
        rec_sweep("bbb2222-base-hbbbbbb", 50000, 43, 0.80, 0.05, cb),
    ]):
        out = ledger.summarise("test word")
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
        rec_sweep("aaa1111-base-haaaaaa", 25000, 42, 0.85, 0.02, ca),
        rec("bbb2222-base-hbbbbbb", 50000, 50000, 43, 0.80, 0.03),
    ]):
        out = ledger.summarise("test word")
    # with a single swept group, B is that group's own recorded-threshold FA
    line50 = next(l for l in out.splitlines() if "50000" in l)
    assert "adv FA@0.5" in line50 and "detection@0.5" in line50
    assert "-  (no sweep on file" in line50
    line25 = next(l for l in out.splitlines() if "25000" in l)
    assert "det@FA<=2.0%  85.0%" in line25
    assert "predate the sweep" in out


def test_swept_and_presweep_vintages_render_side_by_side():
    # A ledger holding BOTH vintages at once - rows with a threshold_sweep
    # and rows without - must stay readable as two readings: the common
    # matched-FA budget header appears (swept groups exist) and each swept
    # group reads its own curve AT that budget; the pre-sweep group keeps
    # its @-threshold columns and gets an explicit '-' cell - a number no
    # run produced must not be printed as one.
    c25 = _sweep([0.3, 0.5, 0.7], [0.05, 0.02, 0.0], [0.90, 0.85, 0.60])
    c50 = _sweep([0.3, 0.5, 0.7], [0.06, 0.03, 0.01], [0.88, 0.80, 0.72])
    with tmp_ledger([
        rec_sweep("aaa1111-base-h1111111", 25000, 42, 0.85, 0.02, c25),
        rec_sweep("bbb2222-base-h2222222", 50000, 43, 0.80, 0.03, c50),
        rec("ccc3333-base-h3333333", 75000, 75000, 44, 0.78, 0.03),
    ]):
        out = ledger.summarise("test word")
    # swept groups exist: the common budget header with its derivation
    assert "matched-FA budget: adv FA <=" in out
    # and each swept group reads its own curve at that budget (B is the
    # median of the two groups' recorded-threshold FAs = 2.5%)
    line25 = next(l for l in out.splitlines() if "training-steps=25000" in l)
    line50 = next(l for l in out.splitlines() if "training-steps=50000" in l)
    assert "det@FA<=2.5%  85.0%" in line25
    assert "det@FA<=2.5%  72.0%" in line50
    # pre-sweep group: its @-threshold columns are intact, and its
    # no-sweep cell is explicit, not a borrowed number
    line75 = next(l for l in out.splitlines() if "training-steps=75000" in l)
    assert "adv FA@0.5" in line75 and "detection@0.5" in line75
    assert "-  (no sweep on file" in line75


def _byte_pin_fixture():
    """The byte pin's ledger: one swept group beside one pre-sweep group
    whose two rows sit on different corpora - so one render exercises the
    budget header, both vintages' row shapes, every footnote block, and the
    cross-corpus warning on stderr."""
    c = _sweep([0.3, 0.5, 0.7], [0.05, 0.02, 0.0], [0.90, 0.85, 0.60])
    rows = [
        rec_sweep("aaa1111-base-h1111111", 25000, 42, 0.85, 0.02, c),
        rec("bbb2222-base-h2222222", 50000, 50000, 43, 0.80, 0.03),
    ]
    other_corpus = rec("ccc3333-base-h3333333", 50000, 50000, 44, 0.78, 0.03)
    other_corpus["corpus_id"] = "a1b2c3d"
    rows.append(other_corpus)
    with tmp_ledger(rows) as path, _err() as err:
        out = ledger.summarise("test word")
    return path, out, err.getvalue()


# Byte-pins for test_renderer_output_is_byte_pinned: the exact text
# summarise prints for _byte_pin_fixture(). The provenance citations and
# dates inside them are the renderer's own words (src/train/ledger.py), copied
# verbatim because that is what a byte pin pins. Re-capture DELIBERATELY -
# this fixture is fixed, it does not grow like the real ledger: run
# _byte_pin_fixture(), paste its stdout (minus the path-bearing first line)
# and stderr into the two constants here, and say in the commit that you did.
# A pin that silently drifts to keep the suite green has stopped being a pin.
_PINNED_STDOUT_BODY = """
  matched-FA budget: adv FA <= 2.0% - the median, across the
  1 swept group(s), of each group's own median adv FA at its recorded
  threshold. Every swept group is read AT that budget: the best detection on its
  own curve with FA <= B - a step-function point pick, never interpolated
  (CLAUDE.md: never compare models at a fixed threshold; bug.md C3, 2026-09-22).

  oww   training-steps=25000                   n=1  adv FA@0.5  2.0%  detection@0.5  85.0%  det@FA<=2.0%  85.0%
  oww   training-steps=50000                   n=2  adv FA@0.5  3.0% [3.0-3.0]  detection@0.5  79.0% [78.0-80.0]  det@FA<=2.0%  -  (no sweep on file; @-threshold reading only)

  [min-max] is the repeat-to-repeat spread. The noise floor in
  this repo is 10 points, measured at an identical config (77% and
  67% on the same holdout) - a difference inside that band is not
  a result, no matter which side of it the mean lands on.

  The @ value is the threshold the column was READ at. These are
  one-threshold readings, not a matched-FA comparison: a detection
  difference between rows is not a verdict - 'Never compare models
  at a fixed threshold' (CLAUDE.md); 77% vs 67% at 0.5 was the
  SAME config (bug.md C3, 2026-09-22).

  det@FA<=B: the matched-FA reading (bug.md C3 step 2, 2026-09-22 - the
  40-eval manual 0.25-0.85 job this exists to stop). One budget for every
  swept group; each reads its OWN curve, never another group's.  '-': the
  records predate the sweep - @-threshold columns only, not comparable at B.
  '*': the group's curve never reaches FA <= B; the marked value is its best
  reachable point, at its own FA - a floor, not a reading at the budget."""

_PINNED_STDERR = """  WARNING: group oww training-steps=50000 spans 2 corpus_id(s) a1b2c3d, base (2 record(s)) - cross-corpus rows must never be silently averaged: part of the [min-max] spread is TTS redraw, not seed noise (src/scripts/compare_arms.py is the per-corpus view)
"""


def test_renderer_output_is_byte_pinned():
    # The renderer's whole output over the synthetic fixture, byte for byte.
    # The only nondeterministic characters are the scratch file path the
    # header line names - pinned structurally (the rest of that line is
    # exact, including the record count); everything after the first newline,
    # and all of stderr, must match the constants above to the character.
    path, out, err = _byte_pin_fixture()
    head, sep, body = out.partition("\n")
    assert sep, out
    assert head == f"ledger: {path}  (3 record(s))"
    assert body == _PINNED_STDOUT_BODY
    assert err == _PINNED_STDERR


def test_corpus_id_is_part_of_the_duplicate_key():
    """The duplicate key is (config-hash, seed, corpus_id), pinned in BOTH directions.

    Direction one: a corpus_axes sweep holds the trainer fixed and redraws the TTS per
    arm, so two rows sharing a config hash and a seed but differing in corpus are two
    ARMS, not one run that disagreed with itself - no DETERMINISM REGRESSION, and the
    cross-corpus warning is the one that fires instead. Direction two: the same triple
    with divergent evals IS the regression the line exists for, so the key cannot be
    widened past the corpus either.

    Neither direction was pinned before. Every fixture that varied corpus_id varied the
    seed along with it - the byte pin's two corpora sit at seeds 43 and 44 - where the
    pair key and the triple key group identically, so reverting src/train/ledger.py to
    (config-hash, seed) rendered byte-for-byte the same output and the suite stayed green.
    """
    flat = rec("aaa1111-base-h1111111", 25000, 25000, 42, 0.85, 0.02)
    other_corpus = rec("bbb2222-base-h1111111", 25000, 25000, 42, 0.70, 0.03)
    other_corpus["corpus_id"] = "a1b2c3d"
    with tmp_ledger([flat, other_corpus]), _err() as err:
        ledger.summarise("test word")
    err = err.getvalue()
    assert "DETERMINISM REGRESSION" not in err, err
    assert "spans 2 corpus_id(s)" in err, err

    twin = rec("ccc3333-base-h1111111", 25000, 25000, 42, 0.60, 0.04)
    with tmp_ledger([flat, twin]), _err() as err2:
        ledger.summarise("test word")
    err2 = err2.getvalue()
    assert "DETERMINISM REGRESSION" in err2, err2
    assert "aaa1111-base-h1111111" in err2 and "ccc3333-base-h1111111" in err2, err2


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
