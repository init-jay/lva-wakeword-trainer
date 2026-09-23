"""Guards for tools/compare_arms.py, the cross-arm ledger reader.

The tool's whole contract is honest reading of the sweep arms: group on the
grid VALUE (the arm), count distinct (config-hash, seed) pairs (bug.md C2),
never pool per speaker or per negative category (CLAUDE.md), read
matched-FA as a point pick on each arm's OWN curve at the ledger's common
budget (bug.md C3 step 2), and label the mixed vintages. Warnings (divergent
duplicates, multi-corpus arms) go to stderr like train/ledger.py's.

Plain-python convention (tests/_runner.py): no pytest, no fixture files in
the repo - the synthetic ledger is written to a tempfile and handed to the
real CLI through its --ledger flag, so the whole main() -> render() path is
exercised and captured stdout/stderr is what gets asserted on.

Fixture rows are shaped EXACTLY like output/hey_seeree/runs.jsonl lines
(field names copied from a real row and from tests/test_ledger.py's
helpers); one test pins that against the real ledger when it is on file.
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import compare_arms  # noqa: E402

REAL_LEDGER = REPO_ROOT / "output" / "hey_seeree" / "runs.jsonl"


# --- fixtures: exact shapes from a real runs.jsonl line ---------------------

def _curve(thresholds, adv_rate, pos_rate, adv_n=298, pos_n=51):
    """An eval threshold_sweep block in the shape eval_model.py writes it
    (same keys as tests/test_ledger.py's _sweep; the FA axis is the
    extend+hey_other subset, never pooled with the other categories)."""
    return {
        "thresholds": list(thresholds),
        "adv_rate": list(adv_rate),
        "positives_rate": list(pos_rate),
        "adv_n": adv_n,
        "pos_n": pos_n,
    }


def rec(tag, seed, value, det, fa, extend=(9, 148), hey_other=(2, 150),
        ryan=(3, 6), jay=(33, 35), jen=(10, 10), threshold=0.5,
        curve=None, vhs=None, corpus_id="f9c065b"):
    """One ledger record, field-for-field in the shape of a real row:
    the same top-level keys train/ledger.py record() files, and the same
    eval_block keys eval_model.py --json writes (model/threshold/backend/
    timestamp/false_accepts_by_category/adversarial/positives/per_speaker/
    command_following/gates, plus threshold_sweep and voice_holdout_set when
    carried). positives.missed is a list of FILENAMES, as on file."""
    pos_n = ryan[1] + jay[1] + jen[1]
    pos_det = ryan[0] + jay[0] + jen[0]
    adv_n = extend[1] + hey_other[1]
    adv_fired = extend[0] + hey_other[0]
    eval_block = {
        "model": f"output/hey_seeree/oww/hey_seeree_{tag}.onnx",
        "threshold": threshold,
        "backend": "openWakeWord via openwakeword.model.Model (onnx)",
        "timestamp": "2026-09-23T00:00:00+00:00",
        "false_accepts_by_category": {
            "command": {"n": 12, "fired": 0, "rate": 0.0,
                        "median_peak": 0.0009, "worst_peak": 0.0009},
            "extend": {"n": extend[1], "fired": extend[0],
                       "rate": extend[0] / extend[1],
                       "median_peak": 0.0009, "worst_peak": 0.9841},
            "general": {"n": 36, "fired": 0, "rate": 0.0,
                        "median_peak": 0.0009, "worst_peak": 0.0024},
            "hey_other": {"n": hey_other[1], "fired": hey_other[0],
                          "rate": hey_other[0] / hey_other[1],
                          "median_peak": 0.0009, "worst_peak": 0.9658},
            "other_ww": {"n": 8, "fired": 0, "rate": 0.0,
                         "median_peak": 0.0009, "worst_peak": 0.0010},
            "running": {"n": 12, "fired": 0, "rate": 0.0,
                        "median_peak": 0.0009, "worst_peak": 0.0576},
        },
        "adversarial": {"n": adv_n, "fired": adv_fired, "rate": fa},
        "positives": {
            "n": pos_n,
            "detected": pos_det,
            "rate": det,
            "latency_median_ms": 27.03125,
            "latency_p90_ms": 155.0625,
            "missed": [f"hey_seeree_{i:04d}.wav" for i in range(pos_n - pos_det)],
        },
        "per_speaker": {
            "jay": {"n": jay[1], "detected": jay[0],
                    "rate": jay[0] / jay[1], "ci95": [0.8139, 0.9842],
                    "median_latency_ms": 51.4375},
            "jen": {"n": jen[1], "detected": jen[0],
                    "rate": jen[0] / jen[1], "ci95": [0.7225, 1.0],
                    "median_latency_ms": -108.125},
            "ryan": {"n": ryan[1], "detected": ryan[0],
                     "rate": ryan[0] / ryan[1], "ci95": [0.1876, 0.8124],
                     "median_latency_ms": 20.0625},
        },
        "command_following": {"n": pos_n, "immediately_after": pos_det,
                              "pause_then_command": pos_det, "pause_ms": 300},
        "gates": [
            {"check": f"extend+hey_other false accepts  {adv_fired}/{adv_n}",
             "gate": "< 6%", "pass": fa < 0.06},
            {"check": f"clean positive detection        {pos_det}/{pos_n}",
             "gate": ">= 98%", "pass": det >= 0.98},
            {"check": f"weakest speaker (ryan)          {ryan[0]}/{ryan[1]}",
             "gate": ">= 98%", "pass": ryan[0] / ryan[1] >= 0.98},
        ],
    }
    if curve is not None:
        eval_block["threshold_sweep"] = curve
    if vhs is not None:
        eval_block["voice_holdout_set"] = {
            "directory": "data/corpus/eval/voice_holdout_tts",
            "n": 45,
            "detected": int(round(vhs * 45)),
            "rate": vhs,
            "latency_median_ms": 12.0,
            "missed": [],
        }
    return {
        "wake_word": "hey seeree",
        "target": "oww",
        "tag": tag,
        "corpus_id": corpus_id,
        "seed": seed,
        "config": {"steps": 25000, "lr": 0.003,
                   "target_phrase": "hey seeree", "seed": seed},
        "wall_time": {"training (oww)": 938.6, "eval": 70.9},
        "eval_block": eval_block,
        "grid": {"real-vtlp": value},
        "recorded_utc": "2026-09-23T00:00:00+00:00",
    }


# --- runner: the real CLI against a tempfile ledger -------------------------

def _run(records, grid_key="real-vtlp"):
    """Write the synthetic ledger and drive compare_arms.main() through its
    --ledger flag; return (stdout, stderr) captured."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            compare_arms.main(["--wake-word", "hey seeree",
                               "--grid-key", grid_key,
                               "--ledger", str(path)])
    return out.getvalue(), err.getvalue()


# --- tests ------------------------------------------------------------------

def test_fixture_shape_matches_real_ledger():
    """The fixture's field names are the real file's field names, not
    inventions: every fixture key must exist on a real recorded row."""
    if not REAL_LEDGER.is_file():
        return  # fresh checkout: nothing real to pin against
    real = [json.loads(l) for l in REAL_LEDGER.read_text().splitlines()
            if l.strip()]
    assert real
    fx = rec("aaa1111-cf9c065b-h1234567", 42, "ryan=30", 0.90, 0.037,
             curve=_curve([0.5], [0.03], [0.9]), vhs=0.87)["eval_block"]
    row = rec("aaa1111-cf9c065b-h1234567", 42, "ryan=30", 0.90, 0.037)
    missing = set(row) - set(real[0])
    assert not missing, f"fixture top-level keys not in the real ledger: {missing}"
    # threshold_sweep and voice_holdout_set are the NEW-format eval keys:
    # every row on file predates them, so pin their NAMES against what
    # eval_model.py itself writes, and the rest of the eval block against
    # the real rows.
    if "eval" not in sys.modules:
        import types as _types
        _pkg = _types.ModuleType("eval")
        _pkg.__path__ = [str(REPO_ROOT / "eval" / "src")]
        sys.modules["eval"] = _pkg
    from eval.eval_model import threshold_sweep  # noqa: E402
    sweep_keys = set(threshold_sweep([0.5, 0.3], [0.4], grid=[0.5]).keys())
    assert set(fx["threshold_sweep"]) == sweep_keys, \
        f"threshold_sweep keys drifted from eval_model.py: {sweep_keys}"
    assert set(fx["voice_holdout_set"]) == {"directory", "n", "detected",
                                            "rate", "latency_median_ms",
                                            "missed"}
    eval_missing = (set(fx) - {"threshold_sweep", "voice_holdout_set"}
                    - set(real[0]["eval_block"]))
    assert not eval_missing, \
        f"fixture eval_block keys not in the real ledger: {eval_missing}"
    for speaker in ("jay", "ryan"):
        sp_missing = (set(row["eval_block"]["per_speaker"][speaker])
                      - set(real[0]["eval_block"]["per_speaker"][speaker]))
        assert not sp_missing, f"per_speaker[{speaker}] keys invented: {sp_missing}"
    for cat in ("extend", "hey_other", "command"):
        cat_missing = (set(row["eval_block"]["false_accepts_by_category"][cat])
                       - set(real[0]["eval_block"]["false_accepts_by_category"][cat]))
        assert not cat_missing, \
            f"false_accepts_by_category[{cat}] keys invented: {cat_missing}"


def test_arm_grouping_and_duplicate_collapse():
    """Group on the grid VALUE; the '' arm reads (none); exact
    (config-hash, seed) duplicates collapse to n=1 (2 runs); rows whose
    grid dict lacks the key are out of scope."""
    out, err = _run([
        # arm '' - two records of the SAME (config-hash, seed) pair, identical
        # evals (the C2 duplicate: same run at two commits).
        rec("aaa1111-cf9c065b-h1111111", 42, "", 0.90, 0.03),
        rec("bbb2222-cf9c065b-h1111111", 42, "", 0.90, 0.03),
        # arm ryan=30 - two DISTINCT pairs (two seeds), no collapse.
        rec("ccc3333-cf9c065b-h2222222", 42, "ryan=30", 0.88, 0.03),
        rec("ddd4444-cf9c065b-h2222222", 43, "ryan=30", 0.92, 0.04),
        # a row from the old training-steps sweep: no real-vtlp key.
        {**rec("eee5555-cf9c065b-h3333333", 42, "", 0.5, 0.5),
         "grid": {"training-steps": 25000}},
    ])
    assert "grid key 'real-vtlp' in 4" in out
    assert "ARM real-vtlp=(none)   corpus f9c065b   n=1 (2 runs)" in out
    assert "ARM real-vtlp=ryan=30   corpus f9c065b   n=2" in out
    assert "eee5555" not in out  # the non-arm row is out of scope
    assert err == ""


def test_divergent_duplicate_warns_naming_both_tags():
    """Same (config-hash, seed) pair whose evals DIFFER must not collapse
    silently: a DETERMINISM WARNING on stderr names both tags."""
    out, err = _run([
        rec("aaa1111-cf9c065b-h1111111", 42, "ryan=30", 0.90, 0.03),
        rec("bbb2222-cf9c065b-h1111111", 42, "ryan=30", 0.67, 0.05),
    ])
    assert "DETERMINISM WARNING" in err
    assert "aaa1111-cf9c065b-h1111111" in err
    assert "bbb2222-cf9c065b-h1111111" in err
    # Both values stay in the row: the pair does not collapse to one sample.
    assert "n=1 (2 runs)" in out
    assert "67.0-90.0" in out  # min-max carries both readings


def test_multi_corpus_arm_is_loudly_warned():
    """An arm whose rows carry more than one corpus_id must warn - cross-
    corpus rows must never be silently averaged."""
    out, err = _run([
        rec("aaa1111-1111111-h1111111", 42, "ryan=30", 0.90, 0.03,
            corpus_id="f9c065b"),
        rec("bbb2222-2222222-h2222222", 43, "ryan=30", 0.88, 0.04,
            corpus_id="ab34d90"),
    ])
    assert "WARNING" in err
    assert "ryan=30" in err and "f9c065b" in err and "ab34d90" in err
    assert "silently averaged" in err
    # The arm line itself shows both corpora (sorted), so the table says
    # what it mixed.
    assert "corpus ab34d90 + f9c065b" in out


def test_matched_fa_budget_and_unreachable_floor():
    """B is the ledger's rule (median across swept arms of each arm's
    median recorded-threshold FA, derivation printed); each arm reads its
    OWN curve at B as a point pick; an arm whose curve cannot reach B is
    marked '*' with its best-reachable point as a floor."""
    out, err = _run([
        # arm ryan=30: curve reaches FA <= 3.5 -> picks the 45.0% point at
        # FA 2.0 (not the 40.0% point at FA 1.0: max detection wins).
        rec("aaa1111-cf9c065b-h1111111", 42, "ryan=30", 0.90, 0.03,
            curve=_curve([0.5, 0.6, 0.7], [0.05, 0.02, 0.01],
                         [0.50, 0.45, 0.40])),
        # arm ryan=60: curve's best FA is 5.0 > 3.5 -> unreachable, floor.
        rec("bbb2222-cf9c065b-h2222222", 43, "ryan=60", 0.95, 0.04,
            curve=_curve([0.5, 0.6, 0.7], [0.05, 0.06, 0.07],
                         [0.60, 0.50, 0.40])),
    ])
    assert err == ""
    # The derivation: medians 3.0 and 4.0 -> B = 3.5, both arms swept.
    assert "matched-FA budget: adv FA <= 3.5% - the median, across the" in out
    assert "2 swept arm(s), of each arm's own median adv FA at its recorded" in out
    # Arm ryan=30: the 45.0% point is picked (single sample -> bare value).
    block30 = out.split("ARM real-vtlp=ryan=30")[1].split("ARM ")[0]
    assert "det@FA<=3.5%   45.0%" in block30
    assert "det@FA<=3.5%   40.0%" not in block30
    # Arm ryan=60: unreachable -> the 60.0% floor at its own FA, marked '*'.
    block60 = out.split("ARM real-vtlp=ryan=60")[1]
    assert "det@FA<=3.5%   60.0% *" in block60
    assert ("no curve point at adv FA <= 3.5% (its best point is FA 5.0%)"
            in block60)
    assert "a floor, not a reading at the budget" in block60


def test_sweepless_rows_keep_labelled_fallback():
    """Rows without threshold_sweep (every pre-sweep row) keep the
    labelled @-threshold reading and get '- (no sweep on file)' in the
    matched cell - train/ledger.py's mixed-vintage behavior. When NO arm
    has a sweep at all, there is no budget to derive."""
    out, _ = _run([
        rec("aaa1111-cf9c065b-h1111111", 42, "", 0.90, 0.037),
        rec("bbb2222-cf9c065b-h2222222", 43, "ryan=30", 0.88, 0.033),
    ])
    assert "detection@0.5" in out and "adv FA@0.5" in out
    assert "det@FA<=B        -  (no sweep on file; @-threshold reading only)" in out
    # No budget is invented from nothing: the derivation header is absent.
    assert "matched-FA budget" not in out
    # Mixed-vintage: one swept arm, one not - the swept arm is read at B,
    # the other keeps the labelled fallback.
    out2, _ = _run([
        rec("aaa1111-cf9c065b-h1111111", 42, "ryan=30", 0.90, 0.03,
            curve=_curve([0.5, 0.6], [0.05, 0.02], [0.50, 0.45])),
        rec("bbb2222-cf9c065b-h2222222", 43, "ryan=60", 0.88, 0.037),
    ])
    assert "matched-FA budget: adv FA <= 3.0%" in out2  # only ryan=30 swept
    assert "det@FA<=3.0%   45.0%" in out2
    assert "det@FA<=3.0%   -  (no sweep on file; @-threshold reading only)" in out2


def test_per_speaker_wilson_and_voice_holdout_and_footer():
    """Per speaker (never pooled): n/m with the eval harness's Wilson CI
    (ryan 3/6 -> [18.8-81.2], the value eval_model.py wilson_interval gives);
    voice_holdout_set is labelled a ranking signal, not a gate; the footer
    (redraw noise, one repeat is not a result, 10-point floor) is ALWAYS
    printed, even for a single record."""
    out, _ = _run([
        rec("aaa1111-cf9c065b-h1111111", 42, "ryan=30", 0.90, 0.03,
            vhs=0.8667),
    ])
    assert "per speaker (never pooled across speakers):" in out
    assert "3/6  50.0%  [18.8-81.2]" in out      # ryan, the eval's Wilson CI
    assert "33/35  94.3%" in out                 # jay
    # Adversarial FA per category: extend and hey_other separately, and the
    # other categories listed (context), never a pooled adversarial line.
    assert "9/148  6.1%" in out                  # extend
    assert "2/150  1.3%" in out                  # hey_other
    assert "0/12  0.0%  (context)" in out        # command, context only
    assert "(ranking signal, not a gate)" in out
    assert "voice_holdout  86.7%" in out
    # The footer, always: redraw noise, one repeat, the 10-point floor.
    assert "ARMS DIFFER IN CORPUS IDENTITY" in out
    assert "one repeat is not a result" in out
    assert "10-point run-to-run noise floor" in out
    assert "n<=35" in out


def test_empty_ledger_and_absent_key():
    """An empty ledger and a ledger with no rows for the key both render a
    labelled nothing, not a traceback - the tool is read-only on history."""
    out, _ = _run([], grid_key="real-vtlp")
    assert "no records carry that grid key" in out
    out2, _ = _run([{**rec("aaa1111-cf9c065b-h1111111", 42, "", 0.9, 0.03),
                     "grid": {"training-steps": 25000}}],
                   grid_key="training-steps")
    assert "grid key 'training-steps' in 1" in out2
    assert "ARM training-steps=25000" in out2


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
