"""Guards for src/train/corpus/real.py's per-speaker copy weight
(--real-copies-override). The global copy weight is one lever aimed at every
speaker at once; this flag aims the same lever at one thin voice, so these
tests pin: the named speaker gets the override, every other speaker keeps the
base weight, and the shifted (VTLP) variants are real new acoustic forms, not
duplicates or silence.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from train.corpus.real import balanced_copy_weights, copy_real_samples  # noqa: E402

SR = 16000


def _make_tree(root, clips_per_speaker):
    for speaker, n in clips_per_speaker.items():
        d = Path(root) / speaker
        d.mkdir(parents=True)
        data = (20000 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)
                ).astype(np.int16)
        for i in range(n):
            scipy.io.wavfile.write(str(d / f"hey_seeree_{i:04d}.wav"), SR, data)


def test_per_speaker_override_changes_only_that_speakers_copies():
    # ryan=3 with base 1: ryan gets 3 copies per clip, jay stays at 1.
    # Destinations are real_{i}_{speaker}_... so the counts are checkable.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=1, per_speaker_copies={"ryan": 3})
        ryan = sum(1 for p in out.glob("*.wav") if "_ryan_" in p.name)
        jay = sum(1 for p in out.glob("*.wav") if "_jay_" in p.name)
        assert ryan == 2 * 3, f"ryan clips should carry 3 copies each, got {ryan}"
        assert jay == 1 * 1, f"jay keeps the base weight, got {jay}"
        assert count == ryan + jay


def test_override_none_is_the_unchanged_default_path():
    # per_speaker_copies=None must be byte-equivalent behaviour to the old
    # signature: every speaker at the base weight.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=2)
        ryan = sum(1 for p in out.glob("*.wav") if "_ryan_" in p.name)
        jay = sum(1 for p in out.glob("*.wav") if "_jay_" in p.name)
        assert (ryan, jay) == (4, 2)
        assert count == 6


def test_vtlp_copies_add_variants_not_raw_duplicates():
    # per_speaker_vtlp={"ryan": 3}: ryan gets 3 real_v*_<ratio>_ files per clip
    # (names carry the ratio, so duplicates are impossible by construction) and
    # jay is untouched. Content must actually differ from the source - a
    # "shifted" copy equal to the raw clip means the shift was inert.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=1, per_speaker_vtlp={"ryan": 3})
        v = sorted(p for p in out.glob("*.wav") if p.name.startswith("real_v"))
        assert len(v) == 2 * 3, f"expected 6 shifted ryan clips, got {len(v)}"
        assert all("_ryan_" in p.name for p in v)
        # ratios land in the child-lever range and differ across copies
        ratios = {float(p.name.split("_")[2]) for p in v}
        assert all(1.15 <= r <= 1.30 for r in ratios), ratios
        raw = (out / "real_0_ryan_hey_seeree_0000.wav").read_bytes()
        assert any(p.read_bytes() != raw for p in v), \
            "shifted copies are byte-identical to the raw clip - inert"
        # NOT near-silence: the dtype bug (float into the int16 contract) produces
        # files that differ from the raw clip and are digitally silent.
        for p in v:
            sr_x, x = scipy.io.wavfile.read(p)
            assert int(np.abs(x.astype(np.int64)).max()) > 1000, \
                f"{p.name} is near-silent - shift dtype contract broken"
        assert count == 2 + 6 + 1  # ryan raw + ryan shifted + jay raw


# ---------------------------------------------------------------------------
# --balance-real-copies. The rule under test is the one the measurements asked
# for: a FLAT weight leaves a thin voice the thinnest in the positive set, and
# the least-recorded speaker can read worse on her own training clips than the
# better-recorded ones do - a corpus-coverage failure. Deriving the weight
# from the counts means recording more of a thin speaker shrinks their lift
# instead of requiring a flag edit.


def _counts(**kw):
    return dict(kw)


def test_balance_equalises_rows_up_and_never_down():
    from train.corpus.real import balanced_copy_weights
    # the richest speaker is 8 clips x 10 = 80 rows, so it is the target.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5, ryan=3), 10)
    assert w["jay"] == 10, "the richest speaker keeps the base weight"
    assert w["jen"] == 16, f"ceil(80/5)=16, got {w['jen']}"
    assert w["ryan"] == 27, f"ceil(80/3)=27, got {w['ryan']}"
    rows = {k: w[k] * v for k, v in (("jay", 8), ("jen", 5), ("ryan", 3))}
    assert rows["jen"] >= 80 and rows["ryan"] >= 80, "everyone reaches the target"
    assert rows["jay"] == 80
    assert any("jay" in n and "rows" in n for n in notes), notes
    # Never below base: a speaker richer than the target still gets base.
    w2, _ = balanced_copy_weights(_counts(jay=8, jen=20), 10, speakers=["jay"])
    assert w2["jay"] == 10 and w2["jen"] == 10, "no speaker is cut to tidy the table"


def test_balance_only_touches_the_named_speakers():
    from train.corpus.real import balanced_copy_weights
    # Only the named speakers are balanced; the child voice must keep its own
    # weight - more recordings, not more copies, is its lever.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5, ryan=3), 10,
                                     speakers=["jay", "jen"])
    assert w == {"jay": 10, "jen": 16, "ryan": 10}, w
    assert any("ryan" in n and "not in the balance set" in n for n in notes), notes
    try:
        balanced_copy_weights(_counts(jay=8), 10, speakers=["jey"])
    except ValueError:
        pass
    else:
        raise AssertionError("a typo'd speaker name must fail loud, not train at base")


def test_balance_explicit_override_wins_over_the_derived_number():
    from train.corpus.real import balanced_copy_weights
    # A hand-set weight is a decision, a derived one is arithmetic.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5), 10,
                                     explicit={"jen": 40})
    assert w["jen"] == 40 and w["jay"] == 10
    assert any("explicit override" in n for n in notes), notes


def test_balance_cap_is_reported_not_silent():
    from train.corpus.real import balanced_copy_weights
    # the thin speaker at 5 clips needs 16x to reach 80 rows; a 1.5 cap (base 10, cap 15) binds.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5), 10, max_multiplier=1.5)
    assert w["jen"] == 15, w
    assert any("capped" in n for n in notes), notes


def test_balance_cap_below_the_base_weight_says_it_cannot_bind():
    from train.corpus.real import balanced_copy_weights
    # A 0.5 cap on a base of 10 asks for 5 copies - below the base weight, where
    # equalising never goes. The weight must stay at the base (not drop to the cap),
    # and the note must say the cap cannot bind: claiming "capped at 0.5x" next to a
    # weight of 10x is a message that contradicts the number beside it.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5), 10, max_multiplier=0.5)
    assert w["jen"] == 10, w
    assert w["jay"] == 10, w
    assert any("cannot bind" in n for n in notes), notes
    assert not any("capped at" in n for n in notes), \
        f"a cap that did not bind must not be reported as one: {notes}"


def test_balance_spec_parses_off_all_and_a_list():
    from train.corpus.real import parse_balance_spec
    assert parse_balance_spec("") is None
    assert parse_balance_spec("   ") is None
    assert parse_balance_spec("all") == "all"
    assert parse_balance_spec("ALL") == "all"
    assert parse_balance_spec("jay, jen") == ["jay", "jen"]
    try:
        parse_balance_spec(",,")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty speaker list must fail loud")


def test_speaker_clip_counts_matches_what_the_copier_reads():
    from train.corpus.real import speaker_clip_counts
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "samples"
        _make_tree(src, {"jay": 3, "jen": 2})
        # a loose file at the root is counted, under the same key copy_real_samples uses
        data = (20000 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)).astype(np.int16)
        scipy.io.wavfile.write(str(src / "loose.wav"), SR, data)
        assert speaker_clip_counts(src) == {"jay": 3, "jen": 2, "(loose files)": 1}
        assert speaker_clip_counts(Path(tmp) / "nope") == {}


def test_balance_flows_through_to_the_written_copies():
    from train.corpus.real import balanced_copy_weights, speaker_clip_counts
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"jay": 4, "jen": 2})
        w, _ = balanced_copy_weights(speaker_clip_counts(src), 5)
        copy_real_samples(src, out, copies=5, per_speaker_copies=w)
        names = [p.name for p in out.iterdir()]
        jay = len([n for n in names if "_jay_" in n])
        jen = len([n for n in names if "_jen_" in n])
        assert (jay, jen) == (20, 20), f"rows should be equal: jay={jay} jen={jen}"


def test_balance_accepts_the_all_sentinel_as_every_speaker():
    from train.corpus.real import balanced_copy_weights, parse_balance_spec
    # The trainers translate "all" to None before calling, so passing the sentinel
    # straight through was never exercised - until a dry run did it and the
    # `for s in speakers` loop iterated the STRING. The error named speakers
    # ['a', 'l', 'l'], which reads like a typo in the flag rather than a type
    # confusion, so the conversion now lives in the function too.
    counts = _counts(jay=8, jen=5, ryan=3)
    sentinel = parse_balance_spec("all")
    assert sentinel == "all", "parse returns the sentinel, not a list"
    w, _ = balanced_copy_weights(counts, 10, speakers=sentinel)
    assert w == balanced_copy_weights(counts, 10, speakers=None)[0], \
        f"'all' must mean every speaker: {w}"
    assert w == {"jay": 10, "jen": 16, "ryan": 27}, w


def test_a_speaker_carrying_an_override_is_not_the_balance_anchor():
    """The anchor is drawn from the SPEAKERS STILL BEING BALANCED, not the loudest name.

    The richest speaker carrying an explicit override is the one being lifted deliberately,
    so anchoring on it named a speaker nobody was balanced to and a row count nobody
    reaches - next to a table in the same output that said otherwise. jen (5 clips) is now
    the anchor at 5*10=50 rows, so ryan (3) lifts to ceil(50/3)=17.
    """
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5, ryan=3), 10,
                                     explicit={"jay": 100})
    assert w["jay"] == 100, w
    assert w["jen"] == 10, w
    assert w["ryan"] == 17, f"ceil(50/3)=17, got {w['ryan']}"
    assert any("balanced against jen at 50 rows" in n for n in notes), notes
    assert not any("balanced against jay" in n for n in notes), notes


def test_every_speaker_overridden_says_there_is_nothing_to_equalise():
    # The all-explicit branch has no balanced speakers left to anchor on; it must say so
    # rather than name the richest speaker, whose weight is itself an override.
    w, notes = balanced_copy_weights(_counts(jay=8, jen=5), 10,
                                     explicit={"jay": 3, "jen": 40})
    assert w == {"jay": 3, "jen": 40}, w
    assert any("nothing to equalise" in n for n in notes), notes
    assert not any("balanced against" in n for n in notes), notes


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
