"""Guards for train/mww/features.py's identity-aware split.

WHY THESE TESTS EXIST. microWakeWord splits its feature corpus per FILE
(microwakeword/audio/clips.py, an HF train_test_split over the directory). A person's
real recordings enter the corpus as N copies of each - that is how openWakeWord weights
them, and the lever that measured jen 3/10 -> 8/10 on the mWW holdout on 2026-09-24
(sweeps/mww-real-copies-probe.yaml). Per-file scattering puts copy 3 of a recording in
training and copy 7 of it in validation, and mWW then SELECTS the weights it ships on
validation `average_viable_recall` - so the copies are not just a flattering number,
they bias which checkpoint ships. train/corpus/real.py's NOTE told the port not to
bring raw copies for exactly this reason; the split below is what makes the note
obsolete, and these tests are what make the split safe.

The contract:

* copies (real_<i>_...) and vocal-tract variants (real_v<i>_<ratio>_...) of one
  recording share a split - all three of them, in every corpus layout;
* anything that is not a copy keeps its own identity, so a corpus with no copies
  partitions exactly as it did before (TTS renders and ambient sets are untouched);
* 2 * split_count of the RECORDINGS are held out, one row each (broad and shallow):
  a row budget at 10x copies means a tenth of the recordings and is the wrong shape for
  a selection signal - that is what the first identity-aware run measured scattering;
  the partition is stable across runs and across input order, and goes through hashlib
  because `hash()` is salted per process;
* a corpus whose copy block is bigger than the holdout refuses instead of quietly
  producing an empty validation set (which would make selection a no-op).
"""

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported from train.mww.split, not features: features pulls in microwakeword and
# mmap_ninja at module scope, and the Makefile:53 suite runs under the oww venv.
from train.mww.split import group_partition, recording_identity  # noqa: E402


def _names(speaker="jen", base="hey_seeree_0012", copies=5, variants=3):
    raw = [f"real_{i}_{speaker}_{base}.wav" for i in range(copies)]
    vtlp = [f"real_v{i}_1.{20 + i}_{speaker}_{base}.wav" for i in range(variants)]
    return raw + vtlp


def test_copies_and_variants_share_one_identity():
    names = _names()
    ids = {recording_identity(n) for n in names}
    assert ids == {"jen_hey_seeree_0012.wav"}, ids
    # The copy index and the variant's ratio must BOTH be stripped: leaving either in is
    # a silent no-op - every file is its own group again and the leak is back.
    one = ["real_0_jay_x.wav", "real_9_jay_x.wav", "real_v2_1.25_jay_x.wav"]
    assert len({recording_identity(n) for n in one}) == 1, \
        "raw copies and a shifted variant did not collapse to one recording"


def test_non_copy_files_keep_their_own_identity():
    for name in ("kokoro_af_bella_5f3c.wav", "piper_pm_en_GB-aru-medium_03_ab12.wav",
                 "vtlp1.20_piper_pf_en_US-amy-medium_c9d0.wav",
                 "data/external-ish_ambient_0001.wav"):
        assert recording_identity(name) == name, name
    # And a name that merely STARTS like a copy is not collapsed with one.
    assert recording_identity("reallife_take.wav") == "reallife_take.wav"


def _split_of(names, split_count=0.1, **kw):
    part = group_partition(names, split_count, **kw)
    by_group = defaultdict(set)
    for n, s in part.items():
        by_group[recording_identity(n)].add(s)
    return part, by_group


def test_no_recording_straddles_the_split():
    names = []
    for speaker in ("jay", "jen", "ryan"):
        for clip in range(40):
            names += _names(speaker=speaker, base=f"hey_seeree_{clip:04d}",
                            copies=10, variants=4)
    part, by_group = _split_of(names)
    # "dropped" is the third legal value (a held-out recording's surplus copies); the
    # one thing that must never happen is a recording present in BOTH train and a
    # held-out bucket - that is the leak the whole function exists to close.
    both = {g: s for g, s in by_group.items() if "train" in s and (s & {"validation", "test"})}
    assert not both, f"{len(both)} recordings leaked across splits: {list(both)[:3]}"
    # And a held-out recording contributes exactly holdout_copies rows, no more.
    held = {g: s for g, s in by_group.items() if not (s <= {"train"})}
    assert held, "nothing was held out"
    for g, s in held.items():
        kept = [n for n in names if recording_identity(n) == g and part[n] != "dropped"]
        assert len(kept) <= 1, f"{g} contributes {len(kept)} validation rows"


def test_the_holdout_is_broad_and_shallow():
    # 2 * split_count of the RECORDINGS are held out, one row each: a selection signal
    # wants many distinct utterances, not ten copies of a few (measured: with the copies
    # kept, the identity-aware 10x runs scattered 59%->90% pooled and one collapsed).
    names = []
    for clip in range(100):
        names += _names(base=f"hey_seeree_{clip:04d}", copies=10, variants=0)
    part, by_group = _split_of(names, split_count=0.1)
    groups = set(by_group)
    held = {g for g, s in by_group.items() if s & {"validation", "test"}}
    assert abs(len(held) - 0.2 * len(groups)) <= 2, f"{len(held)} of {len(groups)} held"
    rows_held = sum(1 for v in part.values() if v in ("validation", "test"))
    assert rows_held == len(held), f"{rows_held} rows for {len(held)} recordings"
    dropped = sum(1 for v in part.values() if v == "dropped")
    assert dropped == len(held) * 9, f"{dropped} dropped, expected {len(held) * 9}"
    val = sum(1 for v in part.values() if v == "validation")
    test = sum(1 for v in part.values() if v == "test")
    assert 0.4 * rows_held <= test <= 0.6 * rows_held, f"val/test imbalance: {val}/{test}"


def test_holdout_copies_keeps_more_rows_when_asked():
    names = []
    for clip in range(50):
        names += _names(base=f"hey_seeree_{clip:04d}", copies=5, variants=0)
    part, by_group = _split_of(names, split_count=0.2, holdout_copies=3)
    held = [g for g, s in by_group.items() if s & {"validation", "test"}]
    per = [sum(1 for n in names if recording_identity(n) == g
               and part[n] in ("validation", "test")) for g in held]
    assert set(per) == {3}, f"holdout_copies=3 produced {set(per)}"


def test_partition_is_independent_of_input_order():
    names = _names(copies=6, variants=2) + [f"k_{i}.wav" for i in range(50)]
    a = group_partition(names, 0.1)
    b = group_partition(list(reversed(names)), 0.1)
    assert a == b, "the split moved when the file list order moved"
    # The upstream shuffle consumed the global RNG and depended on listing order; a
    # split that drifts between two runs of the same corpus makes two 'identical'
    # configurations differ by which clips were held out.
    assert group_partition(sorted(names), 0.1) == a


def test_copy_block_larger_than_the_holdout_refuses():
    # One recording and a split_count that rounds to no recordings held out at all:
    # validation would be empty, and an empty validation is a run whose checkpoint
    # selection silently does nothing.
    names = _names(copies=30, variants=0)
    try:
        group_partition(names, split_count=0.01)
    except SystemExit as e:                      # SystemExit carries the message itself
        assert "validation or test empty" in str(e), str(e)
    else:
        raise AssertionError("refused nothing: validation would be empty")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
