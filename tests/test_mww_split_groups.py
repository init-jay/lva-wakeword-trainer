"""Guards for src/train/mww/features.py's identity-aware split.

WHY THESE TESTS EXIST. microWakeWord splits its feature corpus per FILE
(microwakeword/audio/clips.py, an HF train_test_split over the directory). A person's
real recordings enter the corpus as N copies of each - that is how openWakeWord weights
them, and the copy weight is a lever on this target. Per-file scattering puts copy 3 of
a recording in training and copy 7 of it in validation, and mWW then SELECTS the weights
it ships on validation `average_viable_recall` - so the copies are not just a flattering
number, they bias which checkpoint ships. src/train/corpus/real.py's NOTE told the port not
to bring raw copies for exactly this reason; the split below is what makes the note
obsolete, and these tests are what make the split safe.

The contract:

* every derived file of one recording shares its split - the raw copies
  (real_<i>_...), the vocal-tract variants of REAL clips (real_v<i>_<ratio>_...) and the
  formant-shifted copies of SYNTHETIC clips (vtlp<ratio>_<source>...), which three
  different producers write into the same directory;
* anything that is not a copy keeps its own identity, so a corpus with no copies
  partitions exactly as it did before (TTS renders and ambient sets are untouched);
* 2 * split_count of the RECORDINGS are held out, one row each (broad and shallow):
  a row budget at a high copy factor holds out a fraction of the recordings counted many
  times each, which is the wrong shape for a selection signal - that is what the first
  identity-aware run measured scattering;
* the partition takes no seed: it is a hash of the recording identities, so it cannot
  move when a training seed does, and it is stable across runs and across input order
  (hashlib, because `hash()` is salted per process);
* a corpus whose copy block is bigger than the holdout refuses instead of quietly
  producing an empty validation set (which would make selection a no-op).
"""

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Imported from train.mww.split, not features: features pulls in microwakeword and
# mmap_ninja at module scope, and the Makefile:53 suite runs under the oww venv.
from train.mww.split import (group_partition, partition_indices,  # noqa: E402
                             recording_identity)


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
                 "data/external-ish_ambient_0001.wav"):
        assert recording_identity(name) == name, name
    # And a name that merely STARTS like a copy is not collapsed with one.
    assert recording_identity("reallife_take.wav") == "reallife_take.wav"


def test_a_shifted_synthetic_copy_groups_with_its_source():
    # corpus/augment.py writes its formant-shifted TTS copies as vtlp<ratio>_<source>.wav
    # INTO THE SAME positives/ directory as the source clip, so both are candidates for
    # the same partition. Same voice, same utterance, shifted formants: letting the copy
    # keep its own identity is the real-copies leak one producer over, and it was pinned
    # that way by an earlier version of this file. If the naming ever changes, this test
    # and _COPY_PREFIX_RE change together.
    source = "piper_pm_en_GB-alan-low_47ceb314.wav"
    shifted = "vtlp1.15_" + source
    assert recording_identity(shifted) == source, recording_identity(shifted)
    names = [source, shifted] + [f"k_{i}.wav" for i in range(40)]
    part = group_partition(names, 0.1)
    # Same side of the train/holdout boundary, always. "dropped" belongs to the held-out
    # side: a held-out recording keeps holdout_copies rows and the rest of its block is
    # dropped, so a two-file group can legitimately read validation/dropped - what it must
    # never read is train/validation, which is the leak.
    held_out = {"validation", "test", "dropped"}
    assert (part[source] in held_out) == (part[shifted] in held_out), \
        f"a shifted copy straddled the split from its source: {part[source]}/{part[shifted]}"


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
    # wants many distinct utterances, not ten copies of a few (with the copies kept, the
    # identity-aware runs scattered across repeats of one configuration, and one collapsed).
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


def test_partition_indices_address_the_names_they_came_from():
    """by_mode holds INDICES INTO names, not positions in the assignment's iteration order.

    group_partition returns {name: mode} and walks groups in hash order, while the caller
    selects rows BY POSITION from a dataset in listing order - so enumerating the
    assignment's values as indices would address unrelated rows, and every COUNT would
    still look right because the totals are unchanged. That is why the correspondence is
    pinned here rather than the counts.
    """
    names = [f"real_{i}_jay_hey_seeree_{i:04d}.wav" for i in range(60)]
    names += [f"vtlp1.20_jen_hey_seeree_{i:04d}.wav" for i in range(60)]
    assignment = group_partition(names, 0.1)
    by_mode, dropped = partition_indices(names, 0.1)

    assert set(by_mode) == {"train", "validation", "test"}, by_mode.keys()
    for mode, idxs in by_mode.items():
        assert [names[i] for i in idxs] == [
            n for n in names if assignment[n] == mode], mode
        assert idxs == sorted(idxs), f"{mode} must stay in listing order"
        assert all(0 <= i < len(names) for i in idxs), mode
    assert dropped == [n for n in names if assignment[n] == "dropped"]

    # The trap, spelled out: reading the assignment's values as indices gives a different
    # selection, so the assertions above are not vacuous on this fixture.
    naive = {"train": [], "validation": [], "test": []}
    for i, mode in enumerate(assignment.values()):
        if mode in naive:
            naive[mode].append(i)
    assert naive["train"] != by_mode["train"], (
        "fixture no longer separates the two readings - hash order happens to coincide "
        "with listing order here, so change the names before trusting this test")


def test_extra_copies_of_a_held_out_recording_come_back_by_name():
    """Rows excluded from every split are RETURNED, not merely absent from by_mode.

    A caller that forgot to handle them would silently train on a held-out recording, and
    nothing downstream would notice: the counts still add up.
    """
    base = [f"real_{i}_jay_hey_seeree_{i:04d}.wav" for i in range(60)]

    def with_copies(src):
        """`base` plus five extra copies of the recording `src` is a copy of."""
        return base + [f"real_{i}_{recording_identity(src)}" for i in range(200, 205)]

    # Duplicating a recording changes the row counts the split sizes are derived from, so
    # the recording to duplicate is chosen against the FINAL name list: only one the
    # partition holds out can have copies dropped from it. Deterministic (hashlib over the
    # identity, no seed), so the choice is stable across runs and machines.
    src = next(n for n in base
               if group_partition(with_copies(n), 0.1)[n] in ("validation", "test"))
    ident = recording_identity(src)
    names = with_copies(src)

    by_mode, dropped = partition_indices(names, 0.1, holdout_copies=1)
    assert len(dropped) == 5, f"6 copies of one recording, 1 may survive: {dropped}"
    assert all(recording_identity(n) == ident for n in dropped), dropped
    kept = [names[i] for m in ("train", "validation", "test") for i in by_mode[m]]
    assert not any(n in dropped for n in kept), "a dropped row reached a split"
    assert sum(1 for n in kept if recording_identity(n) == ident) == 1, kept


def test_a_name_the_filesystem_cannot_decode_still_partitions():
    """A filename carrying surrogate escapes partitions instead of raising.

    A directory holding a byte sequence that is not valid utf-8 yields a `Path.name` with
    surrogate escapes; hashing that with `.encode()` raises UnicodeEncodeError. The
    pipeline upstream of this module had already shuffled such a name without complaint,
    so the split was the first place it could fail - and it failed the whole feature build.
    `os.fsencode` round-trips the original bytes instead.
    """
    odd = "real_0_jay_hey_seeree_\udcff\xfe.wav"      # not encodable as utf-8
    try:
        odd.encode()
    except UnicodeEncodeError:
        pass
    else:
        raise AssertionError("fixture name must be unencodable or this proves nothing")

    names = [odd] + [f"real_{i}_jay_hey_seeree_{i:04d}.wav" for i in range(40)]
    part = group_partition(names, 0.1)                 # raised before the fix
    assert part[odd] in ("train", "validation", "test", "dropped"), part[odd]

    by_mode, dropped = partition_indices(names, 0.1)
    accounted = dropped + [names[i] for m in by_mode.values() for i in m]
    assert sorted(accounted) == sorted(names), "every name lands in exactly one place"


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
