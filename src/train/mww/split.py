"""The identity-aware corpus split, on its own.

Why this is not inside features.py: features.py imports microwakeword and
mmap_ninja at module scope, so a test of pure partition arithmetic would need
the whole TensorFlow environment (Makefile:53 runs the suite with the oww venv).
The split is the piece that decides which recordings select the shipped
weights, so it is worth being able to run it anywhere.

THE PROBLEM IT SOLVES. Upstream's split is per FILE over a directory that
copy_real_samples has filled with N copies of each recording plus its
vocal-tract variants, and that corpus/augment.py has filled with formant-shifted
copies of the synthetic clips. Copy 3 of one utterance can land in training
while copy 7 of the same utterance lands in validation, so the selection metric
measures recall of something the model has already seen. The fix is to hold out
whole recordings and spend the validation budget on distinct recordings, one row
each.
"""

import hashlib
import os
import re

# real_<copy>_<speaker>_<file>.wav               (copy_real_samples' raw copies)
# real_v<variant>_<ratio>_<speaker>_<file>.wav   (its vocal-tract variants)
# vtlp<ratio>_<file>.wav                         (corpus/augment.py's shifted TTS clips)
#
# The third form is easy to miss because it is written by a different producer:
# src/train/corpus/augment.py:143 names its formant-shifted copies vtlp{ratio:.2f}_{clip.name},
# with no real_ prefix. A shifted copy of a synthetic clip is the same utterance in the
# same voice, so leaving it its own identity leaks exactly what this module exists to stop.
_COPY_PREFIX_RE = re.compile(
    r"^(?:real_(?:v\d+_[0-9.]+_|\d+_|v\d+_)|vtlp[0-9.]+_)")   # raw string: \d is a class, not an escape


def recording_identity(name):
    """The filename of the RECORDING a corpus file is a copy of.

    Two producers write copies, and each has to be stripped or the copy keeps its own
    identity and can land on the other side of the split from its source:

      real_7_<speaker>_<word>_0012.wav        copy_real_samples' raw copy of a recording
      real_v3_1.22_<speaker>_<word>_0012.wav  its vocal-tract variant of the SAME recording
      vtlp1.22_<tts clip>.wav                 corpus/augment.py's shift of a TTS render

    The first two are one human utterance, copied and weighted. The third is a shifted
    TTS render and groups with the UNSHIFTED render it was made from - not with any real
    recording; a synthetic clip and a human one are never the same utterance, and the
    regex only ever strips a prefix, so it cannot make them one. Everything else (an
    unshifted TTS render, an ambient set member) is its own identity, so it partitions
    exactly as it did when Clips split the directory per file.
    """
    return _COPY_PREFIX_RE.sub("", name)


def group_partition(names, split_count, holdout_copies=1):
    """{name: 'train'|'validation'|'test'|'dropped'}, holding whole recordings together.

    The share is the same as upstream's: 2 * split_count of the ROWS are held out, split
    half validation / half test. Assignment is by hashlib over the recording identity, so
    it is stable across runs and across process start-ups (`hash()` is salted per
    process); the row counts can only wobble with the corpus, never with the order the
    files were listed in.

    THERE IS NO SEED PARAMETER, ON PURPOSE. The partition is a function of the recording
    identities alone, so it cannot move when a training seed does. That is the property
    worth having: a re-seeded run must select its checkpoints against the SAME held-out
    recordings, or two seeds differ in which clips were train and which were validation,
    and the seed comparison measures the split instead of the seed. A `--split-seed` flag
    used to sit on top of this and did nothing - the digest never read it - so it is gone
    rather than wired up; wiring it would have bought the failure mode just described.
    "Does the result depend on WHICH recordings were held out?" is still answerable, just
    not from a command line: change the hash input below (a salt prefix on the identity),
    rebuild the features, and compare the two models. One edit to this module, deliberately
    not a flag - a flag is what makes it easy to move the holdout by accident and then
    read the difference as the seed.

    `holdout_copies` is the part that matters once the copy factor is high. The first
    identity-aware version held out whole copy blocks, which made the budget a ROW budget:
    with N copies of every recording, holding out a fifth of the rows holds out a fifth of
    N times fewer recordings, so validation became a handful of utterances counted N times
    each instead of many counted once - and mWW picks the weights it ships on exactly that
    number, so the runs scattered where the leaky per-file split beside them had looked
    consistent (the duplication averaged the selection signal out). So: the budget is spent
    on RECORDINGS, a held-out recording keeps `holdout_copies` of its rows (1 by default),
    and the rest of its block is DROPPED - not moved to training, which would undo the
    whole point. Dropping is why the counts are printed.

    Refuses (SystemExit) rather than returning a split with an empty validation or test:
    an empty selection set is a run whose checkpoint choice does nothing. It happens for
    a real reason - too few recordings to hold a third of - and the fix is a larger
    split_count or fewer copies, so say exactly that.
    """
    groups = {}
    for n in names:
        groups.setdefault(recording_identity(n), []).append(n)
    # THE UNIT OF THE BUDGET IS THE RECORDING. 2 * split_count of the groups, so the
    # holdout is broad (many distinct utterances) and shallow (few copies of each) -
    # which is what a selection signal needs. Upstream's number is a row share; with a
    # copy factor of 1 the two are identical, which is why this reads as a change in
    # meaning and mostly is not.
    hold_groups = len(groups) * 2 * float(split_count)
    dropped = 0
    out = {}
    # Which of the two held-out buckets is shorter, by ROWS: upstream halves the holdout,
    # and alternating per group would tilt it whenever group sizes differ.
    sizes = {"validation": 0, "test": 0}
    # Sort by the digest so the holdout is chosen by hash, not by name order (name order
    # would put the whole of speaker 'emily' in validation and none in test, or whatever
    # the alphabet says, which is a hidden correlation with the corpus layout).
    for i, (ident, members) in enumerate(sorted(
            groups.items(),
            # fsencode, not str.encode: a filename the filesystem handed back with
            # surrogate escapes (a byte sequence that is not valid utf-8) raises
            # UnicodeEncodeError on .encode(), and the pipeline before this module just
            # shuffled such a name rather than dying on it.
            key=lambda kv: hashlib.sha256(os.fsencode(kv[0])).hexdigest())):
        if i < hold_groups:
            split = min(sizes, key=sizes.get)
            # Deterministic which rows survive: name order inside one recording is
            # copy-index order, which is arbitrary but at least not data-dependent.
            keep = sorted(members)[:max(1, int(holdout_copies))]
            for m in members:
                if m in keep:
                    out[m] = split
                    sizes[split] += 1
                else:
                    # Named, not omitted: build_split excludes every "dropped" row from
                    # all three splits, and a caller that forgot to handle the value
                    # fails loudly instead of silently training on it.
                    out[m] = "dropped"
                    dropped += 1
        else:
            for m in members:
                out[m] = "train"
    counts = {s: sum(1 for v in out.values() if v == s)
              for s in ("train", "validation", "test", "dropped")}
    counts["dropped"] = dropped
    if not counts["validation"] or not counts["test"]:
        raise SystemExit(
            f"ERROR: the identity-aware split left validation or test empty "
            f"({counts['validation']}/{counts['test']} rows of {len(names)} files, "
            f"{len(groups)} recordings). split_count {split_count} holds out "
            f"{hold_groups:.0f} of {len(groups)} recordings, which rounded to none: "
            f"raise --split-count.")
    return out


def partition_indices(names, split_count, holdout_copies=1):
    """(by_mode, dropped): ROW INDICES per split, in the order `names` was given.

    This exists as its own function because it is the half of the split that fails
    SILENTLY. `group_partition` returns {name: mode} and iterates groups in hash order,
    while the caller selects rows BY POSITION from a dataset in listing order - so
    enumerating the assignment's values as indices would address unrelated rows, and
    every count would still look right because the totals are unchanged. Indexing
    `names` is the only correct source, and here it is checkable without the TensorFlow
    environment src/train/mww/features.py needs to run.

    `by_mode` has exactly the three keys a DatasetDict of splits needs; "dropped" rows
    are returned separately as names, because they are excluded from every split and a
    caller that forgot to handle them would silently train on a held-out recording.
    """
    assignment = group_partition(names, split_count, holdout_copies)
    by_mode = {"train": [], "validation": [], "test": []}
    dropped = []
    for idx, name in enumerate(names):
        mode = assignment[name]
        if mode == "dropped":
            dropped.append(name)
        else:
            by_mode[mode].append(idx)
    return by_mode, dropped
