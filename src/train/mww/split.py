"""The identity-aware corpus split, on its own.

Not inside features.py: features.py imports microwakeword and mmap_ninja at
module scope, so the partition arithmetic could not run under the oww venv
(Makefile:53). The split decides which recordings select the shipped weights.

Upstream's split is per FILE over a directory holding N copies of each
recording, its vocal-tract variants, and formant-shifted synthetic copies -
so copy 3 of one utterance can land in training while copy 7 lands in
validation, and the selection metric measures recall of something already
seen. The fix: hold out whole recordings, one row each.
"""

import hashlib
import os
import re

# real_<copy>_<speaker>_<file>.wav               (copy_real_samples' raw copies)
# real_v<variant>_<ratio>_<speaker>_<file>.wav   (its vocal-tract variants)
# vtlp<ratio>_<file>.wav                         (corpus/augment.py's shifted TTS clips)
#
# The third form is written by a different producer (corpus/augment.py, no
# real_ prefix): a shifted synthetic clip is the same utterance, so it must
# share its source's identity.
_COPY_PREFIX_RE = re.compile(
    r"^(?:real_(?:v\d+_[0-9.]+_|\d+_|v\d+_)|vtlp[0-9.]+_)")   # raw string: \d is a class, not an escape


def recording_identity(name):
    """The filename of the RECORDING a corpus file is a copy of.

      real_7_<speaker>_<word>_0012.wav        copy_real_samples' raw copy
      real_v3_1.22_<speaker>_<word>_0012.wav  its vocal-tract variant, same recording
      vtlp1.22_<tts clip>.wav                 corpus/augment.py's shift of a TTS render

    The first two are one human utterance. The third groups with the UNSHIFTED
    render it was made from - a synthetic and a real clip are never the same
    utterance, and the regex only strips prefixes, so it cannot make them one.
    Everything else (an unshifted TTS render, an ambient set member) is its own
    identity and partitions exactly as upstream's per-file split did.
    """
    return _COPY_PREFIX_RE.sub("", name)


def group_partition(names, split_count, holdout_copies=1):
    """{name: 'train'|'validation'|'test'|'dropped'}, holding whole recordings together.

    2 * split_count of the RECORDINGS are held out, half validation / half test.
    The unit is the recording, not the row: at a high copy factor a row budget
    holds out a handful of utterances counted N times each instead of many
    counted once, and mWW picks the weights it ships on exactly that number.

    NO SEED PARAMETER, ON PURPOSE: the partition is a function of the recording
    identities alone (hashlib, because `hash()` is salted per process), so a
    re-seeded run selects against the SAME held-out recordings - otherwise the
    seed comparison measures the split, not the seed.

    `holdout_copies`: a held-out recording keeps that many rows (default 1); the
    rest of its block is DROPPED - not moved to training, which would undo the
    point. Refuses (SystemExit) if validation or test is empty: an empty
    selection set is a run whose checkpoint choice does nothing; the fix is a
    larger split_count or fewer copies.
    """
    groups = {}
    for n in names:
        groups.setdefault(recording_identity(n), []).append(n)
    # THE UNIT OF THE BUDGET IS THE RECORDING. Upstream's number is a row share;
    # at copy factor 1 the two are identical.
    hold_groups = len(groups) * 2 * float(split_count)
    dropped = 0
    out = {}
    # Which held-out bucket is shorter, by ROWS: alternating per group would tilt it
    # whenever group sizes differ.
    sizes = {"validation": 0, "test": 0}
    # Chosen by hash, not name order (name order would put a whole speaker on one
    # side - a hidden correlation with the corpus layout).
    for i, (ident, members) in enumerate(sorted(
            groups.items(),
            # fsencode, not str.encode: a surrogate-escaped filename (not valid utf-8)
            # raises UnicodeEncodeError on .encode(); the pipeline before this
            # module just shuffled such a name.
            key=lambda kv: hashlib.sha256(os.fsencode(kv[0])).hexdigest())):
        if i < hold_groups:
            split = min(sizes, key=sizes.get)
            # Deterministic which rows survive: name order, arbitrary but not data-dependent.
            keep = sorted(members)[:max(1, int(holdout_copies))]
            for m in members:
                if m in keep:
                    out[m] = split
                    sizes[split] += 1
                else:
                    # Named, not omitted: build_split excludes "dropped" rows; a caller
                    # that forgot the value fails loudly instead of training on them.
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

    Its own function because it is the half that fails SILENTLY: `group_partition`
    iterates in hash order, while the caller selects rows BY POSITION from a
    dataset in listing order - enumerating the assignment's values would address
    unrelated rows and every count would still look right. Indexing `names` is
    the only correct source, and checkable here without TensorFlow.

    `by_mode` has exactly the three keys a DatasetDict needs; "dropped" rows come
    back as names - a caller that forgot them would silently train on a held-out
    recording.
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
