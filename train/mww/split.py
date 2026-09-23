"""The identity-aware corpus split, on its own.

Why this is not inside features.py: features.py imports microwakeword and
mmap_ninja at module scope, so a test of pure partition arithmetic would need
the whole TensorFlow environment (Makefile:53 runs the suite with the oww venv).
The split is the piece that decides which recordings select the shipped
weights, so it is worth being able to run it anywhere.

THE PROBLEM IT SOLVES (measured 2026-09-24). Upstream's split is per FILE over a
directory that copy_real_samples has filled with N copies of each recording plus
its vocal-tract variants. Copy 3 of a jen utterance can land in training while
copy 7 lands in validation, so the selection metric measures recall - the
real-copies=10 probe run scored 90.6% pooled on it. The fix is to hold out whole
recordings and spend the validation budget on distinct recordings, one row each.
"""

import hashlib
import re

# real_<copy>_<speaker>_<file>.wav          (copy_real_samples' raw copies)
# real_v<variant>_<ratio>_<speaker>_<file>.wav   (its vocal-tract variants)
_COPY_PREFIX_RE = re.compile(
    r"^real_(?:v\d+_[0-9.]+_|\d+_|v\d+_)")   # raw string: \d is a class, not an escape


def recording_identity(name):
    """The filename of the RECORDING a corpus file is a copy of.

    `real_7_jen_hey_seeree_0012.wav` and `real_v3_1.22_jen_hey_seeree_0012.wav` are one
    utterance, weighted and shifted; they must never straddle a split. Everything else -
    a TTS render, an ambient set member - is its own identity, so it partitions exactly
    as it did when Clips split the directory per file.
    """
    return _COPY_PREFIX_RE.sub("", name)


def group_partition(names, split_count, seed=10, holdout_copies=1):
    """{name: 'train'|'validation'|'test'}, holding whole recordings together.

    The share is the same as upstream's: 2 * split_count of the ROWS are held out, split
    half validation / half test. Assignment is by hashlib over the recording identity, so
    it is stable across runs and across process start-ups (`hash()` is salted per
    process); the row counts can only wobble with the corpus, never with the order the
    files were listed in.

    `holdout_copies` is the part that matters at 10x, and it is a measured correction,
    not a theory. The first identity-aware version held out whole copy blocks, which
    made the budget a ROW budget: at 10x, 20% of the rows is 2% of the recordings, so
    validation became ~30 voices counted ten times each instead of ~300 counted once -
    and mWW picks the weights it ships on exactly that number. Measured 2026-09-24, the
    runs did then scatter (pooled detection 59%, 71%, 90%, and one model that fires on
    everything at ROC AUC 0.17) where the leaky per-file split beside them had been
    consistent (86%, 80%) because the duplication averaged the selection signal out. So:
    the budget is spent on RECORDINGS, a held-out recording keeps `holdout_copies` of
    its rows (1 by default), and the rest of its block is DROPPED - not moved to
    training, which would undo the whole point. Dropping is why the counts are printed.

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
            groups.items(), key=lambda kv: hashlib.sha256(kv[0].encode()).hexdigest())):
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
