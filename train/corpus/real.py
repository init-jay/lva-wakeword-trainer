"""Real voice recordings into a training corpus, weighted by repetition.

Moved from train.py. Two changes, both behaviour-preserving:

- `real_samples_dir` is now a parameter instead of a constant path
  read from train.py's module globals, so a second trainer can point at the same
  recordings without importing train.py.
- The `wake_word` parameter is gone. It was never referenced in the body.
"""

from pathlib import Path

import numpy as np
import scipy.io.wavfile


def speaker_clip_counts(real_samples_dir: Path) -> dict:
    """{speaker: wav count} over the samples tree, the same way copy_real_samples reads it.

    Recursive, so loose files and one-directory-per-speaker both count; loose files
    share the "(loose files)" key because there is no speaker to attribute them to.
    """
    root = Path(real_samples_dir)
    counts = {}
    if not root.exists():
        return counts
    for wav_file in sorted(root.rglob("*.wav")):
        rel = wav_file.relative_to(root)
        speaker = rel.parts[0] if len(rel.parts) > 1 else "(loose files)"
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def parse_balance_spec(spec: str, flag: str = "--balance-real-copies"):
    """'' is off (None); 'all' is the every-speaker sentinel; else a validated list
    of speaker names.

    The spec is kept as written in the corpus manifest, so the *rule* is the identity
    and the multipliers are derived fresh each build - which is what lets recording more
    clips of a thin speaker shrink their lift without editing a flag.
    """
    if not spec or not spec.strip():
        return None
    if spec.strip().lower() == "all":
        return "all"
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError(f"{flag}={spec!r} parses to no speaker names")
    return names


def balanced_copy_weights(counts: dict, base_copies: int, speakers=None,
                          explicit: dict = None, max_multiplier: float = 0.0):
    """Per-speaker copy counts that EQUALISE each speaker's rows, auto-derived from clip counts.

    WHY. `--real-copies` is one weight for everybody, so a speaker's share of the
    positive class is whatever accident of recording left them with. The measured
    consequence is this repo's most stubborn result: the least-recorded speaker can
    read worse on her OWN training clips than the better-recorded voices do - a
    corpus-coverage failure, not a threshold one. Presence, not timbre - so derive
    the weight from the counts instead of carrying it by hand, and let recording
    more of a thin speaker shrink the correction rather than change a flag.

    The rule is EQUALISE UP, never down: the target is the richest named speaker's row
    count at the base weight, and every other named speaker is lifted to it.
    Cutting the richest speaker back to the thinnest one's row count would balance
    the table by removing data rather than adding it, and the only measured way to
    spend one speaker's presence to buy another's came out negative.

    `speakers` is None (= every speaker) or an explicit list. A speaker named in
    `explicit` keeps that weight: a hand-set override is a decision, a derived number is
    arithmetic, and the decision wins.

    `max_multiplier` caps the lift (>0, as a multiple of base_copies) and the cap being
    hit is REPORTED, because "balance these three" silently turning into "one voice
    dominates the corpus" is the dilution failure above wearing a different hat. A cap
    below the base weight cannot bind - there is nothing left to cut - and the note says
    so instead of claiming a cap that did not apply.

    Returns ({speaker: copies}, notes:list[str]) - the notes are for printing: the table
    is the point of the exercise, so it has to be visible in the run log.
    """
    explicit = dict(explicit or {})
    if not counts:
        return {}, ["no real recordings found - nothing to balance"]
    if speakers == "all":
        # parse_balance_spec's sentinel for "every speaker". Accepted here because
        # passing it straight through used to iterate it character-wise and raise
        # about unknown speakers ['a', 'l', 'l'] - a true story from a dry run.
        speakers = None
    named = list(counts) if speakers is None else [s for s in speakers]
    unknown = [s for s in named if s not in counts]
    if unknown:
        raise ValueError(f"unknown speaker(s) {unknown}; the samples tree has "
                         f"{sorted(counts)}")
    target = max(counts[s] for s in named) * base_copies
    richest = max(named, key=lambda s: counts[s])
    weights, notes = {}, []
    for s, n in sorted(counts.items()):
        if s in explicit:
            weights[s] = explicit[s]
            notes.append(f"{s}: {n} clips -> {explicit[s]}x (explicit override, "
                         f"not balanced)")
            continue
        if s not in named:
            weights[s] = base_copies
            notes.append(f"{s}: {n} clips -> {base_copies}x (not in the balance set)")
            continue
        want = -(-target // n) if n else base_copies      # ceil division
        cap = int(base_copies * max_multiplier) if max_multiplier else 0
        if cap and want > cap:
            # The cap is clamped at the base weight, because equalising never cuts a
            # speaker (the rule above). A --balance-max-multiplier below 1.0 therefore
            # cannot bind at all: the most it can express is "no lift", so the weight
            # lands on the base and the note must say that. Reporting "capped at 0.5x"
            # next to a weight of 1.0x is a message contradicting the number beside it -
            # the same class of wrong as a label that disagrees with its command.
            capped = max(cap, base_copies)
            if capped == cap:
                notes.append(f"{s}: capped at {max_multiplier:g}x the base weight - "
                             f"{n} clips cannot reach {target} rows")
            else:
                notes.append(f"{s}: --balance-max-multiplier {max_multiplier:g} is below "
                             f"the base weight, so it cannot bind - the weight stays at "
                             f"{base_copies}x rather than cutting a speaker")
            want = capped
        weights[s] = max(want, base_copies)
    notes.insert(0, f"balanced against {richest} at {target} rows "
                   f"({counts[richest]} clips x {base_copies})")
    total = sum(weights[s] * counts[s] for s in counts)
    for s in sorted(counts):
        rows = weights[s] * counts[s]
        notes.append(f"  {s:<10} {counts[s]:>4} clips x {weights[s]:>3}x = {rows:>6} rows"
                     f" ({rows / total:5.1%} of the real rows)")
    return weights, notes


def copy_real_samples(real_samples_dir: Path, output_dir: Path, copies: int = 10,
                      per_speaker_copies: dict = None,
                      per_speaker_vtlp: dict = None) -> int:
    """Copy real voice recordings to training directory, `copies` times each.

    The copies are NOT redundant. They are written before openwakeword's
    augmentation stage, which globs this whole directory, so each copy is augmented
    independently: background noise from `background_paths` at p=0.75, a room
    impulse response, EQ, pitch shift and gain. AddBackgroundNoise runs
    mode="per_batch" and the copies are named real_{i}_... so sorting spreads them
    ~195 apart - every copy lands in a different batch and draws different noise.
    With augmentation_rounds=3 on top, 10 copies means 30 acoustically distinct
    variants of each recording, not 30 identical ones.

    That is why raising this from 3 to 10 in run 10 improved generalisation instead
    of overfitting: held-out run-on detection went 53% -> 77%, the largest single
    effect measured. Real clips are ~4% of the positive set by default and dominate
    the result, because real speech carries room, mic and delivery characteristics
    that Kokoro does not.

    Batch class balance is unaffected (batch_n_per_class fixes that), so this only
    changes how often a real clip is drawn WITHIN the positive class.

    `per_speaker_copies` overrides the weight for individual speakers. The global
    weight is one lever aimed at everybody; this one is aimed at a voice that is
    thin in the corpus and undetected on its OWN clips while the well-recorded
    voices detect fine - the same repetition lever, at one speaker.

    `per_speaker_vtlp` adds N vocal-tract-length-shifted copies per clip for the
    named speakers (CHILD_STRETCH["m"] ratios, the same range the synthetic child-
    lever uses). It exists because raw repetition buys the weak voice presence at
    the price of diluting the voices that were already detected, while one shifted
    copy is a NEW acoustic variant of the same recording - diversity is not paid
    for in row count, which is exactly how CHILD_STRETCH_FRACTION reasons about
    the synthetic side ("real-clip density drives the result, so buying child
    coverage by spending adult coverage is not a win").

    Recordings may sit loose in the samples directory or be grouped one directory
    per speaker (samples/speaker1/, samples/speaker2/, ...). Both layouts are
    picked up, so speakers can be added, re-recorded, or dropped independently.

    NOTE FOR THE microWakeWord PORT (it used to say raw copies must not port).
    mww generates its features up front but augments each row per read
    (background p=0.75, RIR, gain), so N raw copies are N DIFFERENTLY-augmented rows -
    presence, not repetition. What did forbid the port was the split: mww's
    train/validation/test partition was per FILE, so N copies of one recording scattered
    that speaker into all three splits, and mWW selects the weights it ships on
    validation average_viable_recall - a leak in the selection path, not just in a
    number. train/mww/features.py now splits by recording identity (group_partition), so
    copies and their shifted variants always land together, and the port is safe: a
    thin voice moved in the expected direction on the holdout, the same direction
    the higher global weight measured here.

    What still does NOT port is the PER-SPEAKER raw-copy weight (--real-copies-override):
    mww's sampling weights are one number per FEATURE SET, and synthetic and real clips
    share the positives directory, so it cannot aim at one speaker. Per-speaker diversity
    is what --real-vtlp consumes here (vocal-tract-shifted copies are distinct clips by
    construction). See the NOTE at the copy call in train/mww/corpus.py.
    """
    real_samples_dir = Path(real_samples_dir)
    if not real_samples_dir.exists():
        print("  No real samples found (record your voice first)")
        return 0

    # lazy: augment imports scipy too; keep real.py importable without it eager
    from train.corpus.augment import CHILD_STRETCH, vocal_tract_shift
    vtlp_span = CHILD_STRETCH["m"]

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    per_speaker = {}

    for wav_file in sorted(real_samples_dir.rglob("*.wav")):
        try:
            sr, data = scipy.io.wavfile.read(wav_file)
            if sr != 16000:
                from scipy.signal import resample
                num_samples = int(len(data) * 16000 / sr)
                data = resample(data, num_samples)
                data = np.clip(data, -32768, 32767).astype(np.int16)

            # Flatten the path into the destination filename. Two speakers recording
            # the same phrase produce identical basenames (hey_seeree_0001.wav), so
            # using wav_file.name alone would silently overwrite one with the other.
            rel = wav_file.relative_to(real_samples_dir)
            stem = "_".join(rel.with_suffix("").parts)
            speaker = rel.parts[0] if len(rel.parts) > 1 else "(loose files)"
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
            weight = (per_speaker_copies or {}).get(speaker, copies)

            # Create multiple copies to weight real samples higher
            for i in range(weight):
                dest = output_dir / f"real_{i}_{stem}.wav"
                scipy.io.wavfile.write(str(dest), 16000, data)
                count += 1
            # Shifted variants: N NEW acoustic forms of this recording, named
            # real_v{i}_ so they sort apart from the raw copies (the augmentation
            # rounds spread a speaker's clips across batches by sort order, and a
            # shifted copy draws its own noise/room like a raw copy does).
            for i in range((per_speaker_vtlp or {}).get(speaker, 0)):
                ratio = float(np.random.uniform(*vtlp_span))
                # vocal_tract_shift's contract is int16 in, int16 out (it
                # peak-normalises against 32767). Feeding it float audio returns
                # near-silence - the dtype bug a holdout probe paid for.
                shifted = vocal_tract_shift(
                    np.clip(data, -32768, 32767).astype(np.int16), ratio)
                dest = output_dir / f"real_v{i}_{ratio:.2f}_{stem}.wav"
                scipy.io.wavfile.write(str(dest), 16000, shifted)
                count += 1
        except Exception as e:
            print(f"  Error processing {wav_file}: {e}")

    if per_speaker:
        detail = ", ".join(f"{s}: {n}" for s, n in sorted(per_speaker.items()))
        print(f"  Found {sum(per_speaker.values())} real samples ({detail})")
    if per_speaker_copies:
        overrides = ", ".join(f"{s}: {w}x" for s, w in sorted(per_speaker_copies.items()))
        print(f"  Per-speaker copy overrides: {overrides}")
    if per_speaker_vtlp:
        v = ", ".join(f"{s}: +{w} VTLP copies [{vtlp_span[0]:.2f}-{vtlp_span[1]:.2f}x]"
                      for s, w in sorted(per_speaker_vtlp.items()))
        print(f"  Per-speaker shifted variants: {v}")
    print(f"  Copied {count} real voice samples ({copies}x base weight)")
    return count
