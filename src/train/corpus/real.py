"""Real voice recordings into a training corpus, weighted by repetition.

`real_samples_dir` is a parameter, not a constant from train.py's globals, so a
second trainer can point at the same recordings without importing train.py.
"""

from pathlib import Path

import numpy as np
import scipy.io.wavfile


def speaker_clip_counts(real_samples_dir: Path) -> dict:
    """{speaker: wav count} over the samples tree, the same way copy_real_samples reads it.

    Recursive: loose files and one-directory-per-speaker both count; loose files
    share the "(loose files)" key.
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

    The spec is kept as written in the corpus manifest: the rule is the identity,
    the multipliers are derived fresh each build, so recording more of a thin
    speaker shrinks their lift without editing a flag.
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

    WHY: one flat weight makes a speaker's share of the positive class an accident
    of recording, and the least-recorded speaker can read worse on her OWN training
    clips than the better-recorded voices do - a corpus-coverage failure, not a
    threshold one. So the weight is derived from the counts.

    The rule is EQUALISE UP, never down: the target is the richest speaker the
    balance set is actually balancing (an explicit `--real-copies-override` takes a
    speaker OUT of the set, so it cannot be the anchor) at the base weight; cutting
    the richest speaker back would balance the table by removing data. A hand-set
    override wins over the derived number - the decision beats the arithmetic.

    `max_multiplier` caps the lift (multiple of base_copies) and the cap is
    REPORTED: a cap below the base weight cannot bind (equalising never cuts),
    and the note says so instead of claiming a cap that did not apply.

    Returns ({speaker: copies}, notes:list[str]); the notes go to the run log -
    the table is the point, so it has to be visible.
    """
    explicit = dict(explicit or {})
    if not counts:
        return {}, ["no real recordings found - nothing to balance"]
    if speakers == "all":
        # parse_balance_spec's sentinel for "every speaker"; passing it straight
        # through would iterate it character-wise and raise on unknown speakers.
        speakers = None
    named = list(counts) if speakers is None else [s for s in speakers]
    unknown = [s for s in named if s not in counts]
    if unknown:
        raise ValueError(f"unknown speaker(s) {unknown}; the samples tree has "
                         f"{sorted(counts)}")
    # The anchor is the richest speaker actually being balanced: an explicit
    # override takes a speaker OUT of the set, so letting one anchor the target
    # would name a speaker nobody is balanced to.
    balanced = [s for s in named if s not in explicit]
    if balanced:
        target = max(counts[s] for s in balanced) * base_copies
        richest = max(balanced, key=lambda s: counts[s])
    else:
        target = richest = None
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
            # The cap is clamped at the base weight: equalising never cuts a speaker,
            # so a cap below the base weight cannot bind, and the note must say
            # "no lift" rather than claim a cap that did not apply.
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
    if balanced:
        notes.insert(0, f"balanced against {richest} at {target} rows "
                        f"({counts[richest]} clips x {base_copies})")
    else:
        notes.insert(0, "every speaker in the balance set carries an explicit "
                        "--real-copies-override, so there is nothing to equalise - the "
                        "weights below are the overrides")
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

    The copies are NOT redundant (openWakeWord path): they are written before the
    augmentation stage, which globs this whole directory, so each copy is
    augmented independently (background p=0.75, RIR, EQ, pitch, gain). The
    real_{i}_ names sort ~195 apart, so with mode="per_batch" every copy lands in
    a different batch and draws different noise - with augmentation_rounds=3,
    10 copies means 30 distinct variants. That is why 3 -> 10 in run 10 improved
    generalisation (held-out run-on detection 53% -> 77%): real clips are ~4% of
    the positive set but dominate the result - they carry room, mic and delivery
    characteristics Kokoro does not. Batch class balance is unaffected
    (batch_n_per_class); this only changes how often a real clip is drawn WITHIN
    the positive class.

    `per_speaker_copies` aims the same lever at one speaker: the global weight
    is one number for everybody, this is aimed at a voice that is thin and
    undetected on its OWN clips.

    `per_speaker_vtlp` adds N vocal-tract-length-shifted copies per clip for the
    named speakers (CHILD_STRETCH["m"], the synthetic child-lever's range): a
    shifted copy is a NEW acoustic variant - diversity not paid for in row
    count, which raw repetition is.

    Recordings may sit loose or in one directory per speaker; both layouts are
    picked up.

    NOTE FOR THE microWakeWord PORT: raw copies DO port now. mww augments each
    row per read, so N copies are N differently-augmented rows - presence, not
    repetition. What did forbid the port was the split: mww's partition was per
    FILE, so N copies of one recording scattered that speaker into all three
    splits, and mWW selects the weights it ships on validation
    average_viable_recall - a leak in the selection path. features.py now splits
    by recording identity (group_partition), so copies and their shifted
    variants always land together, and a thin voice moved in the expected
    direction on the holdout.
    What still does NOT port is the PER-SPEAKER raw-copy weight
    (--real-copies-override): mww's sampling weights are one number per FEATURE
    SET, and synthetic and real clips share the positives directory, so it
    cannot aim at one speaker. Per-speaker diversity is what --real-vtlp
    consumes. See the NOTE at the copy call in src/train/mww/corpus.py.
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

            # Flatten the path: two speakers recording the same phrase produce identical
            # basenames; wav_file.name alone would silently overwrite one with the other.
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
            # Shifted variants, named real_v{i}_ so they sort apart from the raw
            # copies: sort order spreads a speaker's clips across augmentation
            # batches, and a shifted copy draws its own noise/room like a raw copy.
            for i in range((per_speaker_vtlp or {}).get(speaker, 0)):
                ratio = float(np.random.uniform(*vtlp_span))
                # vocal_tract_shift's contract is int16 in/out: feeding it float
                # audio returns near-silence - a dtype bug a holdout probe paid for.
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
