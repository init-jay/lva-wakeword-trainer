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

    `per_speaker_copies` overrides the weight for individual speakers (the 2026-09-23
    clean-detection work: the seed-55/56 oww models measured ryan at 1/6 holdout
    and 36% on his OWN training clips - 39/78 still undetected at threshold 0.1,
    i.e. his 78 clips at 10x are 3.5% of the positive set and the model simply
    never learned his timbre, while jay at 160 clips at 10x measured 80% on his
    own clips. Same lever as 3->10, aimed at one speaker.)

    `per_speaker_vtlp` adds N vocal-tract-length-shifted copies per clip for the
    named speakers (CHILD_STRETCH["m"] ratios, the same range the synthetic child-
    lever uses). It exists because raising ryan to 40x raw copies moved his holdout
    1/6 -> 3/6 but cost jay 33/35 -> 23/35: raw repetition buys presence at the
    price of diluting the adult voices, while one shifted copy is a NEW acoustic
    variant of the same recording - diversity is not paid for in row count, which
    is exactly how CHILD_STRETCH_FRACTION reasons about the synthetic side ("real-
    clip density drives the result, so buying child coverage by spending adult
    coverage is not a win").

    Recordings may sit loose in the samples directory or be grouped one directory
    per speaker (samples/speaker1/, samples/speaker2/, ...). Both layouts are
    picked up, so speakers can be added, re-recorded, or dropped independently.

    NOTE FOR THE microWakeWord PORT (verified there 2026-09-23,
    train/mww/corpus.py): only the SHIFTS port. mww generates its features up
    front but augments each row per read (background p=0.75, RIR, gain), so N
    raw copies are N DIFFERENTLY-augmented rows, not identical feature rows -
    the reason they must not port is stronger: mww's train/val/test split is
    per FILE (microwakeword/audio/clips.py:140-156), so N copies of one clip
    scatter that speaker into all three splits - validation/test leak.
    Vocal-tract-shifted copies are distinct clips and do not leak, which is
    what mww's --real-vtlp consumes. See train/mww/corpus.py's NOTE at its
    copy call.
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
                # near-silence - the dtype bug a holdout probe paid for in 2026-09-23.
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
