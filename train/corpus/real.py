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


def copy_real_samples(real_samples_dir: Path, output_dir: Path, copies: int = 10) -> int:
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

    Recordings may sit loose in the samples directory or be grouped one directory
    per speaker (samples/speaker1/, samples/speaker2/, ...). Both layouts are
    picked up, so speakers can be added, re-recorded, or dropped independently.

    NOTE FOR THE microWakeWord PORT: the repetition trick is specific to a pipeline
    that augments by globbing a directory. microWakeWord generates spectrogram
    features up front, so N identical copies would be N identical feature rows
    rather than N augmented variants - actively worse than one. Weighting there has
    to happen through its own sampling/class weights instead. See ARCHITECTURE.md.
    """
    real_samples_dir = Path(real_samples_dir)
    if not real_samples_dir.exists():
        print("  No real samples found (record your voice first)")
        return 0

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

            # Create multiple copies to weight real samples higher
            for i in range(copies):
                dest = output_dir / f"real_{i}_{stem}.wav"
                scipy.io.wavfile.write(str(dest), 16000, data)
                count += 1
        except Exception as e:
            print(f"  Error processing {wav_file}: {e}")

    if per_speaker:
        detail = ", ".join(f"{s}: {n}" for s, n in sorted(per_speaker.items()))
        print(f"  Found {sum(per_speaker.values())} real samples ({detail})")
    print(f"  Copied {count} real voice samples ({copies}x weight)")
    return count
