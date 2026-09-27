"""Audio transforms applied to a corpus of WAVs: trimming, and child-range copies.

Both trainers need both: both frontends place a fixed-size window relative to the
END of the array, so trailing silence displaces the phrase and teaches a later
alignment (openWakeWord's mechanism is in trim_silence; mww's clip is 1500 ms vs
2000 ms - the failure mode is the same). Child-range copies cover a corpus that is
otherwise adult-only - the largest single tuning win (run 13: a 4-year-old
24% -> 83%).

add_child_range_copies reads the voice's sex from the filename (kokoro_af_bella_*
-> "af"; Piper clips are named piper_p{sex}_... so the same extraction works).

time_stretch lives in tts_protocol/audio.py (the shared piper engine applies speed
with it, and that layer must not import train/); re-exported here, so
`from train.corpus.augment import time_stretch` keeps working.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

from tts_protocol.audio import time_stretch  # noqa: E402  (re-export; see module docstring)

# Vocal-tract-length perturbation, per voice sex.
#
# Run 12 measured the second speaker - a 4-year-old - at 24% against 97% for the
# adult (34% on his OWN training clips): his fundamental (median 291 Hz) sits
# outside the range of almost everything the model has seen (adult female 269 Hz,
# adult male 153 Hz). openwakeword's PitchShift is +/-3 semitones at p=0.25
# against a 13.6-semitone gap. The ratios are per sex: a listening test put
# af_bella at 1.28 closest to the child, and male voices "sound like teenagers
# up to 1.30 and useless above that (chipmunk)" - training on an artefact teaches
# the artefact.
#
# f -> 272-306 Hz, straddling the child. m -> 152-172 Hz, the adult-male-to-child gap.
CHILD_STRETCH = {"f": (1.20, 1.35), "m": (1.15, 1.30)}

# These clips are ADDED to the corpus, not substituted: substituting would thin
# adult coverage in proportion, and real-clip density drives the result.
CHILD_STRETCH_FRACTION = 0.5


def vocal_tract_shift(data: np.ndarray, ratio: float, sr: int = 16000) -> np.ndarray:
    """Raise F0 and formants by `ratio`, keeping the clip's original duration.

    Resampling alone raises pitch and formants together - what a shorter vocal
    tract does, and why this reaches a child voice where a formant-corrected shift
    would not - but shortens the clip by the same factor; the stretch puts the
    duration back. Verified against ffmpeg asetrate+atempo: same F0 to the
    estimator's resolution, and atempo drifts ~3% in length.
    """

    frac = Fraction(ratio).limit_denominator(100)
    shifted = resample_poly(data.astype(np.float64), frac.denominator, frac.numerator)
    out = time_stretch(shifted, float(ratio), sr=sr)

    peak = np.abs(out).max()
    if peak > 32767:
        out = out * (32767 / peak)
    return out.astype(np.int16)


def add_child_range_copies(directory: Path, desc: str,
                           fraction: float = CHILD_STRETCH_FRACTION) -> int:
    """Add pitch/formant-shifted copies of the SYNTHETIC clips in `directory`.

    Synthetic only: the child needs no shifting, and the adult speaker is male,
    so shifting the real recordings reaches the teen range that ~15 Kokoro male
    voices already cover more cheaply than 160 clips of one speaker. Sex comes
    from the filename (voice prefix second letter; piper_p{sex}_... the same
    position), so Piper clips participate on equal terms; an unknown sex
    (piper_pu_...) is skipped rather than shifted by a guessed ratio - shifting a
    male voice by the female range produces the artefact run 12 warned about.

    If Piper clips displace Kokoro ones without being shiftable, the child-range
    lever's COVERAGE shrinks in proportion, and the likeliest casualty is the
    4-year-old. Copies are ADDED - see CHILD_STRETCH_FRACTION.
    """
    clips = [p for p in sorted(directory.glob("*.wav"))
             if p.name.startswith(("kokoro_", "runon_", "piper_"))]
    if not clips:
        return 0

    written = 0
    skipped_unknown = 0
    for clip in tqdm(clips, desc=desc, unit="clip"):
        # kokoro_{voice}_{uuid}.wav -> af_bella: sex is the voice prefix's second
        # letter (af_/bf_ female, am_/bm_ male); piper_p{sex}_... same index.
        parts = clip.stem.split("_")
        if len(parts) < 3 or len(parts[1]) != 2:
            continue
        sex = parts[1][1]
        if sex == "u":
            skipped_unknown += 1
            continue
        span = CHILD_STRETCH.get(sex)
        if span is None:
            continue
        if np.random.random() >= fraction:
            continue

        try:
            sr, data = scipy.io.wavfile.read(clip)
        except Exception:
            continue
        if sr != 16000 or data.ndim != 1 or len(data) < 480:
            continue

        ratio = float(np.random.uniform(*span))
        shifted = vocal_tract_shift(data, ratio)
        scipy.io.wavfile.write(
            str(directory / f"vtlp{ratio:.2f}_{clip.name}"), 16000, shifted)
        written += 1

    print(f"  Added {written} pitch/formant-shifted copies of {len(clips)} synthetic clips")
    if skipped_unknown:
        print(f"  WARNING: {skipped_unknown} clip(s) skipped - voice sex unknown "
              f"(piper_pu_*). The child-range lever does not cover them; see "
              f"PIPER_VOICE_SEX in corpus/piper.py.")
    return written


def trim_silence(data: np.ndarray, sr: int = 16000, top_db: float = 40.0,
                 pad_ms: float = 30.0, frame_ms: float = 10.0) -> np.ndarray:
    """
    Trim leading and trailing silence using short-time RMS energy.

    OpenWakeWord's create_fixed_size_clip (openwakeword/data.py:719) aligns the END
    OF THE ARRAY with the end of the fixed-size window, not the end of the speech:

        start = max(0, n_samples - (len(x) + end_jitter))

    Trailing silence therefore pushes the phrase earlier in the window than the
    alignment the model actually sees when streaming detection fires. Leading
    silence matters here too: recordings from record_samples.py are a fixed 2s
    buffer with the phrase somewhere inside it, so untrimmed they fill the window
    and land at a completely different offset than the tight Kokoro clips.
    """
    if data.size == 0:
        return data

    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(data) // frame
    if n_frames < 2:
        return data

    frames = data[:n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return data

    voiced = np.flatnonzero(rms > peak * (10 ** (-top_db / 20)))
    if voiced.size == 0:
        return data

    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(data), (voiced[-1] + 1) * frame + pad)

    # Never hand back a clip too short to contain a wake word: if the energy
    # detection produced something implausible, keep the original.
    if end - start < int(sr * 0.2):
        return data

    return data[start:end]


def trim_directory(directory: Path, desc: str):
    """Trim silence from every WAV in a directory, in place."""
    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        return 0, 0.0

    removed_ms = []
    for wav_file in tqdm(wavs, desc=desc):
        try:
            sr, data = scipy.io.wavfile.read(wav_file)
            if data.ndim > 1:
                data = data[:, 0]
            trimmed = trim_silence(data, sr)
            if len(trimmed) < len(data):
                removed_ms.append((len(data) - len(trimmed)) / sr * 1000)
                scipy.io.wavfile.write(str(wav_file), sr, trimmed.astype(np.int16))
        except Exception as e:
            print(f"  Error trimming {wav_file.name}: {e}")

    return len(removed_ms), float(np.mean(removed_ms)) if removed_ms else 0.0
