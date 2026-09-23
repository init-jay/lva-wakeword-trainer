"""Audio transforms applied to a corpus of WAVs: trimming, and child-range copies.

Moved verbatim from train.py. Both trainers need these for the same reasons:

- Trimming, because BOTH frontends place a fixed-size window relative to the end of
  the array, so trailing silence displaces the phrase and teaches a later alignment.
  The openWakeWord mechanism is documented in trim_silence below; microWakeWord's
  clip duration differs (1500 ms vs 2000 ms) but the failure mode does not.
- Child-range copies, because the corpus is otherwise adult-only. This was the
  largest single win in the tuning log (run 13: a 4-year-old 24% -> 83%).

NOTE ON PIPER CLIPS: add_child_range_copies reads the voice's sex from the filename
convention (kokoro_af_bella_<uuid> -> "af" -> female; Piper clips are named
piper_p{sex}_... so the same extraction works) - see the function docstring for the
unknown-sex case it skips.

time_stretch moved to tts-service on 2026-09-08 (tts_protocol/audio.py) because the
shared piper engine applies speed with it and that layer must not import train/;
this module re-exports it, so `from train.corpus.augment import time_stretch`
keeps working everywhere, and the note about scipy-only inside it now lives there.
(2026-09-09: the monolith became the protocol package; same module, new name.)
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
# Run 12 measured the second speaker - a 4-year-old - at 24% detection against 97%
# for the adult, and 34% on his OWN training clips. He was 26% of the real corpus,
# so this is not under-representation: his fundamental sits outside the range of
# almost everything the model has ever seen. Measured medians: the child 291 Hz,
# the adult female 269 Hz, the adult male 153 Hz, Kokoro am_adam 132 Hz,
# af_bella 227 Hz. openwakeword's own
# PitchShift is +/-3 semitones at p=0.25 against a 13.6-semitone gap, so it cannot
# close it - and it is the same resample-plus-stretch operation as this, just with
# a range a quarter the size (torch_pitch_shift/main.py:156-168).
#
# The ratios are per sex because one global range serves neither. A listening test
# on vtlp_demo/ (run 12): af_bella at 1.28 is the closest thing to the child in the set,
# while male voices "sound like teenagers up to R1.30 and useless above that
# (chipmunk)". Male voices therefore cover the 152-172 Hz gap between the two real
# speakers rather than reaching a child, which they cannot do without artefact -
# and training on an artefact teaches the artefact.
#
# f -> 272-306 Hz, straddling the child. m -> 152-172 Hz, the adult-male-to-child gap.
CHILD_STRETCH = {"f": (1.20, 1.35), "m": (1.15, 1.30)}

# These clips are ADDED to the corpus, not substituted into it. Substituting would
# thin out adult coverage in proportion, which is the trade run 10 warns about:
# real-clip density drives the result, so buying child coverage by spending adult
# coverage is not a win.
CHILD_STRETCH_FRACTION = 0.5


def vocal_tract_shift(data: np.ndarray, ratio: float, sr: int = 16000) -> np.ndarray:
    """Raise F0 and formants by `ratio`, keeping the clip's original duration.

    Resampling alone raises pitch and formants together - which is what a shorter
    vocal tract does, and why this reaches a child voice where a formant-corrected
    shift would not - but it also shortens the clip by the same factor. The stretch
    puts the duration back, so the only thing that changed is the speaker, not the
    delivery speed. Verified against `ffmpeg -af asetrate,aresample,atempo` on the
    vtlp_demo/ clips: same F0 to within the estimator's resolution, and this keeps
    the original length exactly where atempo drifts ~3%.
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

    Only synthetic clips are shifted, and the ratio comes from the voice's sex,
    which is why the voice is in the filename. Real recordings are left alone: the
    child needs no shifting, and the adult speaker is male, so shifting them reaches
    the teen range that ~15 Kokoro male voices already cover far more cheaply than
    160 clips of one speaker.

    Piper clips participate on equal terms, by carrying the sex in the same
    position: corpus/piper.py names them `piper_p{sex}_...` precisely so the
    extraction below needs no special case. A Piper voice whose sex has not been
    established is written `piper_pu_...` and falls out at the CHILD_STRETCH lookup
    rather than being shifted by a guessed ratio - shifting a male voice by the
    female range produces the artefact run 12 warned about, and training on an
    artefact teaches the artefact.

    This matters more than it looks. If Piper clips displace Kokoro ones without
    being shiftable, the child-range lever's COVERAGE shrinks in proportion, and the
    likeliest casualty is the 4-year-old that run 13 exists to detect.

    Copies are ADDED - see CHILD_STRETCH_FRACTION.
    """
    clips = [p for p in sorted(directory.glob("*.wav"))
             if p.name.startswith(("kokoro_", "runon_", "piper_"))]
    if not clips:
        return 0

    written = 0
    skipped_unknown = 0
    for clip in tqdm(clips, desc=desc, unit="clip"):
        # kokoro_{voice}_{uuid}.wav -> af_bella; the sex is the voice prefix's
        # second letter (af_/bf_ female, am_/bm_ male). Piper clips are named
        # piper_p{sex}_... so the same index lands on the same thing.
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

    # Never hand back a clip too short to contain a wake word - if the energy
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
