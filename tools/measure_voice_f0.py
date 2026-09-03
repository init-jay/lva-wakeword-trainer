#!/usr/bin/env python3
"""Median F0 per TTS voice, to fill in PIPER_VOICE_SEX without listening to 96 voices.

WHY THIS EXISTS. add_child_range_copies picks a pitch/formant ratio from the voice's
sex, and that lever is the largest single win in tuning.md - a 4-year-old went from
24% detection to 83% once the corpus stopped being adult-only (run 13). Kokoro
encodes sex in the voice id (af_/am_/bf_/bm_), so it was free there. Piper voice
names do not, and a voice with no entry gets NO child-range copy, so the lever
quietly loses reach in proportion to how much of the corpus is Piper.

Filling 96 entries in by ear is more listening than anyone will actually do, and the
failure mode of not doing it is silent. So measure it: sex here is only a proxy for
F0, and F0 is directly measurable from clips the audit already wrote.

    python measure_voice_f0.py voice_audit_piper/
    python measure_voice_f0.py voice_audit_piper/ --python   # paste-ready dict

ONLY 1.0x CLIPS ARE USED. audit_voices.py applies speed by RESAMPLING, which moves
pitch with it (see its piper_render docstring), so a 1.3x clip reads 1.3x high. The
filename filter below is what keeps that out, and it is the whole reason this is not
simply "average every clip for the voice".

THE SPLIT IS 185 Hz, AND THE 160-200 Hz BAND IS GENUINELY AMBIGUOUS. It matters less
than it looks: the two ratio ranges in CHILD_STRETCH nearly coincide there. At 177 Hz
the male range gives 204-230 Hz and the female range 212-239 Hz. Validate any new run
the way this one was validated - against the voices whose NAME states the answer
(hfc_male/hfc_female, northern_english_male, southern_english_female).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile

SR = 16000
SPLIT_HZ = 185.0

# Piper voice names are {lang}-{name}-{quality}; anything after the quality is the
# speaker. Splitting on the quality is what turns a filename back into the
# "voice:speaker" key that voice_sex() looks up.
QUALITIES = {"x_low", "low", "medium", "high"}


def estimate_f0(x, sr=SR, fmin=70.0, fmax=400.0, rms_floor=500.0, clarity=0.3):
    """Median F0 over voiced frames, by autocorrelation.

    numpy/scipy only, deliberately - the same constraint the rest of the corpus
    tooling runs under, so this works outside the trainer image.
    """
    x = np.asarray(x, dtype=np.float64)
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    pitches = []
    for i in range(0, max(0, len(x) - frame), hop):
        f = x[i:i + frame]
        if np.sqrt((f ** 2).mean()) < rms_floor:      # silence
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        lag = lo + int(np.argmax(seg))
        if ac[lag] / ac[0] < clarity:                 # unvoiced / no clear period
            continue
        pitches.append(sr / lag)
    return float(np.median(pitches)) if pitches else None


def split_name(stem):
    """en_US-l2arctic-medium-BWC -> ('en_US-l2arctic-medium', 'BWC')."""
    parts = stem.split("-")
    for i, p in enumerate(parts):
        if p in QUALITIES:
            return "-".join(parts[:i + 1]), "-".join(parts[i + 1:]) or None
    return stem, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips_dir", help="audit output, e.g. voice_audit_piper/")
    p.add_argument("--speed", default="1.0",
                   help="only clips rendered at this speed (default: %(default)s). "
                        "Do not widen it - the others are pitch-shifted.")
    p.add_argument("--split", type=float, default=SPLIT_HZ,
                   help="Hz below which a voice is called male (default: %(default)s)")
    p.add_argument("--python", action="store_true",
                   help="emit a paste-ready PIPER_VOICE_SEX body")
    args = p.parse_args()

    pattern = f"*_{args.speed}_*.wav"
    by_key = defaultdict(list)
    for wav in sorted(Path(args.clips_dir).glob(pattern)):
        stem = re.sub(rf"_{re.escape(args.speed)}_\d+$", "", wav.stem)
        sr, data = scipy.io.wavfile.read(wav)
        if data.ndim > 1:
            data = data[:, 0]
        hz = estimate_f0(data, sr)
        if hz:
            voice, speaker = split_name(stem)
            by_key[f"{voice}:{speaker}" if speaker else voice].append(hz)

    if not by_key:
        raise SystemExit(f"no clips matched {pattern} in {args.clips_dir}")

    rows = sorted((k, float(np.median(v))) for k, v in by_key.items())
    if args.python:
        for key, hz in rows:
            print(f'    "{key}": "{"m" if hz < args.split else "f"}",  # {hz:.0f} Hz')
        return

    ambiguous = [(k, hz) for k, hz in rows if 160 <= hz <= 200]
    print(f"{len(rows)} voices, split at {args.split:.0f} Hz\n")
    print(f"{'voice':<44}{'F0 Hz':>7}  sex")
    for key, hz in sorted(rows, key=lambda r: r[1]):
        print(f"{key:<44}{hz:>7.0f}  {'m' if hz < args.split else 'f'}")
    print(f"\n{sum(1 for _, hz in rows if hz < args.split)} male, "
          f"{sum(1 for _, hz in rows if hz >= args.split)} female")
    if ambiguous:
        print(f"{len(ambiguous)} in the ambiguous 160-200 Hz band - low stakes, the "
              f"two CHILD_STRETCH ranges nearly coincide there")
    print("\nSanity-check against voices whose NAME states the answer before trusting "
          "this\n(hfc_male/hfc_female, northern_english_male, southern_english_female).")
    print("Re-run with --python to get a paste-ready PIPER_VOICE_SEX body.")


if __name__ == "__main__":
    main()
