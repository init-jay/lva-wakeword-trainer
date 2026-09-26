"""Guards for trim_silence (src/train/corpus/augment.py).

Why it matters: BOTH frontends place a fixed-size window relative to the END
of the array, so trailing silence displaces the phrase in the window the
model is trained and scored with - untrimmed, a recording's phrase lands at
a different offset than the tight TTS clips and the model is taught the
wrong alignment (trim_silence docstring). The tests pin, on a synthetic
int16 clip, that leading and trailing silence come off with the 30 ms pad,
that the speech survives byte-for-byte, and that the degenerate inputs
(all-silent, too short) return the original rather than crash or emit a
runt.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from train.corpus.augment import trim_silence  # noqa: E402

SR = 16000
FRAME = int(SR * 10 / 1000)          # trim_silence's default frame_ms=10
PAD = int(SR * 30 / 1000)            # trim_silence's default pad_ms=30

# 4800 = 0.3 s, 11200 = 0.7 s.
TONE_START, TONE_END = 4800, 11200


def _clip_with_silence():
    # 0.3 s silence, 0.4 s of 440 Hz, 0.3 s silence. Cosine, deliberately:
    # it starts and ends at full amplitude, so the first/last voiced
    # samples are unambiguous (a sine would open on zero). At 10 ms frames
    # the tone occupies whole frames 30..69 and the surrounding frames are
    # pure zero - no energy can leak across a boundary to blur the cut.
    tone = (8000 * np.cos(2 * np.pi * 440 * np.arange(TONE_END - TONE_START) / SR)
            ).astype(np.int16)
    clip = np.zeros(TONE_END + 4800, dtype=np.int16)
    clip[TONE_START:TONE_END] = tone
    return clip


def test_leading_and_trailing_silence_removed_speech_kept():
    clip = _clip_with_silence()
    out = trim_silence(clip, SR)

    # Expected cut: first voiced frame minus one 30 ms pad, last voiced
    # frame plus one; one frame of slack per side absorbs any RMS rounding.
    exp_start = 30 * FRAME - PAD      # 4320
    exp_end = (69 + 1) * FRAME + PAD  # 11680
    assert abs(len(out) - (exp_end - exp_start)) <= FRAME
    # Both silences are gone: only the 30 ms pad on each side may survive
    # out of the 0.3 s of silence that framed the tone (one frame of slack
    # per side for RMS rounding).
    assert len(out) <= len(clip) - 2 * (4800 - PAD) + 2 * FRAME

    # The speech survives verbatim: the tone starts one 30 ms pad in from
    # the trimmed start, so its position inside the output recovers the
    # slice offset in clip coordinates (4800 - 480 = 4320). The output must
    # be exactly that unmodified slice - nothing resampled or re-encoded.
    voiced = np.flatnonzero(np.abs(out) > 100)
    assert voiced[0] == PAD
    start = TONE_START - voiced[0]
    assert start == 30 * FRAME - PAD
    assert np.array_equal(out, clip[start:start + len(out)])
    # ...and the tone's tail is complete, with at least the pad behind it.
    assert voiced[-1] + start == TONE_END - 1
    assert len(out) - 1 - voiced[-1] >= PAD - FRAME


def test_all_silent_clip_does_not_crash_and_comes_back_whole():
    # Silence has no voiced frame, so the function's silence definition
    # (peak RMS <= 0) must hand the clip back unmodified, not trim to
    # nothing - an empty "positive" would be a silent corpus gap.
    clip = np.zeros(16000, dtype=np.int16)
    out = trim_silence(clip, SR)
    assert len(out) == len(clip)
    assert np.array_equal(out, clip)


def test_clip_too_short_to_frame_comes_back_whole():
    clip = np.zeros(100, dtype=np.int16)
    out = trim_silence(clip, SR)
    assert np.array_equal(out, clip)


def test_empty_clip_comes_back_whole():
    clip = np.zeros(0, dtype=np.int16)
    assert trim_silence(clip, SR) is clip


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
