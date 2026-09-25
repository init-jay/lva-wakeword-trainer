"""Audio conventions shared by every engine, plus the one DSP helper they need.

Everything in this layer speaks 16 kHz mono int16, because that is the one shape
both trainers and the eval harness consume - the same convention the corpus layer
always used. Engines produce their native rate and convert at the edge:

  * kokoro renders at 24 kHz and resamples down (engines/kokoro_http.py,
    engines/kokoro_mlx.py - the latter asks the model for 16 kHz directly)
  * piper renders at 22.05 kHz and resamples down (engines/piper.py)

`time_stretch` moved here from train/corpus/augment.py (2026-09-08), because the
piper engine applies speed with it and this package must not import train/.
train/corpus/augment.py re-exports it, so every existing import keeps working.
"""

# 16 kHz because that is what the corpus is: every clip in the pipeline is written
# and read at 16000.
SR = 16000


def to_int16(audio) -> "np.ndarray":
    """float32 in [-1, 1] -> int16, matching what the HTTP path returns.

    Moved verbatim from the old in-repo MLX adapter (now
tts-service/engines/kokoro_mlx/). The server sends a WAV that scipy
    reads as int16 already; MLX hands back floats, so the scaling happens here.
    Clipped rather than normalised: normalising would make each clip's gain depend
    on its own peak, which is a per-clip volume difference the model could learn
    instead of the phrase.
    """
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    return np.clip(a * 32767.0, -32768, 32767).astype(np.int16)


def time_stretch(x, factor: float, sr: int = 16000,
                 frame_ms: float = 30.0, seek_ms: float = 7.0):
    """Lengthen `x` by `factor` without moving pitch (WSOLA overlap-add).

    Moved verbatim from train/corpus/augment.py.

    Plain overlap-add at a fixed hop cuts frames at arbitrary phase and the
    reassembled periods fight each other, which on a voiced phrase sounds like
    added roughness. WSOLA slides each analysis frame within +/-`seek_ms` to the
    offset that best correlates with what naturally followed the previous frame,
    so consecutive frames stay in phase.

    scipy only, deliberately: the trainer image has no ffmpeg (Dockerfile:7) and
    torchaudio is not importable from the eval tools, so anything relying on either
    could not be checked outside the container.
    """
    import numpy as np

    if abs(factor - 1.0) < 1e-3 or len(x) < int(sr * frame_ms / 1000) * 2:
        return x.astype(np.float64)

    x = x.astype(np.float64)
    N = int(sr * frame_ms / 1000)
    hop_in = N // 4
    hop_out = max(1, int(round(hop_in * factor)))
    seek = int(sr * seek_ms / 1000)
    win = np.hanning(N + 1)[:N]

    out = np.zeros(int(len(x) * factor) + 2 * N)
    weight = np.zeros_like(out)
    tail = None
    i = 0
    while True:
        want = i * hop_in
        offset = 0
        if tail is not None:
            lo, hi = max(0, want - seek), min(len(x) - N, want + seek)
            if hi > lo:
                seg = x[lo:hi + len(tail)]
                if len(seg) >= len(tail):
                    offset = lo + int(np.argmax(np.correlate(seg, tail, "valid"))) - want
        start = want + offset
        dest = i * hop_out
        if start < 0 or start + N > len(x) or dest + N > len(out):
            break
        out[dest:dest + N] += x[start:start + N] * win
        weight[dest:dest + N] += win
        nxt = start + hop_out
        tail = x[nxt:nxt + N // 2] if nxt + N // 2 <= len(x) else None
        i += 1

    covered = weight > 1e-6
    out[covered] /= weight[covered]
    return out[:int(len(x) * factor)]


def phrase_end_sample(timestamps, wake_word: str, sr: int = 16000):
    """Sample index where the wake word ends, or None if the words do not line up.

    Verified rather than assumed: the timestamps are matched against the words of
    the wake phrase before their times are used. A mismatch (different
    tokenisation, a normalisation rule splitting a word) would otherwise cut at
    the wrong place silently, and a wrong cut here is what broke the alignment
    last time.

    The timestamps are the protocol's word-time dicts (wire.py) - the shape the
    Kokoro servers return and the one the mlx fork normalises to. The cut point
    is computed from them on the CLIENT side, which is why this function lives
    here rather than in any engine: nothing engine-specific about it.
    """
    if not timestamps:
        return None

    strip = str.maketrans("", "", ".,!?;:\"'")
    expected = [w.translate(strip).lower() for w in wake_word.split()]
    got = [str(t.get("word", "")).translate(strip).lower()
           for t in timestamps[:len(expected)]]
    if got != expected:
        return None

    end = timestamps[len(expected) - 1].get("end_time")
    return int(end * sr) if end else None
