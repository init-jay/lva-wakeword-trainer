"""Kokoro TTS in-process on Apple Silicon, via MLX.

THE SAME MODEL AS THE KOKORO SERVICE, A DIFFERENT RUNTIME. docker-compose runs
Kokoro-82M behind Kokoro-FastAPI and train.py talks to it over HTTP; this renders the
same model through MLX in the training process itself. Measured on an M1 Max at
16 kHz, against that server on its fastest configuration (host CPU, batched):

                     Kokoro-FastAPI      MLX single    MLX batch of 10
    plain                  88 ms/clip      57 ms/clip       25 ms/clip
    run-on                229 ms/clip      89 ms/clip       63 ms/clip

Single-clip MLX beats the server's BATCHED path, which is the interesting part: the
simplest possible integration - render one clip at a time, no joining, no splitting -
is already ~1.7x on the corpus stage.

WORD TIMESTAMPS COST +0.2 ms/clip, so they are always requested. They are not a
luxury: run-on positives are cut just after the wake word, and that boundary comes
from per-word times. The mechanism is exact rather than heuristic - Kokoro predicts a
duration for every phoneme and the audio is rendered from precisely those durations,
so word boundaries are correct by construction. Verified against Kokoro-FastAPI on
the same phrase and voice: the cut after "seeree" agreed to 3 ms, against a
RUNON_TAIL_MS of 150-300 and the +153 ms error of the estimate-based fallback.

    upstream: https://github.com/gabrimatic/kokoro-mlx
    fork:     https://github.com/init-jay/kokoro-mlx  (adds return_timestamps)

WHY THE FORK. Upstream computes the durations and discards them; the fork returns
them. Nothing else differs.

TWO THINGS TO KNOW BEFORE USING THIS FOR A REAL CORPUS.

  * IT OFFERS FEWER VOICES: 28 English against the server's 42. Voice diversity is
    load-bearing here - the corpus is built per voice, and a narrower set means less
    speaker variation for the model to generalise from. Check the count the run
    prints rather than assuming.
  * ITS AUDIO IS NOT THE SERVER'S. bf16 weights from mlx-community/Kokoro-82M-bf16
    and misaki G2P, against the server's own stack. Timing agrees; pronunciation is
    unverified, and this repo already excludes voices that mispronounce the wake word
    (MISPRONOUNCING_VOICES), so G2P differences are not cosmetic. A corpus generated
    this way needs an eval against one that was not, not an assumption.

ALSO: it renders ~430 ms of leading silence, which the server does not. The
timestamps account for it, so cuts are unaffected, and corpus/augment.py's trimming
removes it later. Only code that mixes a pre-trim timestamp with post-trim audio
would be wrong, and nothing does that today.
"""
import sys
import threading

import numpy as np

# 16 kHz because that is what the corpus is: train.py writes every clip at 16000 and
# both trainers read it. Asking the model for it directly avoids a resample.
SAMPLE_RATE = 16000

# The URL scheme that selects this backend. train.py threads a Kokoro URL through
# every call site, so rather than restructure that, "mlx://" is a URL that happens to
# mean "in this process" - KokoroPool, generate_kokoro_samples and
# generate_runon_samples then need no changes at all.
URL_SCHEME = "mlx://"

_tts = None
_lock = threading.Lock()


def is_mlx_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith("mlx")


def available() -> tuple:
    """(usable, reason). Never raises - callers decide what to do about it."""
    if sys.platform != "darwin":
        return False, f"MLX is Apple Silicon only; this is {sys.platform}"
    try:
        import kokoro_mlx  # noqa: F401
    except ImportError as e:
        return False, (f"kokoro-mlx not installed ({e}). It lives in the "
                       "train-applesilicon/ host environment, not the trainer images.")
    return True, ""


def _get():
    """The model, loaded once.

    Double-checked under a lock because run_jobs calls this from a thread pool.
    KokoroTTS.generate is itself thread-safe (it holds its own lock), so the model is
    shared rather than one per thread - loading it per thread would cost ~330 MB each
    and buy nothing, since the lock serialises the work anyway.
    """
    global _tts
    if _tts is None:
        with _lock:
            if _tts is None:
                from kokoro_mlx import KokoroTTS
                _tts = KokoroTTS.from_pretrained()
    return _tts


def voices() -> list:
    """English voice ids, filtered the same way get_kokoro_voices filters the server's."""
    return sorted(v for v in _get().list_voices()
                  if str(v).startswith(("af_", "am_", "bf_", "bm_")))


def _to_int16(audio) -> np.ndarray:
    """float32 in [-1, 1] -> int16, matching what the HTTP path returns.

    The server sends a WAV that scipy reads as int16 already; MLX hands back floats,
    so the scaling happens here. Clipped rather than normalised: normalising would
    make each clip's gain depend on its own peak, which is a per-clip volume
    difference the model could learn instead of the phrase.
    """
    a = np.asarray(audio, dtype=np.float32)
    return np.clip(a * 32767.0, -32768, 32767).astype(np.int16)


def render(voice: str, text: str, speed: float):
    """16 kHz int16 audio, or None on failure. Mirrors kokoro_tts()."""
    try:
        res = _get().generate(text, voice=voice, speed=speed,
                              sample_rate=SAMPLE_RATE)
        return _to_int16(res.audio)
    except Exception:
        return None


def render_timed(voice: str, text: str, speed: float):
    """(16 kHz int16 audio, word timestamps), or (None, None). Mirrors kokoro_tts_timed().

    Timestamps are dicts with word / start_time / end_time in seconds - the same
    shape /dev/captioned_speech returns, which is what phrase_end_sample reads.
    """
    try:
        res = _get().generate(text, voice=voice, speed=speed,
                              sample_rate=SAMPLE_RATE, return_timestamps=True)
        ts = res.timestamps or None
        # The fork returns dicts; tolerate objects in case that changes, because
        # phrase_end_sample calls .get() on each entry.
        if ts and not isinstance(ts[0], dict):
            ts = [{"word": getattr(t, "word", ""),
                   "start_time": getattr(t, "start_time", None),
                   "end_time": getattr(t, "end_time", None)} for t in ts]
        return _to_int16(res.audio), ts
    except Exception:
        return None, None
