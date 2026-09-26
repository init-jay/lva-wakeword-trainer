"""Kokoro TTS in-process on Apple Silicon, via MLX.

THE SAME MODEL AS THE KOKORO SERVICE, A DIFFERENT RUNTIME. docker-compose runs
Kokoro-82M behind Kokoro-FastAPI and the trainer talks to it over HTTP; this
server renders the same model through MLX in its own process instead. Measured on an M1 Max at
16 kHz, against that server on its fastest configuration (host CPU, batched):

                     Kokoro-FastAPI      MLX single    MLX batch of 10
    plain                  88 ms/clip      57 ms/clip       25 ms/clip
    run-on                229 ms/clip      89 ms/clip       63 ms/clip

Single-clip MLX beats the server's BATCHED path, which is the interesting part: the
simplest possible integration - render one clip at a time, no joining, no splitting -
is already ~1.7x on the corpus stage. The batch column is the join-and-split path
in Engine.batch, applied to this backend's word timestamps; see tts-service/README.md
for the re-measurement.

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
    (src/wordlists/<word>.yaml, voices.kokoro.mispronouncing), so G2P differences are
    not cosmetic. A corpus generated
    this way needs an eval against one that was not, not an assumption.

ALSO: it renders ~430 ms of leading silence, which the server does not. The
timestamps account for it, so cuts are unaffected, and corpus/augment.py's trimming
removes it later. Only code that mixes a pre-trim timestamp with post-trim audio
would be wrong, and nothing does that today.

The in-process Kokoro server since the tts-service split (2026-09-08): its own uv
project and venv, Apple Silicon only. The trainer speaks its protocol port like any
other - it has no idea this one never leaves the Mac:

    uv run --project tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine \
        --port 8900
"""
import sys
import threading

import numpy as np

from tts_protocol.engine import Engine
from tts_protocol.audio import to_int16

# 16 kHz because that is what the corpus is: train.py writes every clip at 16000 and
# both trainers read it. Asking the model for it directly avoids a resample.
SAMPLE_RATE = 16000

_tts = None
_lock = threading.Lock()


def available() -> tuple:
    """(usable, reason). Never raises - callers decide what to do about it."""
    if sys.platform != "darwin":
        return False, f"MLX is Apple Silicon only; this is {sys.platform}"
    try:
        import kokoro_mlx  # noqa: F401
    except ImportError as e:
        return False, (f"kokoro-mlx not installed ({e}). It is a dependency of "
                       "this uv project: tts-service/engines/kokoro_mlx")
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


def render(voice: str, text: str, speed: float):
    """16 kHz int16 audio, or None on failure. Mirrors kokoro_tts()."""
    try:
        res = _get().generate(text, voice=voice, speed=speed,
                              sample_rate=SAMPLE_RATE)
        return to_int16(res.audio)
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
        return to_int16(res.audio), ts
    except Exception:
        return None, None


class KokoroMlxEngine(Engine):
    """The in-process backend as a protocol server: name is "kokoro-mlx"."""

    name = "kokoro-mlx"
    supports_timestamps = True
    batch_mode = "batch"

    def available(self):
        return available()

    def voices(self, **kwargs):
        # Loading the model here rather than lazily on the first clip, so a failure
        # lands before the run prints its plan - the same reason the HTTP path probes
        # the server up front instead of discovering it is down mid-corpus.
        english = voices()
        print(f"Kokoro voices available: {len(english)} (MLX, in-process)")
        if len(english) < 40:
            print(f"  NOTE: the HTTP service offers 42 English voices; this offers "
                  f"{len(english)}. Voice diversity is a corpus lever - see "
                  f"tts-service/engines/kokoro_mlx/.")
        return english

    def timed_render(self, voice, text: str, speed: float = 1.0):
        return render_timed(voice, text, speed)
