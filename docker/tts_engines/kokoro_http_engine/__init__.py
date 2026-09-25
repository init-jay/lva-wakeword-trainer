"""Kokoro over HTTP (Kokoro-FastAPI): the Docker route, CPU and CUDA alike.

The Kokoro HTTP protocol adapter since the tts-service split (2026-09-08): it
fronts a running Kokoro-FastAPI process and exposes it on the repo's TTS
protocol port. It lives in the docker/ build context (not in src/tts-service/)
because it exists only where a Kokoro-FastAPI image runs: the compose service
runs the FastAPI server and this wrapper in one container -

    python -m kokoro_http_engine --url http://127.0.0.1:8880 --port 8899

(docker/Dockerfile.kokoro, whose CMD starts both). On a Mac this module is
never run at all: the mlx engine is a separate in-process server
(tts-service/engines/kokoro_mlx/), and that is why there is no `engines/kokoro`
project next to it. The URL-facing functions below are what src/train/corpus/
kokoro.py used to be before it became a thin re-export of the protocol
client.
"""
import base64
import io
import sys
import warnings
from fractions import Fraction

import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import resample_poly

from tts_protocol.engine import Engine, split_joined

# Same filter the old module set: the TTS reads below occasionally hit a partial
# read and urllib3 warns, once per clip.
warnings.filterwarnings("ignore", message="Reached EOF prematurely")
# Streaming servers emit a placeholder RIFF length; the data itself is fine. The
# eval generators used to swallow this around their own read - same class of event,
# one place.
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


def _wav_to_16k_int16(content: bytes) -> np.ndarray:
    """WAV bytes from the server -> 16 kHz mono int16.

    The shared tail of kokoro_tts and kokoro_tts_timed. resample_poly rather than
    scipy.signal.resample (the old FFT path here): rational up/down with no
    FFT-length sensitivity - the same choice the piper path made for the same
    reason (engines/piper.py). Kokoro is mono, so the first-channel downmix the
    old code did is unchanged in effect; the eval scripts used to average
    channels, which on mono audio is the same samples.
    """
    sr, data = wavfile.read(io.BytesIO(content))
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        frac = Fraction(16000, int(sr)).limit_denominator(1000)
        data = resample_poly(data.astype(np.float64), frac.numerator, frac.denominator)
    return np.clip(data, -32768, 32767).astype(np.int16)


def get_kokoro_voices(kokoro_url: str) -> list:
    """Get all available English voices from Kokoro."""
    try:
        r = requests.get(f"{kokoro_url}/v1/audio/voices", timeout=5)
        voices = r.json().get("voices", [])
        # Filter to English voices (a = American, b = British)
        voices = [v["id"] if isinstance(v, dict) else v for v in voices]
        english = [v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))]
        print(f"Kokoro voices available: {len(english)}")
        return english
    except Exception as e:
        print(f"ERROR: Cannot connect to Kokoro at {kokoro_url}: {e}")
        print("Make sure the Kokoro-FastAPI process it fronts is up - the host uv")
        print("venv (scripts/start-kokoro-host.sh) or the in-image server (docker)")
        print("- and that this wrapper was pointed at it with --url.")
        sys.exit(1)


def kokoro_tts(kokoro_url: str, voice: str, text: str, speed: float):
    """Render one utterance as 16 kHz int16 audio, or None on failure."""
    try:
        r = requests.post(
            f"{kokoro_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed
            },
            timeout=30
        )
        if r.status_code != 200:
            return None
        return _wav_to_16k_int16(r.content)
    except Exception:
        return None


def kokoro_tts_timed(kokoro_url: str, voice: str, text: str, speed: float):
    """Render an utterance and return (16 kHz int16 audio, word timestamps).

    /dev/captioned_speech returns per-word start/end times alongside the audio, which
    is what makes an exact cut possible for the run-on positives: it says precisely
    where the wake word ends inside the utterance, instead of that having to be
    inferred from a separate phrase-alone rendering.

    Returns (None, None) if the endpoint is unavailable, so callers can fall back.
    """
    try:
        r = requests.post(
            f"{kokoro_url}/dev/captioned_speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed,
                "stream": False,
                "return_timestamps": True,
            },
            timeout=60
        )
        if r.status_code != 200:
            return None, None

        payload = r.json()
        return _wav_to_16k_int16(base64.b64decode(payload["audio"])), payload.get("timestamps")
    except Exception:
        return None, None


def kokoro_tts_batch(kokoro_url: str, voice: str, texts: list, speed: float):
    """Render several utterances in ONE request and split them apart.

    Measured against a Kokoro-FastAPI server: a single "hey seeree" request costs
    ~119 ms of fixed overhead plus ~42 ms per second of audio, so for a phrase under
    a second, THREE QUARTERS of the request is overhead. Batching amortises it.

    In isolation a batch of 16-32 reaches ~37 ms/clip against 182 individually (5x),
    but real batches are smaller: buckets hold `samples_per_voice / len(grid)` clips,
    about 9.5 for plain positives at the defaults. Measured end to end at that shape
    it is 137 -> 42 ms/clip, a 3.3x speedup. A coarser speed grid would enlarge the
    buckets but measured no faster (39 ms/clip at 0.10 steps), so the finer grid is
    kept for the extra speed diversity.

    Every text in a batch shares one voice and one speed - that is what makes it a
    single forward pass - so callers must group by (voice, speed) before calling.

    The split and the re-basing live in split_joined (engine.py); this is the URL
    entry point the run-on generator in src/train/oww/train.py still calls directly.
    """
    if not texts:
        return []
    joined = ". ".join(t.rstrip(".") for t in texts) + "."
    data, timestamps = kokoro_tts_timed(kokoro_url, voice, joined, speed)
    if data is None or not timestamps:
        return [(None, None)] * len(texts)
    return split_joined(data, timestamps, texts)


class KokoroHttpEngine(Engine):
    """One Kokoro-FastAPI server as a registered engine: name is "kokoro-http"."""

    name = "kokoro-http"
    supports_timestamps = True
    batch_mode = "batch"

    def __init__(self, url: str):
        self.url = url.strip().rstrip("/")

    def available(self):
        try:
            r = requests.get(f"{self.url}/v1/audio/voices", timeout=10)
            r.raise_for_status()
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def voices(self, **kwargs):
        # The optional catalog arguments (languages/max_speakers) are Piper's;
        # this catalog has no options, so they are ignored, not rejected.
        return get_kokoro_voices(self.url)

    def timed_render(self, voice, text: str, speed: float = 1.0):
        return kokoro_tts_timed(self.url, voice, text, speed)

    def render(self, voice, text: str, speed: float = 1.0):
        return kokoro_tts(self.url, voice, text, speed)

    def batch(self, voice, speed: float, texts: list):
        # The URL functions take the url; the engine hands them ours.
        if not texts:
            return []
        return kokoro_tts_batch(self.url, voice, texts, speed)
