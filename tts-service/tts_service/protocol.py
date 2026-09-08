"""The wire format of the TTS service: one JSON line per message, one exchange
per connection.

Why this shape:

  * ONE LINE each way, so framing is trivial - a request is written whole and a
    response is read up to the newline. There is no partial-header state machine
    to get wrong, which is the whole failure surface of HTTP and of length-
    prefixed binary protocols.
  * ONE EXCHANGE PER CONNECTION, so neither side has to decide what a dropped
    connection means mid-batch: the client retries the whole render (the server
    holds no per-connection state - the engine is shared, and engine calls are
    serial under a lock anyway).
  * AUDIO AS BASE64 WAV INSIDE THE JSON, not a binary frame: a batch of 16 16 kHz
    int16 clips is ~1.6 MB raw, ~2.1 MB base64 - small enough that the 33%
    encoding cost and the one-line limit are noise next to the seconds the model
    spends. Text and audio never have to be cut at the same byte.

Request / response shapes (both sides build and check these):

    {"op": "voices", "max_speakers": 0}
      -> {"ok": true,  "voices": [...]}          # shape is engine-specific
      -> {"ok": false, "error": "..."}

    {"op": "render", "voice": "af_bella" | ["en_US-lessac-medium", 12],
     "speed": 1.0, "texts": ["hey seeree", ...], "timestamps": true}
      -> {"ok": true, "clips": ["<wav b64>" | null, ...],
          "timestamps": [[{"word","start_time","end_time"}, ...] | null, ...]}

The two failure channels are the engines' inherited conventions (engine.py):
a null entry in `clips` is the KOKORO convention - the engine rendered nothing
but is alive (transient TTS error; the caller retries that one clip alone); an
`"ok": false` envelope is the PIPER convention - the engine or the transport is
the problem and the client raises it so the caller can report WHICH voice.
"""
import base64
import io
import json

import numpy as np
import scipy.io.wavfile

# A 512 MB line is ~12,000 hours of 16 kHz mono. Anything bigger is a bug, not a
# batch, and reading it would pin memory for nothing.
MAX_LINE = 512 * 1024 * 1024


def encode_msg(msg: dict) -> bytes:
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode_line(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))


def wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """16 kHz int16 array -> wav file bytes, in memory."""
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    return buf.getvalue()


def wav_audio(payload: bytes) -> np.ndarray:
    """wav file bytes -> (sr, int16 array). The service speaks 16 kHz only, so
    the sr check is a protocol assertion, not a resample."""
    sr, data = scipy.io.wavfile.read(io.BytesIO(payload))
    if sr != 16000:
        raise ValueError(f"service returned {sr} Hz audio; 16000 expected")
    return data


def b64_encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def b64_decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))
