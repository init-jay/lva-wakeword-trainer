"""The TTS wire protocol: one JSON line per message, one exchange per connection.

THIS IS THE CONTRACT. A machine joins the TTS side of this repo by running a
server that answers these two ops; it joins the training/eval side by speaking
them as a client. Nothing else crosses the wire.

Why this shape:

  * ONE LINE each way, so framing is trivial - a request is written whole and a
    response is read up to the newline. There is no partial-header state machine
    to get wrong, which is the whole failure surface of HTTP and of length-
    prefixed binary protocols.
  * ONE EXCHANGE PER CONNECTION, so neither side has to decide what a dropped
    connection means mid-batch: the client retries the whole render (the server
    holds no per-connection state - the engine is shared, and engine calls are
    serial under a lock anyway, server.py).
  * AUDIO AS BASE64 WAV INSIDE THE JSON, not a binary frame: a clip is tens of
    KB to a couple of MB raw - small enough that the 33% encoding cost and the
    one-line limit are noise next to the seconds the model spends. Text and
    audio never have to be cut at the same byte.
  * RENDER CARRIES ONE TEXT, NOT A LIST. Batching is the CLIENT's job: it joins
    a group of texts into one long utterance, sends it, and splits the answer
    on the word timestamps (engine.Engine.batch). That is exactly what the old
    HTTP path did - the Kokoro server was always handed one long string and
    gave back one audio - so a server implementing this protocol has no notion
    of batching to get wrong. A host that cannot timestamp (Piper) cannot be
    split, and the client simply sends the group one text at a time.

Request / response shapes (both sides build and check these):

    {"op": "voices", "max_speakers": N, "languages": ["en_US", "en_GB"]}
      -> {"ok": true, "voices": [...], "timestamps": <bool>, "speaker": <bool>}
      -> {"ok": false, "error": "..."}

    "voices" is the catalog: plain names for a Kokoro host, [name, speaker]
    pairs for a Piper host - the "speaker" flag is how the client tells which.
    "timestamps" is how the client learns a render will carry word times; the
    corpus probe used to pay a test synthesis to find that out, and a declared
    capability is what a server knows about itself. The two request fields are
    optional and engine-specific: Piper honours both, Kokoro ignores both.

    {"op": "render", "voice": "af_bella" | ["en_US-lessac-medium", "12"],
     "speed": 1.0, "text": "hey seeree", "timestamps": true}
      -> {"ok": true, "clip": "<wav b64>" | null,
          "timestamps": [{"word","start_time","end_time"}, ...] | null}
      -> {"ok": false, "error": "..."}

Audio crosses the wire as 16 kHz mono int16 - the corpus rate, asserted in
wav_audio below. Rate conversion and rate control happen on the server, where
the engine lives (the Piper host resamples 22050 -> 16000 and applies speed
with pitch-preserving WSOLA; the Kokoro host resamples 24000 -> 16000 - see
the engine adapters), so a client never has to know what its server emits.

The two failure channels are the engines' inherited conventions (engine.py):
a null clip is the KOKORO convention - the engine rendered nothing but is
alive (transient TTS error; the caller retries that one clip alone); an
`"ok": false` envelope is the PIPER convention - the engine or the transport
is the problem and the client raises it so the caller can report WHICH voice.
"""
import base64
import io
import json

import numpy as np
import scipy.io.wavfile

# A 512 MB line is ~12,000 hours of 16 kHz mono. Anything bigger is a bug, not
# a batch, and reading it would pin memory for nothing.
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
    """wav file bytes -> (sr, int16 array). The protocol speaks 16 kHz only, so
    the sr check is a protocol assertion, not a resample."""
    sr, data = scipy.io.wavfile.read(io.BytesIO(payload))
    if sr != 16000:
        raise ValueError(f"server returned {sr} Hz audio; 16000 expected")
    return data


def b64_encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def b64_decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))
