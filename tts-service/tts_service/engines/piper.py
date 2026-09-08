"""Piper TTS over the Wyoming protocol: JSONL-over-TCP.

Moved verbatim from train/corpus/piper.py (2026-09-08) - the protocol functions and
its speed handling. What did NOT move: the wake-word policy tables
(MISPRONOUNCING_PIPER_VOICES, UNAUDITED_PIPER_VOICES, PIPER_VOICE_SEX), voice_sex()
and select_piper_voices() - those are training policy, not engine knowledge, and
they stayed in train/corpus/piper.py, which re-exports this module's functions.

THE SPEED PROBLEM. Wyoming's `synthesize` event has no rate control, so speed has to
be applied after synthesis. audit_voices.py:191 does it by resampling, which moves
pitch as well as rate - correct there (the point is to deny the ASR a comfortable
rendering), wrong here. `PLAIN_SPEEDS = (0.7, 1.6)` in train.py means DELIVERY RATE:
Kokoro's speed parameter re-times the phrase without a chipmunk, and the model is
meant to learn that the phrase can be said quickly, not by someone with a shorter
vocal tract. Pitch is already covered, separately, by add_child_range_copies. So
speed here goes through `time_stretch` (WSOLA, pitch-preserving) instead; using
resampling would silently entangle the speed sweep with the child-range lever and
make run 13's result impossible to attribute.

PIPER IS STOCHASTIC. VITS samples noise and durations per call, so the same
(voice, speaker, text, speed) renders differently every time - measured in
audit_voices.py, where one speaker scored 0% then 100% across passes. For a corpus
that is free diversity; for the audit it is why `--repeats` exists. It also means
the corpus is not reproducible from a seed, which is already true of the Kokoro
path (train.py sets no seed).

EXERCISED AGAINST A LIVE SERVICE. piper_voices enumerated 2005 (voice, speaker) pairs
across en_US/en_GB, and piper_render returns 16 kHz int16 at exact speed ratios
(0.7000, 1.2501, 1.6001 measured on one clip).

MEASURE SPEED ON ONE CLIP, NOT ACROSS CALLS. Three separate renderings at
1.0/1.6/0.7 gave 0.964 s / 0.501 s / 1.194 s - looks wrong, is not: VITS samples
durations per call, so the base clip differs every time. Stretch ratios are only
meaningful against a single rendering.

ONE SERVER SERVES ONE REQUEST AT A TIME. wyoming-piper holds exactly one loaded
voice in a module-level global and reloads it whenever a request names a different
one (handler.py:333-346, `if voice_name != _VOICE_NAME`). That is why this engine
is batch_mode="serial" - order is a correctness property (a reloaded voice mid-batch
costs a fresh InferenceSession), and why the corpus generator keeps voice as the
outer loop. Parallelism comes from separate instances, each with its own voice -
which means sharding a multi-voice corpus BY VOICE across instances, never
round-robin (see client.py, which deliberately does not do that for serial engines).
"""

import json
import socket
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..audio import SR, time_stretch
from ..engine import Engine

# The Wyoming JSONL-over-TCP framing: one JSON header line per event, then
# `data_length` bytes of JSON and `payload_length` bytes of audio.
#
# Duplicated from audit_voices.py rather than shared. That script is standalone by
# design - it runs on the host against live TTS and ASR services and takes no
# dependency on this package - and collapsing the two would drag the corpus layer
# into it. Worth revisiting if a third caller appears.


def _send(sock, etype, data=None, payload=None):
    header = {"type": etype}
    if data is not None:
        header["data"] = data
    if payload is not None:
        header["payload_length"] = len(payload)
    sock.sendall((json.dumps(header) + "\n").encode())
    if payload is not None:
        sock.sendall(payload)


def _read_event(sock, buf):
    """Next event, plus the remaining buffer and this event's audio payload."""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf, b""
        buf += chunk
    line, _, buf = buf.partition(b"\n")
    header = json.loads(line)
    n = header.get("data_length") or 0
    while len(buf) < n:
        buf += sock.recv(65536)
    data = json.loads(buf[:n]) if n else header.get("data", {})
    buf = buf[n:]
    p = header.get("payload_length") or 0
    while len(buf) < p:
        buf += sock.recv(65536)
    return {"type": header.get("type"), "data": data}, buf[p:], buf[:p]


def piper_voices(host, port, languages=("en_US", "en_GB"), max_speakers=0):
    """[(voice, speaker_or_None), ...] for the requested languages.

    `max_speakers` caps how many speakers of a multi-speaker model are sampled -
    en_US-libritts_r-medium alone carries 904, and taking all of them would swamp
    the corpus with one model's phonemisation. The sample is evenly spaced rather
    than the first N, because speaker ids are ordered by the source corpus and the
    head of that list is not representative.
    """
    sock = socket.create_connection((host, port), timeout=30)
    try:
        sock.settimeout(30)
        _send(sock, "describe")
        event, _, _ = _read_event(sock, b"")
        info = event["data"] if event else {}
    finally:
        sock.close()

    out = []
    for program in info.get("tts", []):
        for voice in program.get("voices", []):
            langs = voice.get("languages") or [voice.get("language")]
            if languages and not any(str(l).startswith(tuple(languages)) for l in langs):
                continue
            speakers = [s.get("name") for s in (voice.get("speakers") or [])]
            if not speakers:
                out.append((voice["name"], None))
                continue
            if max_speakers and len(speakers) > max_speakers:
                idx = np.linspace(0, len(speakers) - 1, max_speakers).astype(int)
                speakers = [speakers[i] for i in sorted(set(idx))]
            out.extend((voice["name"], s) for s in speakers)
    return out


def piper_render(host, port, voice, speaker, text, speed=1.0):
    """Synthesize one phrase, returned as 16 kHz mono int16.

    `speed` > 1 is faster. Applied with time_stretch after resampling to 16 kHz, so
    it changes delivery rate without moving pitch - see the module docstring.

    Raises on transport failure (the caller reports which voice failed) - unlike
    the kokoro engines, which return None.
    """
    sock = socket.create_connection((host, port), timeout=120)
    try:
        sock.settimeout(120)
        v = {"name": voice}
        if speaker is not None:
            v["speaker"] = str(speaker)
        _send(sock, "synthesize", {"text": text, "voice": v})
        buf, pcm, rate = b"", b"", 22050
        while True:
            event, buf, payload = _read_event(sock, buf)
            if event is None:
                break
            if event["type"] in ("audio-start", "audio-chunk"):
                rate = event["data"].get("rate", rate)
                pcm += payload
            elif event["type"] == "audio-stop":
                break
    finally:
        sock.close()

    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)

    if rate != SR:
        # resample_poly rather than resample: rational up/down, no FFT-length
        # sensitivity, and it is what vocal_tract_shift already uses.
        frac = Fraction(SR, int(rate)).limit_denominator(1000)
        audio = resample_poly(audio.astype(np.float64), frac.numerator, frac.denominator)

    if abs(speed - 1.0) > 1e-3:
        # speed 1.6 = 1.6x faster = 1/1.6 the duration.
        audio = time_stretch(np.asarray(audio, dtype=np.float64), 1.0 / float(speed), sr=SR)

    return np.clip(audio, -32768, 32767).astype(np.int16)


class PiperWyomingEngine(Engine):
    """One wyoming-piper server as a registered engine: name is "piper".

    Voices are (name, speaker_or_None) pairs, the unit the corpus layer and the
    exclusion tables work in - a bare name is accepted and read as speaker None.
    """

    name = "piper"
    supports_timestamps = False
    batch_mode = "serial"

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)

    def available(self):
        try:
            piper_voices(self.host, self.port)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def voices(self, languages=("en_US", "en_GB"), max_speakers=0):
        return piper_voices(self.host, self.port, languages=languages,
                            max_speakers=max_speakers)

    def timed_render(self, voice, text: str, speed: float = 1.0):
        if isinstance(voice, tuple):
            name, speaker = voice
        else:
            name, speaker = voice, None
        return piper_render(self.host, self.port, name, speaker, text, speed), None
