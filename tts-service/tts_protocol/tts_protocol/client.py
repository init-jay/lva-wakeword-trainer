"""The TTS protocol client: a tcp:// server (server.py) as an Engine.

    tts_protocol.TtsClient("tcp://127.0.0.1:8899")

The client is deliberately thinner than any engine: it has NO knowledge of what
the server hosts. Voice catalogs, word timestamps, rate conversion, and the two
failure conventions all come back in the envelope (wire.py), and the only
thing the client adds on top is the batch algorithm it shares with the engines
(Engine.batch - join a group into one render, split on word timestamps). That
is the point of the protocol: a new machine runs a known server in front of
whatever model it has, and every caller here keeps working by pointing at the
port.

    TtsClient("tcp://a:8899")                     # one server
    [TtsClient(u) for u in "tcp://a:8899,tcp://b:8900".split(",")]   # a pool

One exchange per connection (wire.py): a new socket per request, so a dead
server is seen at connect time, not mid-batch, and there is no connection pool
to keep honest. Connect cost against loopback or the Docker bridge is
sub-millisecond next to a render.

The spec is `tcp://host:port` and nothing else. The old raw forms (an
OpenAI-compatible URL, a bare host:port meaning Wyoming) are deliberately
rejected: they used to mean different backends with different audio, and a
silent misread would render a corpus from the wrong engine.

Failure channels, inherited from the engines (engine.py): a null clip in the
response renders as (None, None) - the KOKORO convention, a transient miss the
corpus generator retries alone; an {"ok": false} envelope RAISES
TtsProtocolError - the PIPER convention, so the caller can report which
(voice, speaker) failed.
"""
import socket
import threading

from .engine import Engine
from .wire import (MAX_LINE, b64_decode, decode_line, encode_msg, wav_audio)

# The lock guards the catalog cache only: it is filled once, by the first
# thread, instead of racing N probe threads into N catalog fetches at the top
# of a corpus run.
_cache_lock = threading.Lock()


class TtsProtocolError(RuntimeError):
    """The server answered, and it is reporting a failure (wire.py)."""


def _parse_spec(spec: str):
    spec = spec.strip()
    if not spec.startswith("tcp://"):
        raise ValueError(
            f"unsupported TTS spec {spec!r} - the protocol client speaks "
            f"tcp://host:port (a tts-protocol server), not a raw engine URL")
    host, _, port = spec[len("tcp://"):].rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"unsupported TTS spec {spec!r} - expected tcp://host:port")
    return host, int(port)


class TtsClient(Engine):
    """One tts-protocol server. Conforms to Engine so the corpus code the
    trainers and eval share drives it exactly like an in-process engine:
    .voices() / .timed_render() / .batch(), with the same two failure
    conventions."""

    name = "tts"

    # Set from the server's declared capabilities on the first catalog call
    # (wire.py). Until then both are the safe defaults: no timestamps (batch
    # degrades to a loop), plain voice names.
    supports_timestamps = False
    speaker_voices = False

    def __init__(self, spec: str, connect_timeout: float = 10.0,
                 timeout: float = 600.0):
        self.host, self.port = _parse_spec(spec)
        self.spec = f"tcp://{self.host}:{self.port}"
        self.connect_timeout = connect_timeout
        # A joined batch of long texts can legitimately take a while; 600 s is
        # above the longest measured batch by an order of magnitude and still
        # turns a wedged server into an error instead of a hang.
        self.timeout = timeout
        self._catalog = None
        # The engine name the server declares in its catalog answer (wire.py):
        # what the probe prints, so a URL pointed at the wrong server (a Piper
        # port, a stale process) says so at the top of a run.
        self.server_engine = None

    def _request(self, payload: dict) -> dict:
        try:
            sock = socket.create_connection((self.host, self.port),
                                            timeout=self.connect_timeout)
        except OSError as e:
            raise TtsProtocolError(
                f"{self.spec} unreachable: {type(e).__name__}: {e}") from None
        try:
            sock.settimeout(self.timeout)
            sock.sendall(encode_msg(payload))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    raise TtsProtocolError(f"{self.spec} closed mid-response")
                buf += chunk
                if len(buf) > MAX_LINE:
                    raise TtsProtocolError(f"{self.spec} response over {MAX_LINE} B")
            msg = decode_line(buf)
        finally:
            sock.close()
        if not msg.get("ok"):
            raise TtsProtocolError(f"{self.spec}: {msg.get('error')}")
        return msg

    def available(self):
        """Never raises. A server is usable if it answers a catalog request."""
        try:
            voices = self.voices()
            return True, f"{len(voices)} voices"
        except Exception as e:
            return False, str(e)

    def voices(self, **kwargs):
        """The catalog, cached. (voice, speaker) pairs when the server declares
        speaker voices, plain names otherwise - the same shapes the corpus
        layer has always worked in."""
        if self._catalog is None:
            with _cache_lock:
                if self._catalog is None:
                    payload = {"op": "voices"}
                    if kwargs.get("max_speakers"):
                        payload["max_speakers"] = int(kwargs["max_speakers"])
                    if kwargs.get("languages"):
                        payload["languages"] = [str(l) for l in kwargs["languages"]]
                    msg = self._request(payload)
                    self.server_engine = msg.get("engine")
                    self.supports_timestamps = bool(msg.get("timestamps"))
                    self.speaker_voices = bool(msg.get("speaker"))
                    raw = msg["voices"]
                    self._catalog = [tuple(v) if isinstance(v, list) else v
                                     for v in raw]
        return self._catalog

    def timed_render(self, voice, text: str, speed: float = 1.0):
        voice = list(voice) if isinstance(voice, tuple) else voice
        msg = self._request({"op": "render", "voice": voice,
                             "speed": float(speed), "text": text,
                             "timestamps": True})
        if msg.get("clip") is None:
            return None, None
        return wav_audio(b64_decode(msg["clip"])), msg.get("timestamps")

    def batch(self, voice, speed: float, texts: list):
        # Inherited from Engine: join, one render, split on word timestamps -
        # or a per-clip loop when the server declared no timestamps (Piper).
        # See Engine.batch for the measurement this exists for.
        return Engine.batch(self, voice, speed, texts)
