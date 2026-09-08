"""The tcp:// engine: a tts-service server (server.py) as an Engine.

    tts_service.engines_from_spec("tcp://127.0.0.1:8899")
    tts_service.engines_from_spec("tcp://a:8899,tcp://b:8898")   # a pool

The client is deliberately thinner than the HTTP engine: it has NO knowledge of
what the server hosts. Joining, splitting, voice catalogs, and the failure
conventions all happen on the server side (server.py forwards to whatever
engine it hosts), and this module only speaks the one-line protocol in
protocol.py. That is the point of the service - a new machine runs one known
binary and whatever model it has, and every existing caller keeps working by
pointing at the port.

One exchange per connection (protocol.py): a new socket per request, so a dead
server is seen at connect time, not mid-batch, and there is no connection
pool to keep honest. Connect cost against loopback or the Docker bridge is
sub-millisecond next to a render.

Failure channels, inherited from the engines (engine.py): a null clip in the
response renders as (None, None) - the KOKORO convention, a transient miss the
corpus generator retries alone; an {"ok": false} envelope RAISES TtsServiceError
- the PIPER convention, so the caller can report which (voice, speaker) failed.
"""
import socket
import threading

from ..engine import Engine
from ..protocol import (MAX_LINE, b64_decode, decode_line, encode_msg,
                        wav_audio)

# The lock guards the voices cache only: it is filled once, by the first
# thread, instead of racing N probe threads into N catalog fetches at the top
# of a corpus run.
_cache_lock = threading.Lock()


class TtsServiceError(RuntimeError):
    """The service answered, and it is reporting a failure (protocol.py)."""


class TtsServiceTcpEngine(Engine):
    name = "tts-service"

    # The service forwards the hosted engine's timestamps, so from this client's
    # point of view the batch is as real as it gets; the server degrades it to a
    # per-clip loop if the hosted engine has none (Piper).
    supports_timestamps = True
    batch_mode = "batch"

    def __init__(self, host: str, port: int, connect_timeout: float = 10.0,
                 timeout: float = 600.0):
        self.host = host
        self.port = int(port)
        self.connect_timeout = connect_timeout
        # A joined batch of long texts can legitimately take a while; 600 s is
        # above the longest measured batch by an order of magnitude and still
        # turns a wedged server into an error instead of a hang.
        self.timeout = timeout
        self.url = f"tcp://{host}:{self.port}"
        self._voices = None

    def _request(self, payload: dict) -> dict:
        try:
            sock = socket.create_connection((self.host, self.port),
                                            timeout=self.connect_timeout)
        except OSError as e:
            raise TtsServiceError(
                f"{self.url} unreachable: {type(e).__name__}: {e}") from None
        try:
            sock.settimeout(self.timeout)
            sock.sendall(encode_msg(payload))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    raise TtsServiceError(f"{self.url} closed mid-response")
                buf += chunk
                if len(buf) > MAX_LINE:
                    raise TtsServiceError(f"{self.url} response over {MAX_LINE} B")
            msg = decode_line(buf)
        finally:
            sock.close()
        if not msg.get("ok"):
            raise TtsServiceError(f"{self.url}: {msg.get('error')}")
        return msg

    def available(self):
        """Never raises. A service is usable if it answers a catalog request."""
        try:
            voices = self.voices()
            return True, f"{len(voices)} voices"
        except Exception as e:
            return False, str(e)

    def voices(self, **kwargs):
        if self._voices is None:
            with _cache_lock:
                if self._voices is None:
                    payload = {"op": "voices"}
                    if kwargs.get("max_speakers"):
                        payload["max_speakers"] = int(kwargs["max_speakers"])
                    self._voices = self._request(payload)["voices"]
        return self._voices

    def timed_render(self, voice, text: str, speed: float = 1.0):
        msg = self._request({"op": "render", "voice": voice,
                             "speed": float(speed), "texts": [text],
                             "timestamps": True})
        return self._results(msg)[0]

    def batch(self, voice, speed: float, texts: list):
        # One request carries the whole group; the server joins, renders, and
        # splits (or loops, for a timestampless host) - see server.py.
        msg = self._request({"op": "render", "voice": voice,
                             "speed": float(speed), "texts": texts,
                             "timestamps": True})
        return self._results(msg)

    @staticmethod
    def _results(msg: dict):
        # clips and timestamps are parallel lists; a None clip is the KOKORO
        # convention (transient miss), paired with its (null) timestamps.
        return [TtsServiceTcpEngine._clip(a, t)
                for a, t in zip(msg["clips"], msg["timestamps"])]

    @staticmethod
    def _clip(audio_b64, ts):
        if audio_b64 is None:
            return (None, None)
        return (wav_audio(b64_decode(audio_b64)), ts)
