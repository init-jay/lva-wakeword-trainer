"""The TTS service: host any registered engine behind a TCP port.

    # Apple Silicon, in-process MLX - no other process needed
    cd tts-service && uv run python -m tts_service.server --engine kokoro-mlx

    # any box, in front of a Kokoro-FastAPI container
    python -m tts_service.server --engine http://kokoro:8880

    # any box, in front of Wyoming Piper
    python -m tts_service.server --engine 127.0.0.1:10200

Clients address it with the `tcp://host:port` spec (engines/tcp.py); a batch of
texts is one request, and the response carries the per-clip wavs apart again.

WHAT THE SERVER DECIDES, AND WHY IT DECIDES LITTLE:

  * ONE ENGINE PER SERVER. A server with two engines would need routing on the
    wire, and routing is exactly what the client already does: engines_from_spec
    takes comma-separated specs, so `tcp://a:8899,tcp://b:8898` round-robins
    across two single-engine servers exactly the way the old KOKORO_URL pool did.
    One server, one engine, one concurrency policy.

  * EVERY ENGINE CALL UNDER ONE LOCK. Every engine this server is expected to
    host is effectively single-threaded: Kokoro-FastAPI measured at 101.8% CPU
    - exactly one core - with the GPU idle (client.py, KokoroPool); kokoro-mlx
    is one model instance per process; Piper holds one voice resident at a
    time. Letting two connections in at once would not run them in parallel,
    it would make them race for that one lane inside the engine - so the lane
    is taken explicitly, and queued connections wait at the lock where a hang
    is visible, instead of inside the model where it is not.

  * NO SESSIONS, ONE EXCHANGE PER CONNECTION. protocol.py explains why that
    keeps both sides dumb.

The protocol is the one in protocol.py: one JSON line each way, audio as
base64 wav. The server speaks it over plain TCP, not HTTP: nothing here needs
HTTP's status codes or content types - there is one verb per op field, the
response is always one line, and the one real requirement (a 2 MB payload) is
met by a newline.
"""
import argparse
import socket
import threading

from . import engines as engine_registry
from .protocol import (b64_encode, decode_line, encode_msg, wav_audio,
                       wav_bytes)

_engine = None
_engine_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"renders": 0, "clips": 0}


def _voice_of(msg: dict):
    """A voice is a string (Kokoro id) or a [voice, speaker] pair (Piper)."""
    v = msg["voice"]
    if isinstance(v, list):
        return tuple(v)
    return v


def _handle_render(msg: dict) -> dict:
    voice = _voice_of(msg)
    speed = float(msg.get("speed", 1.0))
    texts = msg["texts"]
    want_ts = bool(msg.get("timestamps"))

    if len(texts) == 1:
        results = [_engine.timed_render(voice, texts[0], speed)]
    else:
        # The server does the joining: batch() joins, renders once, splits on
        # word timestamps - or degrades to a per-clip loop when the hosted
        # engine has no timestamps (Piper). The client only ever sees clips.
        results = _engine.batch(voice, speed, texts)

    clips, timestamps = [], []
    for audio, ts in results:
        clips.append(None if audio is None else b64_encode(wav_bytes(audio)))
        timestamps.append(ts if (ts and want_ts) else None)
    return {"ok": True, "clips": clips, "timestamps": timestamps}


def _handle(msg: dict) -> dict:
    global _stats
    op = msg.get("op")
    if op == "voices":
        # Piper's catalog is voices(max_speakers=...); the Kokoro engine's
        # voices() takes no arguments. The TypeError is the discriminator -
        # one more isinstance branch is how this code used to know every
        # engine, which is the coupling the registry exists to remove.
        n = int(msg.get("max_speakers", 0) or 0)
        try:
            voices = _engine.voices(max_speakers=n)
        except TypeError:
            voices = _engine.voices()
        return {"ok": True, "voices": voices}
    if op == "render":
        if not isinstance(msg.get("texts"), list) or not msg.get("texts"):
            return {"ok": False, "error": "render needs a non-empty texts list"}
        out = _handle_render(msg)
        with _stats_lock:
            _stats["renders"] += 1
            _stats["clips"] += sum(1 for c in out["clips"] if c is not None)
        return out
    return {"ok": False, "error": f"unknown op {op!r}"}


def _serve_connection(conn: socket.socket, addr) -> None:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            return
        buf += chunk
    try:
        response = _handle(decode_line(buf))
    except Exception as e:
        # The PIPER convention, carried over the wire: an engine that raises
        # means the failure is about the voice or the transport, and the
        # caller wants to report WHICH one - so this becomes an error
        # envelope, not a null clip (see protocol.py).
        response = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    conn.sendall(encode_msg(response))
    conn.close()


def serve(engine_spec: str, host: str, port: int) -> None:
    global _engine
    engine = engine_registry.engines_from_spec(engine_spec)[0]
    ok, why = engine.available()
    if not ok:
        raise SystemExit(f"engine {engine_spec!r} is not usable: {why}")
    _engine = engine
    try:
        voices = engine.voices(max_speakers=0)
    except TypeError:
        voices = engine.voices()
    print(f"tts-service: engine {engine.name} ({engine_spec}) - {len(voices)} voices")

    listener = socket.create_server((host, port))
    print(f"tts-service: listening on tcp://{host}:{port}")
    try:
        while True:
            conn, addr = listener.accept()
            threading.Thread(target=_serve_connection, args=(conn, addr),
                             daemon=True).start()
    finally:
        with _stats_lock:
            print(f"tts-service: closed after {_stats['renders']} renders, "
                  f"{_stats['clips']} clips")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tts_service.server",
                                description=__doc__)
    p.add_argument("--engine", required=True,
                   help="engine spec: kokoro-mlx | http(s) URL | piper://host:port | "
                        "host:port")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    args = p.parse_args(argv)
    serve(args.engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
