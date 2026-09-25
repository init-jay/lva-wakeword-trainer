"""The generic TTS server: host one Engine behind a TCP port, speaking the
protocol in wire.py.

Every engine server in this directory is this file plus a one-line main that
constructs its Engine and calls serve(). The protocol is the contract; this
is the whole server side of it, so a new machine is a new adapter module and
this file, not a new protocol.

    # Apple Silicon, in-process MLX
    cd src/tts-service/engines/kokoro_mlx && uv run python -m kokoro_mlx_engine --port 8900

    # the CUDA box: inside the kokoro image, in front of the in-image FastAPI
    # server (docker/Dockerfile.kokoro starts both, wrapper on 8899)
    python -m kokoro_http_engine --url http://127.0.0.1:8880 --port 8899

    # the piper engine, in-process on any machine (Mac uv project or the
    # Docker image that bakes in the same project)
    cd src/tts-service/engines/piper && uv run python -m piper_engine --port 8898

WHAT THE SERVER DECIDES, AND WHY IT DECIDES LITTLE:

  * ONE ENGINE PER SERVER. A server with two engines would need routing on the
    wire, and routing is exactly what the client already does: it round-robins
    across comma-separated tcp:// specs, so two single-engine servers share the
    work exactly the way the old KOKORO_URL pool did. One server, one engine,
    one concurrency policy.

  * EVERY ENGINE CALL UNDER ONE LOCK. Every engine this server is expected to
    host is effectively single-threaded: Kokoro-FastAPI measured at 101.8% CPU
    - exactly one core - with the GPU idle (train/corpus/kokoro.py,
    KokoroPool); kokoro-mlx is one model instance per process; Piper holds one
    voice resident at a time (engines/piper.py). Letting two connections in at
    once would not run them in parallel, it would make them race for that one
    lane inside the engine - so the lane is taken explicitly, and queued
    connections wait at the lock where a hang is visible, instead of inside
    the model where it is not.

  * NO SESSIONS, ONE EXCHANGE PER CONNECTION. wire.py explains why that keeps
    both sides dumb.

The server speaks the protocol over plain TCP, not HTTP: nothing here needs
HTTP's status codes or content types - there is one verb per op field, the
response is always one line, and the one real requirement (a multi-MB payload)
is met by a newline.
"""
import argparse
import socket
import threading

from .wire import (MAX_LINE, b64_encode, decode_line, encode_msg,
                  wav_bytes)

_engine = None
_engine_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"renders": 0, "clips": 0}
# The request is one small line (wire.py); no live client takes a minute to
# send it. A client that connects and never sends a newline used to pin this
# thread for the process lifetime, so the read is timed and capped like the
# client's identical framing (client.py). 60 s is far above anything real.
_RECV_TIMEOUT = 60.0


def _handle_render(msg: dict) -> dict:
    """One text in, one clip out - see wire.py on why batching is the
    client's job and never crosses the wire as a list."""
    voice = msg["voice"]
    # JSON has no tuples; the piper pair arrives as a list and the engine
    # wrapper reads it as a tuple. Kokoro voice ids are plain strings, which
    # pass through unchanged.
    if isinstance(voice, list):
        voice = tuple(voice)
    with _engine_lock:
        # The engine is the single serial lane (see the module docstring); the
        # encoding below stays outside it because it touches no engine state.
        audio, timestamps = _engine.timed_render(voice, msg["text"],
                                                 float(msg.get("speed", 1.0)))
    if audio is None:
        # The KOKORO convention, carried over the wire: the engine is alive but
        # rendered nothing. The client sees (None, None) and retries the clip
        # alone; the run continues.
        return {"ok": True, "clip": None, "timestamps": None}
    return {"ok": True,
            "clip": b64_encode(wav_bytes(audio)),
            "timestamps": timestamps if bool(msg.get("timestamps")) else None}


def _handle(msg: dict) -> dict:
    op = msg.get("op")
    if op == "voices":
        kwargs = {}
        # Engine-specific, optional: the Piper adapter honours both; the Kokoro
        # adapters ignore them. The capabilities in the answer are what the
        # corpus probe reads instead of paying a test synthesis.
        if msg.get("max_speakers"):
            kwargs["max_speakers"] = int(msg["max_speakers"])
        if msg.get("languages"):
            kwargs["languages"] = [str(l) for l in msg["languages"]]
        with _engine_lock:
            # No TypeError fallback to a no-argument call: it silently served
            # an adapter that raised INSIDE a current voices() its uncapped,
            # unfiltered catalog (2026-09-17 review). The Engine contract is
            # voices(**kwargs) (engine.py) and every adapter in this repo takes
            # it, so a nonconforming one should fail loudly here.
            voices = _engine.voices(**kwargs)
        return {"ok": True,
                # The engine's self-reported name: the corpus probe prints it,
                # so a URL pointed at the wrong server (a Piper port, a stale
                # process) is visible at the top of a run, not at its end.
                "engine": getattr(_engine, "name", "?"),
                "voices": [list(v) if isinstance(v, tuple) else v for v in voices],
                "timestamps": bool(_engine.supports_timestamps),
                "speaker": bool(_engine.speaker_voices)}
    if op == "render":
        if not isinstance(msg.get("text"), str) or not msg.get("text"):
            return {"ok": False, "error": "render needs a non-empty text"}
        out = _handle_render(msg)
        with _stats_lock:
            _stats["renders"] += 1
            _stats["clips"] += 1 if out.get("clip") else 0
        return out
    return {"ok": False, "error": f"unknown op {op!r}"}


def _serve_connection(conn: socket.socket, addr) -> None:
    conn.settimeout(_RECV_TIMEOUT)
    buf = b""
    response = None
    try:
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
            if len(buf) > MAX_LINE:
                # The client caps the identical framing; a line over the cap
                # is a protocol violation, and the answer is the error
                # envelope, not unbounded accumulation.
                response = {"ok": False, "error": f"request over {MAX_LINE} B"}
                break
    except OSError:
        # timed out (a client that never finished the line) or went away
        return
    if response is None:
        try:
            response = _handle(decode_line(buf))
        except Exception as e:
            # The PIPER convention, carried over the wire: an engine that raises
            # means the failure is about the voice or the transport, and the
            # caller wants to report WHICH one - so this becomes an error
            # envelope, not a null clip (see wire.py).
            response = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # The request cap does not fit the reply: a long render legitimately takes
    # minutes, so lift the timeout before sending it.
    conn.settimeout(None)
    conn.sendall(encode_msg(response))
    conn.close()


def serve(engine, host: str, port: int) -> None:
    """Run the server for `engine` forever. `engine` is an Engine (engine.py):
    available() must be true, and voices() must answer - both checked here so a
    dead backend fails at startup, before any client has pointed at the port.
    """
    global _engine
    ok, why = engine.available()
    if not ok:
        raise SystemExit(f"engine {engine.name!r} is not usable: {why}")
    _engine = engine
    # The Engine contract is voices(**kwargs) (engine.py); a nonconforming
    # adapter fails here, at startup, not silently mid-run.
    voices = engine.voices(max_speakers=0)
    print(f"tts-protocol: engine {engine.name} - {len(voices)} voices, "
          f"timestamps={engine.supports_timestamps}, "
          f"speaker={engine.speaker_voices}")

    listener = socket.create_server((host, port))
    print(f"tts-protocol: listening on tcp://{host}:{port}")
    try:
        while True:
            conn, addr = listener.accept()
            threading.Thread(target=_serve_connection, args=(conn, addr),
                             daemon=True).start()
    finally:
        with _stats_lock:
            print(f"tts-protocol: closed after {_stats['renders']} renders, "
                  f"{_stats['clips']} clips")


def main(argv=None) -> int:
    """The fallback main for engine adapters: each one's __main__.py parses its
    own backend flags, builds its Engine, and calls serve(). This entry point
    exists so `python -m tts_protocol.server --engine <name>` keeps working for
    anything importable, but adapters use their own mains because a backend
    URL flag is engine knowledge."""
    p = argparse.ArgumentParser(prog="python -m tts_protocol.server",
                                description=__doc__)
    p.add_argument("--engine", required=True,
                   help="import path of an Engine subclass, e.g. "
                        "kokoro_http_engine.KokoroHttpEngine")
    p.add_argument("--backend", default="",
                   help="passed as a single argument to the Engine constructor")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    args = p.parse_args(argv)
    mod, _, cls = args.engine.rpartition(".")
    import importlib
    engine = getattr(importlib.import_module(mod), cls)(args.backend) \
        if args.backend else getattr(importlib.import_module(mod), cls)()
    serve(engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
