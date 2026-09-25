"""Front a running Kokoro-FastAPI process with the repo's TTS protocol port.

    uv run --project src/tts-service/engines/kokoro python -m kokoro_http_engine \
        --url http://127.0.0.1:8880 --port 8899

The FastAPI process itself is started separately: on a Mac by
src/scripts/start-kokoro-host.sh, in Docker by the kokoro compose service (whose
wrapper runs this same module against the in-image server).
"""
import argparse

from tts_protocol.server import serve

from . import KokoroHttpEngine


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m kokoro_http_engine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8880",
                   help="the Kokoro-FastAPI server, OpenAI-compatible (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1",
                   help="protocol listen host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8899,
                   help="protocol listen port (default: %(default)s)")
    args = p.parse_args(argv)
    serve(KokoroHttpEngine(args.url), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
