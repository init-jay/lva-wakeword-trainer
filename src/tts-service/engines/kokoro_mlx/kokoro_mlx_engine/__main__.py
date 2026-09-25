"""Run Kokoro-82M in-process via MLX behind the repo's TTS protocol port.

Apple Silicon only (the MLX runtime). The model loads at startup - ~5.8 s cold,
~3.5 s with a warm Hugging Face cache (the module docstring carries the
measurements) - so a server that is listening and answers a catalog request has
already paid that.

    uv run --project tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine \
        --port 8900
"""
import argparse
import os

from tts_protocol.server import serve


def _espeak_defaults():
    # misaki (the kokoro-mlx phonemiser) loads espeak-ng through a wheel that
    # hardcodes its own BUILD path - /Users/runner/... - so without these it
    # fails at the FIRST CLIP with a path that mentions neither TTS nor MLX.
    # The Homebrew locations are the ones that exist on a Mac; a pre-set value
    # (a non-standard install) always wins.
    os.environ.setdefault("ESPEAK_DATA_PATH", "/opt/homebrew/share/espeak-ng-data")
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
    data = os.environ["ESPEAK_DATA_PATH"]
    if not os.path.isfile(os.path.join(data, "phontab")):
        raise SystemExit(
            f"espeak-ng data not found at {data} (misaki needs it for G2P).\n"
            "       brew install espeak-ng\n"
            f"       or point ESPEAK_DATA_PATH / PHONEMIZER_ESPEAK_LIBRARY at a"
            f" valid pair.")


def main(argv=None) -> int:
    _espeak_defaults()
    from . import KokoroMlxEngine  # after the env is set: misaki reads it at import

    p = argparse.ArgumentParser(prog="python -m kokoro_mlx_engine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="protocol listen host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8900,
                   help="protocol listen port (default: %(default)s)")
    args = p.parse_args(argv)
    serve(KokoroMlxEngine(), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
