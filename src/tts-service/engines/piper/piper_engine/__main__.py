"""Serve an in-process Piper engine on the repo's TTS protocol port.

    uv run --project tts-service/engines/piper python -m piper_engine \
        --voices data/external/piper/voices --port 8898

No backend process: piper-tts 1.7.0 runs inside this one. The models come
from the shared voices directory that src/scripts/download-external-data.sh
downloads; the default resolves it relative to the repo root.
"""
import argparse
from pathlib import Path

from tts_protocol.server import serve

from . import PiperEngine

# The engine moved under src/ with the restructure, so the repo root is one
# parent deeper than before (src/tts-service/engines/piper/piper_engine/ is
# five levels down, not four). data/external/ stays at the repo root.
DEFAULT_VOICES = Path(__file__).resolve().parents[5] / "data" / "external" / "piper" / "voices"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m piper_engine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voices", default=str(DEFAULT_VOICES),
                   help="piper voices directory (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1",
                   help="protocol listen host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8898,
                   help="protocol listen port (default: %(default)s)")
    args = p.parse_args(argv)
    serve(PiperEngine(args.voices), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
