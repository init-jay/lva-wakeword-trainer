"""TTS corpus generation for wakeword training data.

Generates synthetic positive and negative samples using Kokoro TTS (and, for
the microWakeWord corpus, Piper). These synthetic voices complement the real
recordings in data/recordings/ to give the model a wider range of speaker
characteristics.

TRANSPORT (since the tts-service split, 2026-09-08): the modules in this
package no longer talk to any TTS engine directly. They speak the repo's TTS
PROTOCOL (tts-service/tts_protocol/) as a TCP client over `tcp://` URLs, and
the engines (kokoro-mlx on a Mac, Kokoro-FastAPI or Wyoming Piper in Docker)
run as separate tts-protocol servers that the trainer never has to know about.
The trainer environment therefore carries no engine dependencies at all -
only the protocol client, which is stdlib sockets plus the numpy/scipy it
already has.

Import bootstrap: the protocol package lives at tts-service/tts_protocol/ at
the repo root. The hyphens in both directory names mean it cannot be imported
by those names, so its parent is added to sys.path here and every module
below imports `tts_protocol` from it. The hyphens are deliberate: tts-service
is a self-contained layer with its own venv story and README, not a module
of train/.
"""
import sys
from pathlib import Path

_TTS_PROTOCOL_DIR = str(Path(__file__).resolve().parents[3]
                        / "src" / "tts-service" / "tts_protocol")
if _TTS_PROTOCOL_DIR not in sys.path:
    sys.path.insert(0, _TTS_PROTOCOL_DIR)
