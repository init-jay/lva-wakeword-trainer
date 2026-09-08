"""TTS corpus generation for wakeword training data.

Generates synthetic positive and negative samples using Kokoro TTS.
These synthetic voices complement the real recordings in data/recordings/
to give the model a wider range of speaker characteristics.

Import bootstrap: the shared TTS layer lives in tts-service/ at the repo root.
The hyphen in the directory name means it cannot be imported by that name, so its
parent is added to sys.path here and every module below imports the engines,
the batch algorithm and the pool from tts_service. The hyphen itself is
deliberate: tts-service is a self-contained layer with its own venv story and
README, not a module of train/.
"""
import sys
from pathlib import Path

_TTS_DIR = str(Path(__file__).resolve().parents[2] / "tts-service")
if _TTS_DIR not in sys.path:
    sys.path.insert(0, _TTS_DIR)
