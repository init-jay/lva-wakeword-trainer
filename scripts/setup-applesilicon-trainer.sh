#!/usr/bin/env bash
#
# Prepare the HOST openWakeWord trainer on Apple Silicon. Setup only - it trains
# nothing; scripts/run-oww-training-applesilicon.sh does that.
#
#     ./scripts/setup-applesilicon-trainer.sh
#
# Idempotent: re-running re-applies patches (each is a no-op if already applied) and
# skips downloads that are present.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENV_DIR="train-applesilicon"
CLONE="openwakeword"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this is the Apple Silicon host path; you are on $(uname -s)/$(uname -m)." >&2
    echo "       Everywhere else, use the containers:" >&2
    echo "         COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml" >&2
    exit 2
fi

command -v uv >/dev/null || { echo "ERROR: uv not found - https://docs.astral.sh/uv/" >&2; exit 2; }

# espeak-ng, needed by the Kokoro MLX ENGINE (misaki's G2P) - now an independent uv
# project (tts-service/engines/kokoro_mlx) that reads it at its own startup and
# refuses with a clear message if it is missing. Warned about here only because the
# corpus stage of a full run needs that engine to be running:
ESPEAK_DATA="${ESPEAK_DATA_PATH:-/opt/homebrew/share/espeak-ng-data}"
if [[ ! -f "$ESPEAK_DATA/phontab" ]]; then
    echo "WARNING: no espeak-ng data at $ESPEAK_DATA"
    echo "         The Kokoro engine (tts-service/engines/kokoro_mlx) will not start."
    echo "         Fix with: brew install espeak-ng"
fi

# --- the openWakeWord clone -------------------------------------------------------
#
# AT THE REPO ROOT, NOT INSIDE train-applesilicon/, and not by preference: train/oww/
# train.py resolves it as WORK_DIR/"openwakeword/..." where WORK_DIR is the repo root.
# Putting it anywhere else means patching train.py, which would then differ between
# the host and container paths - and the whole point is that they do not.
#
# .gitignore and .dockerignore both exclude it: it must never be committed, and never
# enter a build context, where it would shadow the clone the image makes for itself
# (patched for a different device).
if [[ ! -d "$CLONE/.git" ]]; then
    echo "==> cloning openWakeWord into $CLONE/"
    git clone https://github.com/dscripka/openWakeWord "$CLONE"
fi

# --- patches ----------------------------------------------------------------------
#
# SIX OF SEVEN, the same six docker/Dockerfile.oww.cpu applies, for the same reasons.
# gpu-resident-features.py is omitted: it moves the 17.28 GB feature array into VRAM,
# and there is none here. On a 64 GB Mac the array simply lives in memory, which
# removes the problem rather than working around it - a memory setting, not a
# hardware limit (docker-compose.cpu.yml).
#
# Each patch prints "WARNING: patch target not found" and exits 0 rather than
# failing if upstream has moved, so re-running after an openWakeWord update is safe
# but the output is worth reading. The reverse failure - the patches silently UNdone
# by a working-tree reset in the clone - is caught by
# run-oww-training-applesilicon.sh, which checks for the sentinel before launching
# and sends you back here.
#
# Plus one the images do NOT apply: macos-dataloader-fork.py. macOS spawns worker
# processes where Linux forks, and the training DataLoader wraps a lambda and a
# method-local class, neither of which can be pickled. It is a no-op anywhere but
# darwin, so it lives here rather than in the shared four.
for p in skip-piper-import honour-augmentation-rounds feature-device-selection \
         configurable-corpus-dir seed-augment configurable-lr macos-dataloader-fork; do
    python3 "patches/$p.py" "$CLONE/openwakeword/train.py"
done

# --- the embedding models ---------------------------------------------------------
#
# Small, and openWakeWord will not run without them. Fetched to a .part file and
# renamed, so an interrupted download cannot leave a truncated model that fails much
# later with a confusing onnxruntime error.
MODELS="$CLONE/openwakeword/resources/models"
mkdir -p "$MODELS"
BASE="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
for m in embedding_model melspectrogram; do
    if [[ ! -f "$MODELS/$m.onnx" ]]; then
        echo "==> downloading $m.onnx"
        curl -L --fail -o "$MODELS/$m.onnx.part" "$BASE/$m.onnx"
        mv "$MODELS/$m.onnx.part" "$MODELS/$m.onnx"
    fi
done

# --- the environments --------------------------------------------------------------
# THREE uv projects, all synced here, because the corpus stage needs the engines
# running while training runs in the trainer's venv:
#
#   $ENV_DIR                       the trainer itself - it now carries only the
#                                  tts-protocol client (path dep) plus the training
#                                  stack; no TTS engine, no Metal dependency, no spacy.
#   tts-service/engines/kokoro_mlx the Kokoro engine (MLX, protocol port 8900)
#   tts-service/engines/piper      the Piper engine (in-process piper-tts, 8898)
#
# The engine syncs are no-ops after the first; on a fresh box they pull torch +
# mlx (GBs), which is why the setup step owns them rather than a training run
# discovering them at clip 1.
for proj in "$ENV_DIR" tts-service/engines/kokoro_mlx tts-service/engines/piper; do
    echo "==> syncing $proj"
    ( cd "$proj" && uv sync --quiet )
done

# openwakeword itself, installed from the clone WITHOUT its dependencies.
#
# --no-deps because its setup.py pins `torchaudio>=0.13.1,<1`, and satisfying that
# would drag torch back to 1.13 and 2022. Neither container obeys that pin either -
# pip leaves the already-installed wheel alone - so this matches their behaviour
# deliberately rather than diverging from it. Everything it actually needs is declared
# in train-applesilicon/pyproject.toml.
#
# MUST COME AFTER `uv sync`, AND MUST BE REDONE AFTER ANY LATER ONE: sync prunes
# whatever is not in the lockfile, and this package is deliberately not - so a bare
# `uv sync` in that directory silently uninstalls openwakeword and the next run dies
# on `ModuleNotFoundError: No module named 'openwakeword'`. Re-running this script is
# the supported repair; it is idempotent.
#
# The VIRTUAL_ENV pin is load-bearing here, not decorative: unlike `uv sync`, `uv pip`
# resolves its target environment from VIRTUAL_ENV before the project's .venv, so with
# any other venv activated in the invoking shell the editable install lands in THAT
# environment (incident of 2026-09-07, see the comment above the sync line).
( cd "$ENV_DIR" && VIRTUAL_ENV="$ENV_DIR/.venv" uv pip install --no-deps -e "../$CLONE" --quiet )

# --- prove it ----------------------------------------------------------------------
#
# The same build-time checks the Dockerfile runs, for the same reason: a broken
# import should surface now, not after an hour of corpus generation. The TTS check
# proves the trainer's half of the split: the protocol client imports and enforces
# the URL policy the run script will rely on - the failures this path has are
# runtime ones (a missing path dep, a broken sys.path bootstrap, a client that
# accepts a URL form it will misread).
ESPEAK_DATA_PATH="$ESPEAK_DATA" \
PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}" \
"$ENV_DIR/.venv/bin/python" - <<'PY'
import torch, torchaudio, onnxruntime
print(f"  torch {torch.__version__}  torchaudio {torchaudio.__version__}")
print(f"  mps available: {torch.backends.mps.is_available()}")
print(f"  onnxruntime {onnxruntime.__version__}: {onnxruntime.get_available_providers()}")
import torch_audiomentations          # noqa: F401
import openwakeword.data              # noqa: F401
print("  openwakeword.data imports OK")

import sys
sys.path.insert(0, ".")
import train.corpus  # noqa: F401  (sys.path bootstrap for tts_protocol)
from tts_protocol import TtsClient, phrase_end_sample
from tts_protocol.client import TtsProtocolError
# The client must accept the URL form the run script exports, and reject the old
# raw forms it used to misread as a different backend.
TtsClient("tcp://127.0.0.1:8900")
for bad in ("http://127.0.0.1:8880", "127.0.0.1:8900", "mlx://"):
    try:
        TtsClient(bad)
    except TtsProtocolError:
        pass
    else:
        sys.exit(f"  TtsClient accepted a non-protocol spec: {bad}")
assert phrase_end_sample(
    {0: {"word": "hey", "start": 0.0, "end": 0.1},
     1: {"word": "seeree", "start": 0.1, "end": 0.3}}, "seeree") == 4800
print("  tts-protocol OK: client URL policy + phrase_end_sample")
PY

echo
echo "==> ready. Train with:"
echo "      ./scripts/run-oww-training-applesilicon.sh \"hey seeree\" --skip-corpus"
echo
echo "    To generate a corpus too, start the engines first (each in its own"
echo "    terminal - the run script's probes tell you which one is missing):"
echo "      uv run --project tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900"
echo "      uv run --project tts-service/engines/piper python -m piper_engine --port 8898"
echo "      ./scripts/run-oww-training-applesilicon.sh \"hey seeree\"  # add --piper-fraction 0.3 for the Piper mix"
