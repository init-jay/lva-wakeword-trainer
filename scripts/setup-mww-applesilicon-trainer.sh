#!/usr/bin/env bash
#
# Prepare the HOST microWakeWord trainer on Apple Silicon. Setup only - it
# trains nothing; scripts/run-mww-training-applesilicon.sh does that.
#

#
#     ./scripts/setup-mww-applesilicon-trainer.sh
#
# Idempotent: re-running keeps a matching clone, re-syncs the venv.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENV_DIR="train-mww-applesilicon"
CLONE="microwakeword"

# THE PINNED COMMIT of github.com/OHF-Voice/micro-wake-word, the fork
# Dockerfile.mww.cpu installs from. The image tracks main (it records whatever it
# fetched in /opt/mww-commit.txt); this script pins exactly, so a run's provenance
# is complete and a move of main cannot change what is installed.
# 4665173 = main as of 2026-07-06, "Pin dependencies (#98)".
MWW_COMMIT="4665173cd35f1cff9a61e06fc427f124766c488e"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this is the Apple Silicon host path; you are on $(uname -s)/$(uname -m)." >&2
    echo "       Everywhere else, use the containers:" >&2
    echo "         COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml" >&2
    exit 2
fi

command -v uv >/dev/null || { echo "ERROR: uv not found - https://docs.astral.sh/uv/" >&2; exit 2; }

# --- the microWakeWord clone ------------------------------------------------------
#
# AT THE REPO ROOT, NOT INSIDE train-mww-applesilicon/. Not for path resolution (the
# stages import microwakeword from the venv, and nothing here reaches into the clone
# by path) but for the same reason as the openwakeword clone: .gitignore and
# .dockerignore both exclude it, so it never gets committed and never enters a build
# context, where a root-level directory named microwakeword could shadow what the
# image installs for itself.
if [[ ! -d "$CLONE/.git" ]]; then
    echo "==> cloning OHF-Voice/micro-wake-word into $CLONE/"
    git clone https://github.com/OHF-Voice/micro-wake-word "$CLONE"
fi
# The clone is a shared mutable state: a later pull or checkout moves HEAD, and
# the venv's editable install silently follows it. Force it back to the pin.
# Re-running this script is the repair for a moved HEAD too - the run script
# checks the pin before spending an hour.
if [[ "$(git -C "$CLONE" rev-parse HEAD 2>/dev/null)" != "$MWW_COMMIT" ]]; then
    echo "==> checking out $MWW_COMMIT in $CLONE/"
    git -C "$CLONE" fetch origin
    git -C "$CLONE" checkout "$MWW_COMMIT"
fi

# --- the environment ---------------------------------------------------------------
#
# VIRTUAL_ENV pinned on both uv invocations, for the exact reason documented in
# setup-applesilicon-trainer.sh: `uv pip` resolves its target environment from
# VIRTUAL_ENV FIRST, and a re-run from a shell with another venv activated (preflight's)
# once installed into THAT environment, surfacing two stages later as a
# ModuleNotFoundError.
echo "==> syncing $ENV_DIR"
( cd "$ENV_DIR" && VIRTUAL_ENV="$ENV_DIR/.venv" uv sync --quiet )

# microWakeWord itself, from the clone.
#
# TWO REASONS FOR --no-deps. The first is the oww one: its setup.py pins
# tensorflow>=2.18 and datasets with no bound at all; letting pip/uv resolve them
# fresh would float the two pins this environment exists to hold tensorflow==2.21.0
# and datasets[audio]<4.0, which the pyproject declares instead. The second is a
# build failure only a non-editable install hits: microwakeword/audio/ has no
# __init__.py, so find_packages() drops the whole subpackage from the WHEEL -
# `uv pip install git+...` produces an importable microwakeword whose
# microwakeword.audio is missing, and the failure surfaces as ModuleNotFoundError
# in the features stage, not at install. An editable install resolves against the
# source tree, where the directory is a namespace subpackage and imports fine -
# which is also why the Dockerfile's `pip install -e .` works.
( cd "$ENV_DIR" && VIRTUAL_ENV="$ENV_DIR/.venv" uv pip install --no-deps -e "../$CLONE" --quiet )
# uv's PEP 660 editable is FINDER-based (appended to sys.meta_path). A
# finder loses to a namespace package: when the caller's cwd is the repo
# root, the clone ROOT directory is a namespace portion via the cwd path
# entry, the PathFinder records it and never consults the finder, and
# `import microwakeword` resolves to the clone root (.__file__ is None) -
# the preflight `import` passes and the train stage dies with
# "No module named microwakeword.model_train_eval" (2026-09-22 host smoke
# run). The Docker image is immune: pip's classic editable writes
# easy-install.pth, a real sys.path entry, which the PathFinder honours.
# Repair: also write a path .pth with the clone root. Idempotent - the
# content is exactly one absolute path.
SITE="$("$ENV_DIR/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$(cd "$CLONE" && pwd)" > "$SITE/microwakeword-clone.pth"

# --- prove it ----------------------------------------------------------------------
#
# The same build-time checks Dockerfile.mww.cpu runs, for the same reason: a
# broken import should surface now, not after the corpus stage has spent its
# fourteen minutes.
"$ENV_DIR/.venv/bin/python" - <<'PY'
import numpy, tensorflow as tf, datasets
print(f"  python  {__import__('sys').version.split()[0]}")
print(f"  numpy   {numpy.__version__}  tensorflow {tf.__version__}")
assert numpy.__version__ >= "2", "mWW needs numpy>=2 - that is why this env is separate from train-applesilicon"
assert datasets.__version__.split(".")[0] == "3", "datasets must stay <4 - 4.x needs torchcodec (Dockerfile.mww.cpu)"
import pymicro_features          # noqa: F401
from mmap_ninja.ragged import RaggedMmap   # noqa: F401
# __file__ is None when the clone root shadows the package as a namespace
# package from this cwd (the PEP 660-finder case above) - the imports below
# would still pass for some submodules and fail for others, so assert the
# resolution, not just the import.
import microwakeword
assert microwakeword.__file__, "microwakeword resolved as a namespace package - see the .pth step above"
from microwakeword.audio.clips import Clips            # noqa: F401
from microwakeword.audio.spectrograms import SpectrogramGeneration  # noqa: F401
from microwakeword.audio.augmentation import Augmentation  # noqa: F401
import microwakeword.model_train_eval  # noqa: F401
import tensorboard                 # noqa: F401
import ai_edge_litert              # noqa: F401
print("  microwakeword + dependencies import OK")
import sys
sys.path.insert(0, ".")
import train.corpus  # noqa: F401  (sys.path bootstrap for tts_protocol)
from tts_protocol import TtsClient  # noqa: F401
TtsClient("tcp://127.0.0.1:8898")   # the URL policy the run script relies on
print("  tts-protocol OK")
PY

echo
echo "==> ready. Train with:"
echo "      ./scripts/run-mww-training-applesilicon.sh \"hey seeree\""
echo
echo "    The corpus stage needs the engines running; start them in other terminals:"
echo "      uv run --project tts-service/engines/piper python -m piper_engine --port 8898"
echo "      uv run --project tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900   # only for the 30% Kokoro mix"
