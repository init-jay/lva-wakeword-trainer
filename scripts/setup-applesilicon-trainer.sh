#!/usr/bin/env bash
#
# Prepare the HOST openWakeWord trainer on Apple Silicon. Setup only - it trains
# nothing; scripts/run-oww-training-applesilicon.sh does that.
#
# WHAT THIS IS FOR. docker/Dockerfile.oww.cpu does all of this at build time; this
# script does the same work outside a container, because the container's torch is
# measurably slower on this hardware. Measured with an identical script in both, torch
# 2.14 at the time:
#
#     matmul 2048^3             container 53.05 ms   host 17.42 ms    host 3.0x
#     train step, 1024 batch    container 26.39 ms   host  4.19 ms    host 6.3x
#     conv1d 512x16x96          container  7.38 ms   host 90.16 ms    host 12x SLOWER
#
# The macOS wheel links Accelerate; the linux/arm64 one does not. It has no oneDNN
# either, which is why convolutions go the other way - but openWakeWord's trainable
# model is Linear x7 and one LSTM with no convolutions at all, so this side of the
# pipeline sits squarely in the half where the host wins. microWakeWord is mixednet,
# convolutional, and must stay in its container. See apple-port.md phase 1b.
#
# WHETHER IT ACTUALLY HELPS THE REAL LOOP IS UNMEASURED, and the honest expectation is
# "less than 6.3x". The real training step draws 50 positives, not 1024, and a batch
# that small is dominated by Python and optimiser overhead rather than GEMM - exactly
# where a faster BLAS matters least. The container run this was written alongside
# measured 26 it/s, so that is the number to beat.
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

# espeak-ng, needed only by the optional MLX TTS backend (misaki's G2P). Warned about
# rather than fatal: the trainer itself does not need it, and --skip-corpus runs never
# touch TTS at all.
#
# The failure without it names a path on the machine the wheel was BUILT on, because
# espeakng-loader hardcodes it - and it surfaces after the model has loaded, so it
# reads as an MLX problem:
#
#     Error processing file '/Users/runner/work/espeakng-loader/.../phontab'
ESPEAK_DATA="${ESPEAK_DATA_PATH:-/opt/homebrew/share/espeak-ng-data}"
if [[ ! -f "$ESPEAK_DATA/phontab" ]]; then
    echo "WARNING: no espeak-ng data at $ESPEAK_DATA"
    echo "         The MLX TTS backend (--kokoro-url mlx://) will not work."
    echo "         Fix with: brew install espeak-ng"
fi

# --- the openWakeWord clone -------------------------------------------------------
#
# AT THE REPO ROOT, NOT INSIDE train-applesilicon/, and not by preference:
# train/oww/train.py resolves it as WORK_DIR/"openwakeword/..." where WORK_DIR is the
# repo root. Putting it anywhere else means patching train.py, which would then differ
# between the host and container paths - and the whole point is that they do not.
#
# .gitignore and .dockerignore both exclude it: it must never be committed, and it
# must never enter a build context, where it would shadow the clone the image makes
# for itself with one patched for a different device.
if [[ ! -d "$CLONE/.git" ]]; then
    echo "==> cloning openWakeWord into $CLONE/"
    git clone https://github.com/dscripka/openWakeWord "$CLONE"
fi

# --- patches ----------------------------------------------------------------------
#
# FOUR OF FIVE, the same four docker/Dockerfile.oww.cpu applies, for the same reasons.
# gpu-resident-features.py is omitted: it moves the 17.28 GB feature array into VRAM,
# and there is none here. On a 64 GB Mac the array simply lives in memory, which is
# the case apple-port.md calls architecturally better rather than merely adequate.
#
# Each patch prints "WARNING: patch target not found" and exits 0 rather than failing
# if upstream has moved, so re-running after an openWakeWord update is safe but the
# output is worth reading.
#
# Plus one the images do NOT apply: macos-dataloader-fork.py. macOS spawns worker
# processes where Linux forks, and the training DataLoader wraps a lambda and a
# method-local class, neither of which can be pickled. It is a no-op anywhere but
# darwin, so it lives here rather than in the shared four.
for p in skip-piper-import honour-augmentation-rounds feature-device-selection \
         configurable-corpus-dir macos-dataloader-fork; do
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

# --- the environment ---------------------------------------------------------------
#
# --extra mlx pulls kokoro-mlx, which renders the Kokoro corpus IN THIS PROCESS
# instead of over HTTP to a container or a host server. Measured on an M1 Max at
# 16 kHz: 57 ms/clip single against the service's 88 ms/clip BATCHED, and 25 ms/clip
# batched. Enabled with `--kokoro-url mlx://` on the training run; see
# train/corpus/kokoro_mlx.py.
#
# It is an extra rather than a plain dependency because it is Apple Silicon only and
# the trainer images must never need it - the backend imports it lazily and reports
# why it is unusable rather than failing at import.
echo "==> syncing $ENV_DIR"
( cd "$ENV_DIR" && uv sync --extra mlx --quiet )

# THE SPACY MODEL misaki NEEDS, INSTALLED HERE RATHER THAN ON FIRST USE.
#
# misaki (kokoro-mlx's G2P) downloads en_core_web_sm lazily by SHELLING OUT to
# `uv pip install`. That child process has no VIRTUAL_ENV when the trainer is invoked
# by absolute path rather than an activated venv, so it dies with
#
#     error: No virtual environment found; run `uv venv` to create an environment
#
# which names uv and says nothing about spacy, misaki or TTS. Worse, it happens at
# the first clip - after the run has printed its plan and started work.
#
# VIRTUAL_ENV is set explicitly for the same reason: spacy's downloader shells out
# too, and inherits whatever this script's environment gives it.
if [[ -d "$ENV_DIR/.venv" ]]; then
    if ! "$ENV_DIR/.venv/bin/python" -c "import en_core_web_sm" 2>/dev/null; then
        echo "==> installing en_core_web_sm (misaki G2P)"
        ( cd "$ENV_DIR" && VIRTUAL_ENV="$(pwd)/.venv" \
            .venv/bin/python -m spacy download en_core_web_sm 2>&1 | tail -1 )
    fi
fi

# openwakeword itself, installed from the clone WITHOUT its dependencies.
#
# --no-deps because its setup.py pins `torchaudio>=0.13.1,<1`, and satisfying that
# would drag torch back to 1.13 and 2022. Neither container obeys that pin either -
# pip leaves the already-installed wheel alone - so this matches their behaviour
# deliberately rather than diverging from it. Everything it actually needs is
# declared in train-applesilicon/pyproject.toml.
#
# MUST COME AFTER `uv sync`, AND MUST BE REDONE AFTER ANY LATER ONE. sync prunes
# whatever is not in the lockfile, and this package is deliberately not - so a bare
# `uv sync` in that directory silently uninstalls openwakeword and the next run dies
# on `ModuleNotFoundError: No module named 'openwakeword'`. Re-running this script is
# the supported way to repair that; it is idempotent.
( cd "$ENV_DIR" && uv pip install --no-deps -e "../$CLONE" --quiet )

# --- prove it ----------------------------------------------------------------------
#
# The same build-time checks the Dockerfile runs, for the same reason: a broken import
# should surface now, not after an hour of corpus generation.
#
# The MLX check RENDERS A CLIP rather than just importing, because the failures this
# path has are runtime ones: espeak-ng data resolving to a build-machine path, a
# missing spacy model, a model that downloads but will not run. An import proves none
# of those. It also exercises the word timestamps, which are what run-on cuts depend
# on - a backend that renders audio but returns no timestamps would silently push
# every run-on onto the +153 ms estimate.
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
from train.corpus import kokoro_mlx

ok, why = kokoro_mlx.available()
if not ok:
    print(f"  kokoro-mlx NOT usable: {why}")
    print("  (the HTTP path still works - see scripts/start-kokoro-host.sh)")
else:
    voices = kokoro_mlx.voices()
    audio, ts = kokoro_mlx.render_timed(voices[0], "hey seeree what is on tonight", 1.0)
    if audio is None or not ts:
        sys.exit("  kokoro-mlx rendered nothing, or returned no timestamps - "
                 "run-on cuts would fall back to the degraded estimate")
    print(f"  kokoro-mlx OK: {len(voices)} English voices, "
          f"{len(audio) / 16000:.2f}s clip, {len(ts)} word timestamps")
PY

echo
echo "==> ready. Train with:"
echo "      ./scripts/run-oww-training-applesilicon.sh \"hey seeree\" --skip-corpus"
echo
echo "    To generate a corpus too, in-process rather than via a TTS server:"
echo "      ./scripts/run-oww-training-applesilicon.sh \"hey seeree\" --kokoro-url mlx://"
