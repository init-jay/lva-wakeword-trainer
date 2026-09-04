#!/usr/bin/env bash
#
# Kokoro-FastAPI on the HOST, on Apple Silicon, using Metal.
#
# WHY THIS IS NOT A DOCKER SERVICE LIKE EVERYTHING ELSE HERE. Docker Desktop passes
# no Metal device through, so `DEVICE_TYPE=mps` inside a container finds nothing and
# falls back to CPU - silently, which is the bad kind of failure. Reaching the GPU on
# a Mac means running outside Docker. That is the same wall docker-compose.mps.yml
# documents for the trainers; this is the one service where going around it pays.
#
# WHAT IT BUYS. Measured on an M1 Max, same Kokoro-FastAPI v0.8.1 install throughout,
# with only DEVICE_TYPE changed between rows:
#
#     host, DEVICE_TYPE=mps      8.02 clips/s   120 ms median   RTF 17.0
#     host, DEVICE_TYPE=cpu      3.56 clips/s   283 ms median   RTF  7.5
#     docker, CPU image          ~4    clips/s
#
# 2.25x for Metal. The host-versus-container difference is nil - 3.56 against ~4 - so
# the gain is the device and not the environment, which is why the CPU row was
# measured on this same install rather than assumed from the Docker number.
#
# A SECOND INSTANCE BUYS NOTHING: 9.07 clips/s against 8.45, and throughput stays
# flat from 1 to 8 client threads while latency grows in proportion. Two processes
# share one GPU and serialise on it. Start ONE. That is the opposite of the CUDA box,
# where instances scale and the compose file runs kokoro and kokoro2 - do not carry
# that habit across.
#
# WHY MPS WINS HERE WHEN IT OFTEN DOES NOT. StyleTTS2's vocoder leans on FFT/STFT
# ops MPS does not implement, and the usual outcome is PYTORCH_ENABLE_MPS_FALLBACK
# scattering them to CPU mid-graph, paying a round trip each way and measuring slower
# than plain CPU. Kokoro-FastAPI avoids that by PLACING the ISTFT layers on CPU
# deliberately and keeping the rest on Metal ("Moving model to MPS device with CPU
# fallback for unsupported operations", api/src/inference/kokoro_v1.py). The fallback
# env var below is belt and braces, not the mechanism.
#
# USAGE:
#     ./scripts/start-kokoro-mps.sh                 # foreground, Ctrl-C to stop
#     ./scripts/start-kokoro-mps.sh --port 8890     # if 8880 is taken by Docker
#
# Then point a training run at it - from inside the compose network the host is
# host.docker.internal, and KOKORO_EXTERNAL stops the script starting its own:
#
#     KOKORO_EXTERNAL=1 KOKORO_URL=http://host.docker.internal:8880 \
#         ./scripts/run-oww-training.sh "hey seeree"
#
# HONEST EXPECTATION: 2.25x is real but it does not close the gap to the training
# box, whose Kokoro is CUDA-accelerated and generates the same corpus far faster.
# This makes an openWakeWord corpus on a Mac take about an hour instead of several.

set -euo pipefail

cd "$(dirname "$0")/.."

PORT=8880
DEVICE=mps
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2"; shift 2 ;;
        # For re-measuring the A/B above rather than for normal use - the Docker CPU
        # image is the better CPU path, since it needs no host Python at all.
        --cpu)    DEVICE=cpu; shift ;;
        *) echo "usage: $0 [--port N] [--cpu]" >&2; exit 2 ;;
    esac
done

# PINNED TO THE VERSION THE DOCKER IMAGE USES. docker-compose.yml builds Kokoro from
# ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1, and a host server on a different version
# would make "the Mac corpus differs from the VM corpus" ambiguous between the engine
# and the device. Bump both together or neither.
KOKORO_REF="v0.8.1"
APP_DIR="data/external/kokoro-fastapi"

if [[ "$(uname -m)" != "arm64" && "$DEVICE" == "mps" ]]; then
    echo "ERROR: --device mps needs Apple Silicon; this is $(uname -m)." >&2
    echo "       On any other machine use the Docker service instead:" >&2
    echo "         docker compose up -d kokoro" >&2
    exit 2
fi

command -v uv >/dev/null || { echo "ERROR: uv not found - see https://docs.astral.sh/uv/" >&2; exit 2; }

# espeak-ng MUST BE A SYSTEM INSTALL, and its absence is the first thing that breaks.
# The espeakng-loader wheel hardcodes the path it was BUILT at, so without a real
# install the server dies at first synthesis with a GitHub Actions runner path:
#
#     Error processing file '/Users/runner/work/espeakng-loader/.../phontab':
#         No such file or directory
#
# It reaches that point having already loaded the model onto Metal, so the failure
# looks like an MPS problem and is not.
ESPEAK_DATA="${ESPEAK_DATA_PATH:-/opt/homebrew/share/espeak-ng-data}"
if [[ ! -f "$ESPEAK_DATA/phontab" ]]; then
    echo "ERROR: no espeak-ng data at $ESPEAK_DATA" >&2
    echo "       brew install espeak-ng" >&2
    echo "       (or set ESPEAK_DATA_PATH if it lives elsewhere)" >&2
    exit 2
fi

# Under data/external/ with the other third-party downloads: not ours, re-fetchable,
# and already gitignored - so the ~450 MB torch install and the 327 MB model cannot
# reach a build context or a commit.
if [[ ! -d "$APP_DIR/.git" ]]; then
    echo "==> cloning Kokoro-FastAPI $KOKORO_REF into $APP_DIR"
    mkdir -p "$(dirname "$APP_DIR")"
    git clone --depth 1 --branch "$KOKORO_REF" \
        https://github.com/remsky/Kokoro-FastAPI.git "$APP_DIR"
fi

cd "$APP_DIR"

# NO EXTRAS, DELIBERATELY. Its pyproject maps torch to a CUDA index for the `gpu`
# extra - on aarch64 that is pytorch-cu129 - which would install CUDA wheels on a
# Mac. With no extra selected torch comes from plain PyPI, and the macOS arm64 wheel
# is the one that carries MPS.
#
# The venv is created FIRST. start-gpu_mac.sh upstream runs `uv pip install -e .`
# before any venv exists, which errors, and the `uv run --no-sync` after it skips
# installing - so the server starts without uvicorn and dies with
# "Failed to spawn: `uvicorn`".
[[ -d .venv ]] || uv venv
uv pip install -e . --quiet

uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0

PROJECT_ROOT="$(pwd)"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/api"
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH="$PROJECT_ROOT/web"
export DEVICE_TYPE="$DEVICE"
export USE_GPU=$([[ "$DEVICE" == "mps" ]] && echo true || echo false)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ESPEAK_DATA_PATH="$ESPEAK_DATA"
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}"

echo "==> Kokoro-FastAPI $KOKORO_REF on $DEVICE, port $PORT"
echo "    trainer URL: http://host.docker.internal:$PORT"
exec .venv/bin/uvicorn api.src.main:app --host 0.0.0.0 --port "$PORT"
