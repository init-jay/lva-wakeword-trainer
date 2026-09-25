#!/usr/bin/env bash
#
# Kokoro-FastAPI on the HOST, for Apple Silicon. CPU by default - see below.
#
# STATUS: this is now a DEBUGGING server, not a training or auditing one. Training
# corpora are rendered by the in-process kokoro-mlx engine in tts-service/engines/
# kokoro_mlx (protocol port 8900), and tools/audit_voices.py and bench_tts.py speak
# the protocol, not this OpenAI-compatible HTTP API - to audit or bench the
# FastAPI engine itself, run the docker kokoro image, which wraps it (protocol
# port 8899). What this host instance is for now is poking at the raw API
# directly. The measurements below remain the reason the Mac's training engine is
# MLX rather than this service. The historical finding
#
# Measured on an M1 Max with tools/bench_tts.py and a direct kokoro_tts_batch probe,
# rendering "hey seeree" through Kokoro-FastAPI v0.8.1. BATCHED, because that is how
# train/oww/train.py actually calls it (--tts-batch defaults to 16):
#
#     host,   cpu,  batched      88 ms/clip   11.4 clips/s   <- best, and the default
#     host,   mps,  UNbatched   110 ms/clip    9.1 clips/s
#     host,   mps,  batched     161 ms/clip    6.2 clips/s
#     docker, cpu,  batched     323 ms/clip    3.1 clips/s   <- what we had
#
# 3.7x, and NOT from the GPU. The container is native arm64 with all 10 CPUs (checked
# - `uname -m` says aarch64, and it is not emulated the way Dockerfile.piper's CUDA
# base was), so the gap is the torch build: the host runs the macOS arm64 wheel on
# Accelerate, the image a generic linux/arm64 one. That is the whole finding.
#
# WHY MPS LOSES DESPITE WINNING AN UNBATCHED BENCHMARK. Unbatched, Metal is 2.25x
# CPU - which is what an earlier measurement here reported, and it was misleading
# because the pipeline never renders unbatched. Kokoro-FastAPI keeps the ISTFT layers
# on CPU while the rest runs on Metal ("Moving model to MPS device with CPU fallback
# for unsupported operations", api/src/inference/kokoro_v1.py). Batching joins ~10
# texts into one long utterance, and ISTFT cost is LINEAR IN AUDIO LENGTH - so
# batching pushes ten times the work into the one stage that is not on the GPU, plus
# a ten times larger transfer back. Batching therefore measures 2.83x FASTER on cpu
# and 0.68x on mps: a net loss.
#
# --mps IS KEPT FOR ONE REAL CASE: run-ons, where it is the fastest option measured
# (143 ms/clip unbatched against 229 batched on cpu). Run-ons are ~40% of positives.
# Nobody has tried splitting the corpus across two servers by clip type; if the
# corpus stage ever needs to be faster than this, that is the next thing to measure.
#
# A SECOND INSTANCE BUYS NOTHING ON METAL: 9.07 clips/s against 8.45, throughput flat
# from 1 to 8 client threads while latency grows in proportion. Two processes share
# one GPU and serialise on it. On the CUDA box instances DO scale and compose runs
# kokoro and kokoro2 - do not carry that habit across.
#
# USAGE:
#     ./scripts/start-kokoro-host.sh                # cpu, foreground, Ctrl-C to stop
#     ./scripts/start-kokoro-host.sh --mps          # Metal, for run-on-heavy work
#     ./scripts/start-kokoro-host.sh --port 8890    # if 8880 is taken by Docker
#
# HONEST EXPECTATION: 2.25x is real but does not close the gap to the training box,
# whose Kokoro is CUDA-accelerated. This makes an oWW corpus on a Mac take about an
# hour instead of several.

set -euo pipefail

cd "$(dirname "$0")/../../"

PORT=8880
# cpu, because batched it is the fastest of the four configurations measured. --mps
# is the opt-in, not the default - the reverse of what this script assumed when it
# was written, and the header says why.
DEVICE=cpu
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2"; shift 2 ;;
        --mps)    DEVICE=mps; shift ;;
        *) echo "usage: $0 [--port N] [--mps]" >&2; exit 2 ;;
    esac
done

# PINNED TO THE VERSION THE DOCKER IMAGE USES. docker-compose.yml builds Kokoro from
# ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1, and a host server on a different version
# would make "the Mac corpus differs from the VM corpus" ambiguous between the engine
# and the device. Bump both together or neither.
KOKORO_REF="v0.8.1"
APP_DIR="data/external/kokoro-fastapi"

if [[ "$(uname -m)" != "arm64" && "$DEVICE" == "mps" ]]; then
    echo "ERROR: --mps needs Apple Silicon; this is $(uname -m)." >&2
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
# The venv is created FIRST: start-gpu_mac.sh upstream runs `uv pip install -e .`
# before any venv exists, which errors, and the `uv run --no-sync` after it skips
# installing - so the server starts without uvicorn and dies with
# "Failed to spawn: `uvicorn`".
[[ -d .venv ]] || uv venv
# PIN uv TO THAT VENV. uv 0.9.10 (Homebrew, Nov 2025) no longer auto-discovers the
# local .venv from this directory - it resolved a DIFFERENT venv of a newer Python,
# where tiktoken 0.8.0's cp312 wheels do not match, and rebuilt it from the sdist,
# which needs a Rust compiler. With VIRTUAL_ENV pinned, uv pip sees tiktoken already
# satisfied and only reinstalls the editable.
export VIRTUAL_ENV="$PWD/.venv"
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
