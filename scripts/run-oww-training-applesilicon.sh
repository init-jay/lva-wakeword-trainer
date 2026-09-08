#!/usr/bin/env bash
#
# Run the openWakeWord trainer on the HOST, on Apple Silicon. The container-free
# counterpart to run-oww-training.sh.
#
# WHY IT IS A SEPARATE SCRIPT rather than a flag on run-oww-training.sh: that script
# is mostly `docker compose` - building images, starting and stopping Kokoro, waiting
# on ports inside a network that does not exist here. Threading a host mode through
# it would leave both paths harder to read than two short scripts.
#
# WHAT IT SHARES: the same train/oww/train.py, the same patches, the same pinned torch
# 2.5.1. That is deliberate. This exists to measure ONE variable - the macOS wheel
# against the linux/arm64 one - and anything else that differs between the two paths
# contaminates the answer; the measurements are in SPEED.md.
#
#   ./scripts/setup-applesilicon-trainer.sh                      # once
#   ./scripts/run-oww-training-applesilicon.sh "hey seeree" --skip-corpus
#
# --skip-corpus IS THE INTENDED WAY TO USE THIS. Generation needs TTS servers, and
# this script starts nothing - point --kokoro-url at a server yourself if you want
# a full run (scripts/start-kokoro-host.sh). --piper-fraction N adds Piper-rendered
# phrase-alone clips to that corpus; it needs its own server in another terminal,
# scripts/start-piper-host.sh (the same uv-venv mechanism, on 127.0.0.1:10200 -
# see PIPER_URL below). For the measurement a full run is the wrong thing to
# include anyway: TTS is the same work either way and would bury the difference
# under 49 minutes of noise.

set -euo pipefail

cd "$(dirname "$0")/.."
ENV_DIR="train-applesilicon"

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.py args...]" >&2
    exit 2
fi

# Same guard as run-oww-training.sh, same reason: a collapsed line continuation once
# passed environment assignments as arguments and a run trained on the wake word
# "hey seereeKOKORO_EXTERNAL=1", naming its outputs after it.
if [[ ! "$WAKE_WORD" =~ ^[A-Za-z][A-Za-z\'’-]*([[:space:]]+[A-Za-z][A-Za-z\'’-]*)*$ ]]; then
    echo "ERROR: '$WAKE_WORD' does not look like a wake word." >&2
    echo "       Expected words only - letters, spaces, apostrophes, hyphens." >&2
    exit 2
fi
shift

if [[ ! -x "$ENV_DIR/.venv/bin/python" ]]; then
    echo "ERROR: $ENV_DIR/.venv missing. Run ./scripts/setup-applesilicon-trainer.sh" >&2
    exit 2
fi
if [[ ! -d openwakeword/openwakeword ]]; then
    echo "ERROR: no openwakeword/ clone. Run ./scripts/setup-applesilicon-trainer.sh" >&2
    exit 2
fi

# host.docker.internal IS A CONTAINER-ONLY NAME. It is what run-oww-training.sh needs
# to reach a host Kokoro from inside the compose network, so it tends to be left
# exported in the shell - and here it resolves to nothing:
#
#     NameResolutionError: Failed to resolve 'host.docker.internal'
#     ERROR: no usable Kokoro servers
#
# On the host the same server is simply localhost. Rewrite rather than fail: the
# intent is unambiguous, and the alternative is an error about DNS for what is really
# a leftover environment variable.
if [[ "${KOKORO_URL:-}" == *host.docker.internal* ]]; then
    KOKORO_URL="${KOKORO_URL//host.docker.internal/localhost}"
    export KOKORO_URL
    echo "=== note: rewrote host.docker.internal -> localhost in KOKORO_URL"
    echo "          ($KOKORO_URL) - that name only resolves inside a container."
fi

# A leftover KOKORO_RUNON_URL points at a second TTS server that no longer has a
# reason to exist - run-ons and plain clips now render through the same pool. Left
# set, it fails the run at the voice probe (it did, against a server that had been
# stopped) long after the shell that exported it is out of mind.
if [[ -n "${KOKORO_RUNON_URL:-}" ]]; then
    echo "=== note: ignoring KOKORO_RUNON_URL - run-ons use the same pool now"
    unset KOKORO_RUNON_URL
fi

# KOKORO_EXTERNAL means nothing here - this script starts no containers, so every
# Kokoro is external. Unset it so it cannot be read as "something was arranged".
unset KOKORO_EXTERNAL

# PIPER_URL
#
# Same contract as KOKORO_URL: train/oww/train.py's --piper-url defaults to
# ${PIPER_URL}, so the script exports it only when it has an opinion, and an
# explicit --piper-url on the command line always wins over both.
#
# The container path reaches Piper as piper:10200 on the compose network, but a
# host process cannot resolve that name - the same dead end host.docker.internal
# is above, and the fix is the same shape: a local server. The Apple Silicon
# one is scripts/start-piper-host.sh (uv venv, voices under data/external/piper)
# on 127.0.0.1:10200. A PIPER_URL that still looks like the compose service
# name is rewritten with a note; any other value (a reachable host:port, e.g.
# Piper on the LAN) passes through untouched.
if [[ "${PIPER_URL:-}" == piper:* ]]; then
    PIPER_PORT="${PIPER_URL#piper:}"
    [[ -z "$PIPER_PORT" || ! "$PIPER_PORT" =~ ^[0-9]+$ ]] && PIPER_PORT=10200
    PIPER_URL="127.0.0.1:${PIPER_PORT}"
    export PIPER_URL
    echo "=== note: rewrote PIPER_URL to $PIPER_URL - the compose service name"
    echo "          only resolves inside the compose network. For a local server:"
    echo "          ./scripts/start-piper-host.sh"
fi

# ESPEAK FOR THE IN-PROCESS TTS BACKEND. misaki, which kokoro-mlx phonemises with,
# loads espeak-ng through a wheel that hardcodes its own build path - so without
# these it fails at the FIRST CLIP with a /Users/runner/... path, long after the run
# has started, and the message mentions neither TTS nor MLX.
#
# Exported unconditionally rather than only for mlx:// because they are inert
# otherwise: nothing else in the run reads them.
export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-/opt/homebrew/share/espeak-ng-data}"
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}"

# Fail before the corpus stage rather than during it. Both TTS backends are
# reachable from here as nothing but sockets, so a server that is down or not
# a TTS server at all would otherwise fail at clip 1 of thousands, and the
# log would not say why.
KOKORO_MLX=false
PIPER_FRACTION=""
PIPER_URL_ARG=""
want=""
for arg in "$@"; do
    if [[ -n "$want" ]]; then
        case "$want" in
            --piper-fraction) PIPER_FRACTION="$arg" ;;
            --piper-url) PIPER_URL_ARG="$arg" ;;
        esac
        want=""
        continue
    fi
    case "$arg" in
        mlx://*|mlx) KOKORO_MLX=true ;;
        --piper-fraction) want="--piper-fraction" ;;
        --piper-fraction=*) PIPER_FRACTION="${arg#*=}" ;;
        --piper-url) want="--piper-url" ;;
        --piper-url=*) PIPER_URL_ARG="${arg#*=}" ;;
    esac
done

if $KOKORO_MLX && [[ ! -f "$ESPEAK_DATA_PATH/phontab" ]]; then
    echo "ERROR: --kokoro-url mlx:// needs espeak-ng data at $ESPEAK_DATA_PATH" >&2
    echo "       brew install espeak-ng" >&2
    exit 2
fi

# Piper: probe only when it will actually render anything - a nonzero fraction
# (train.py's default is 0.0, i.e. off) and the corpus stage running at all
# (--skip-corpus renders nothing, so a dead server is a non-issue there).
# Normalize the off-cases to empty first. The flag can be passed explicitly as
# 0 (the default, i.e. off), and the zero-check must gate the probe itself, not
# just the skip-corpus check below - that was the first draft's bug.
case "$PIPER_FRACTION" in
    0|0.0) PIPER_FRACTION="" ;;
esac
if [[ -n "$PIPER_FRACTION" ]]; then
    for arg in "$@"; do
        [[ "$arg" == "--skip-corpus" ]] && PIPER_FRACTION=""
    done
fi
if [[ -n "$PIPER_FRACTION" ]]; then
    if [[ -n "$PIPER_URL_ARG" ]]; then
        export PIPER_URL="$PIPER_URL_ARG"
    elif [[ -z "${PIPER_URL:-}" ]]; then
        PIPER_URL="127.0.0.1:10200"
        export PIPER_URL
    fi
    # A Describe round trip, not a TCP connect: a bound port owned by a dead or
    # non-Wyoming listener passes a connect check and still fails the corpus
    # stage. This asks the question the corpus stage asks.
    if ! "$ENV_DIR/.venv/bin/python" - "$PIPER_URL" <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
from train.corpus.piper import piper_voices
url = sys.argv[1].rstrip("/")
host, _, port = url.partition(":")
port = int(port) if port.isdigit() else 10200
try:
    pairs = piper_voices(host, port, languages=("en_US", "en_GB"))
except Exception as e:
    print(f"  Piper unreachable: {e}", file=sys.stderr)
    sys.exit(1)
print(f"  Piper probe OK: {len(pairs)} (voice, speaker) pairs at {host}:{port}")
PYEOF
    then
        echo "  No reachable Piper at $PIPER_URL - the corpus stage would fail at its" >&2
        echo "  first phrase-alone render, not now. Start the host server in another" >&2
        echo "  terminal:  ./scripts/start-piper-host.sh   (it downloads voices on first use)" >&2
        echo "  or point PIPER_URL / --piper-url at an existing one." >&2
        exit 1
    fi
fi

# THE CLONE'S PATCHES ARE WORKING-TREE EDITS, and only setup-applesilicon-trainer.sh
# applies them (it runs the scripts in patches/). A working-tree reset in the clone -
# a bare `git checkout .` did exactly this on 2026-09-07 - silently undoes them, and
# the failure then surfaces two stages in, after the corpus is already generated. So
# verify here, before the spend: this is read-only, and the remedy is to re-run
# setup, which is idempotent and re-applies the same patches. Nothing in this script
# ever writes to the clone.
if ! grep -q 'if config.get("piper_sample_generator_path")' openwakeword/openwakeword/train.py; then
    echo "ERROR: the openWakeWord clone is missing its patches - the working tree was" >&2
    echo "       probably reset (e.g. a git checkout in openwakeword/). Re-run" >&2
    echo "       ./scripts/setup-applesilicon-trainer.sh (idempotent) to re-apply" >&2
    echo "       them, then start this run again." >&2
    exit 2
fi

SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
MODEL="output/${SAFE_NAME}/oww/${SAFE_NAME}.onnx"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="training-${SAFE_NAME}-macos-${STAMP}.log"

# THE CONTAINER MAY OWN THESE FILES. Both paths write data/corpus/ and output/, and
# the trainer images run as root - train/ownership.py hands output/ back afterwards,
# but data/corpus/ is left as root wrote it. A host run then fails on permissions
# somewhere unhelpful, so check here where the fix is obvious.
for d in "data/corpus/${SAFE_NAME}/oww" "output/${SAFE_NAME}/oww"; do
    if [[ -e "$d" && ! -w "$d" ]]; then
        echo "ERROR: $d is not writable by $(whoami) - a container run probably made it." >&2
        echo "       sudo chown -R \"$(whoami)\" $d" >&2
        exit 2
    fi
done

# Freshness check, same as the container path: a failed run leaves the previous
# model in place, and an unchanged file has been evaluated as a new one before.
BEFORE_SUM=""
[[ -f "$MODEL" ]] && BEFORE_SUM="$(md5 -q "$MODEL")"

echo "=== $(date '+%H:%M:%S')  host trainer, $("$ENV_DIR/.venv/bin/python" -c 'import torch; print("torch", torch.__version__)')"
echo "=== $(date '+%H:%M:%S')  training (log: $LOG)"

# `script -q` for a pty, so tqdm draws its progress bar - the same reason
# run-oww-training.sh uses it. BSD script takes the command as trailing arguments,
# not GNU's -c, hence the different form here.
set +e
script -q "$LOG" "$ENV_DIR/.venv/bin/python" -m train.oww.train \
    --wake-word "$WAKE_WORD" --data-dir data/external "$@"
STATUS=$?
set -e

echo
# Whether the model was WRITTEN is the real signal, not the exit code - openwakeword
# exits 1 on its own tflite conversion, which this repo replaces with onnx2tflite.py.
if [[ ! -f "$MODEL" ]]; then
    echo "=== TRAINING FAILED (exit $STATUS) - $MODEL does not exist. See $LOG" >&2
    exit "${STATUS:-1}"
fi
AFTER_SUM="$(md5 -q "$MODEL")"
if [[ -n "$BEFORE_SUM" && "$BEFORE_SUM" == "$AFTER_SUM" ]]; then
    echo "=== TRAINING FAILED (exit $STATUS) - $MODEL is unchanged from before this" >&2
    echo "    run. It is the PREVIOUS model. Do not evaluate or deploy it. See $LOG" >&2
    exit "${STATUS:-1}"
fi

echo "=== $(date '+%H:%M:%S')  DONE"
echo "    $MODEL"
echo
echo "    Compare the training rate against the container run's 26 it/s - that is"
echo "    the number this environment exists to beat. Then convert and evaluate:"
echo "      $ENV_DIR/.venv/bin/python -m train.oww.onnx2tflite $MODEL"
