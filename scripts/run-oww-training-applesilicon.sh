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
# contaminates the answer. See apple-port.md phase 1b.
#
#   ./scripts/setup-applesilicon-trainer.sh                      # once
#   ./scripts/run-oww-training-applesilicon.sh "hey seeree" --skip-corpus
#
# --skip-corpus IS THE INTENDED WAY TO USE THIS. Generation needs Kokoro, and this
# script starts nothing - point --kokoro-url at a server yourself if you want a full
# run (scripts/start-kokoro-host.sh). For the measurement it is the wrong thing to
# include anyway: TTS is the same work either way and would bury the difference under
# 49 minutes of noise.

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
