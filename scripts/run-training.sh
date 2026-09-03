#!/usr/bin/env bash
#
# Run a full training pass on the trainer host.
#
# Wraps the sequence that has to happen in order, including the two steps that are
# easy to forget and expensive to get wrong:
#
#   * Kokoro must be UP for generation and DOWN for training. The GPU-resident
#     feature patch holds ~16.6 GiB of VRAM, and openwakeword's validation step
#     allocates ~2.76 GiB in one go. With Kokoro's two CUDA contexts (~2.4 GiB) still
#     resident, that overflows - which killed a run at 37,500 of 50,000 steps, after
#     generation and feature computation had already completed.
#
#   * The model must be checked for freshness. train.py now verifies this itself,
#     but the check is repeated here against the file you are about to copy off the
#     box, because a stale model was evaluated twice before identical checksums gave
#     it away.
#
# Usage:
#   ./scripts/run-training.sh "hey seeree"
#   ./scripts/run-training.sh "hey seeree" --samples-per-voice 400 --training-steps 100000
#
# Any extra arguments are passed through to train.py.

set -euo pipefail

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.py args...]" >&2
    exit 2
fi
shift

# The REPO ROOT, not this script's directory - it moved to scripts/ in the reorg
# and every docker compose call below needs the compose file in the working dir.
cd "$(dirname "$0")/.."

# tr rather than ${x,,} so this does not need bash 4 (macOS ships 3.2).
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
MODEL="my_custom_model/${SAFE_NAME}.onnx"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="training-${SAFE_NAME}-${STAMP}.log"

# Record the current model so a stale one cannot be mistaken for this run's output.
BEFORE_SUM=""
[[ -f "$MODEL" ]] && BEFORE_SUM="$(md5sum "$MODEL" | cut -d' ' -f1)"

cleanup() {
    # Always leave Kokoro running: the next run needs it, and a stopped container
    # produces a confusing "no usable Kokoro servers" failure at startup.
    docker compose start kokoro kokoro2 >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Rebuild first. train.py and the openwakeword patches are baked into the image, so
# a code change that is not rebuilt runs the previous version - which is how a
# validation-batching fix appeared to have no effect and the OOM recurred. Cached
# layers make this a few seconds when nothing has changed.
echo "=== $(date '+%H:%M:%S')  building trainer image"
docker compose build trainer

# Piper, only when the run actually asks for it. Unlike Kokoro it is NOT stopped
# before training: it runs CPU-only (--use-cuda measured 2.5x slower, see
# docker-compose.yml), so it holds no CUDA context and none of the VRAM that the
# GPU-resident feature patch needs. That is the whole reason the Kokoro dance below
# exists, and it does not apply here.
WANTS_PIPER=""
for arg in "$@"; do
    [[ "$arg" == --piper-fraction* ]] && WANTS_PIPER=1
done
if [[ -n "$WANTS_PIPER" ]]; then
    echo "=== $(date '+%H:%M:%S')  starting Piper"
    docker compose up -d piper
fi

echo "=== $(date '+%H:%M:%S')  starting Kokoro"
docker compose up -d kokoro kokoro2

# Wait for readiness rather than assuming: the GPU image spends a while loading
# voices, and train.py's probe would otherwise fail on a container that is up but
# not yet serving.
for name in kokoro:8880 kokoro2:8881; do
    port="${name##*:}"
    for _ in $(seq 1 60); do
        curl -sf "http://localhost:${port}/v1/audio/voices" >/dev/null 2>&1 && break
        sleep 2
    done
done
echo "=== $(date '+%H:%M:%S')  Kokoro ready"

# Piper speaks Wyoming over TCP, not HTTP, so readiness is a connect check rather
# than a curl. Waiting matters as much as it does for Kokoro: the container binds
# the port only after loading its default voice, and train.py's voice enumeration
# would otherwise fail against a container that is up but not yet serving.
if [[ -n "$WANTS_PIPER" ]]; then
    for _ in $(seq 1 60); do
        (exec 3<>/dev/tcp/localhost/10200) 2>/dev/null && { exec 3<&-; break; }
        sleep 2
    done
    echo "=== $(date '+%H:%M:%S')  Piper ready"
fi

# Generation and feature computation. Kokoro is needed for the first, and the GPU
# headroom it occupies is harmless until training starts.
echo "=== $(date '+%H:%M:%S')  training (log: $LOG)"
: > "$LOG"

# Stop Kokoro the moment feature computation finishes, freeing its ~2.4 GiB before
# openwakeword's validation allocation needs it. Started before training so the
# marker cannot be missed.
# Poll the log rather than `tail -f | grep -q`. That pipeline is fragile in two
# ways that both fail SILENTLY, leaving Kokoro running and reproducing the OOM this
# exists to prevent: with pipefail inherited, grep -q exiting on a match kills
# tail -f with SIGPIPE and the pipeline reports failure, so the `&&` never runs; and
# BSD grep buffers stdin, so it may never process a line until EOF, which tail -f
# never sends. A polling loop has neither problem.
MAIN_PID=$$
(
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        if grep -q "Training model" "$LOG" 2>/dev/null; then
            echo "=== $(date '+%H:%M:%S')  training stage reached, stopping Kokoro"
            docker compose stop kokoro kokoro2 >/dev/null 2>&1 || true
            break
        fi
        sleep 2
    done
) &
WATCH_PID=$!

# Build the command with each argument quoted, so it survives being passed to
# `script` as a single string.
CMD="docker compose run --rm trainer python -m train.oww.train"
CMD="$CMD --wake-word $(printf '%q' "$WAKE_WORD") --data-dir /app/data"
for arg in "$@"; do CMD="$CMD $(printf '%q' "$arg")"; done

# Run under `script` so the container gets a pty. Piping to tee otherwise denies
# docker a TTY, and tqdm then has no terminal to draw on - the training progress
# bar disappears entirely. `script -e` propagates the child's exit status.
#
# Foreground with PIPESTATUS, because a backgrounded pipeline reports tee's status
# rather than docker's and would call a failed run a success.
set +e
if script -qec true /dev/null >/dev/null 2>&1; then
    script -qec "$CMD" /dev/null 2>&1 | tee -a "$LOG"
else
    # BSD/macOS script takes different flags; fall back to a plain pipe, which
    # costs the live progress bar but keeps the log and the exit status.
    eval "$CMD" 2>&1 | tee -a "$LOG"
fi
STATUS=${PIPESTATUS[0]}
set -e

# `wait` after `kill` suppresses bash's asynchronous "Terminated" job-control
# message, which would otherwise print in the middle of the summary.
{ kill "$WATCH_PID" 2>/dev/null; pkill -P "$WATCH_PID" 2>/dev/null; \
  wait "$WATCH_PID"; } 2>/dev/null || true

echo
# Whether the model was WRITTEN is the real signal, not the exit code. openwakeword
# saves the .onnx and then tries to convert it to tflite via onnx_tf, which this
# image deliberately does not carry (it never worked - tensorflow-cpu 2.8.1 against
# protobuf >= 3.20 - and onnx2tf replaced it), so that step exits 1. Treating that
# as a failure would discard a good run - the README documents it under "TFLite
# conversion error at end".
if [[ ! -f "$MODEL" ]]; then
    echo "=== TRAINING FAILED (exit $STATUS) - $MODEL does not exist. See $LOG"
    exit "${STATUS:-1}"
fi

AFTER_SUM="$(md5sum "$MODEL" | cut -d' ' -f1)"
if [[ -n "$BEFORE_SUM" && "$BEFORE_SUM" == "$AFTER_SUM" ]]; then
    echo "=== TRAINING FAILED (exit $STATUS) - $MODEL is unchanged from before this"
    echo "    run. It is the PREVIOUS model. Do not evaluate or deploy it. See $LOG"
    exit "${STATUS:-1}"
fi

if [[ $STATUS -ne 0 ]]; then
    echo "=== NOTE: training exited $STATUS but the model WAS written."
    echo "    Normally the tflite conversion failing after the .onnx is saved."
    echo "    Convert with train/oww/onnx2tflite.py, which verifies the result."
fi

# Name the output by commit so a model can be traced back to the code that made it.
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "$STAMP")"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
TAGGED="my_custom_model/${SAFE_NAME}/${SAFE_NAME}_${COMMIT}${DIRTY}.onnx"
mkdir -p "$(dirname "$TAGGED")"
cp "$MODEL" "$TAGGED"

echo "=== $(date '+%H:%M:%S')  DONE"
echo "    $TAGGED  ($(du -h "$TAGGED" | cut -f1), md5 ${AFTER_SUM:0:8})"
[[ -n "$DIRTY" ]] && echo "    NOTE: working tree was dirty - this model is not reproducible from $COMMIT"
echo
echo "    scp to the eval machine, then:"
echo "      docker compose run --rm eval python -m eval.compare_models \\"
echo "          --models <new> <previous-best>"
