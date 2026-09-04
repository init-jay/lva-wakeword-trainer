#!/usr/bin/env bash
#
# Run a full training pass on the trainer host.
#
# Wraps the sequence that has to happen in order, including the two steps that are
# easy to forget and expensive to get wrong:
#
#   * Kokoro must be UP for generation and DOWN for training. The GPU-resident
#     feature patch holds ~16.6 GiB of VRAM, so anything else on the card is the
#     difference between a run and an OOM. Two have happened, at opposite ends of
#     training: the validation step's ~2.76 GiB allocation killed a run at 37,500 of
#     50,000 steps, and the feature array's 16.09 GiB killed one before step 1 with
#     ~8 GiB of Kokoro still resident. Both after generation and feature computation
#     had completed - the expensive half is always already spent when this bites.
#
#   * The model must be checked for freshness. train.py now verifies this itself,
#     but the check is repeated here against the file you are about to copy off the
#     box, because a stale model was evaluated twice before identical checksums gave
#     it away.
#
# Usage:
#   ./scripts/run-oww-training.sh "hey seeree"
#   ./scripts/run-oww-training.sh "hey seeree" --samples-per-voice 400 --training-steps 100000
#   SKIP_CORPUS=1 ./scripts/run-oww-training.sh "hey seeree"
#
# Any extra arguments are passed through to train.py.
#
#   SKIP_BUILD=1     use the existing image (see below for when)
#   SKIP_CORPUS=1    reuse data/corpus/<wake>/oww/ instead of regenerating it
#
# SKIP_CORPUS IS FOR RESUMING, NOT FOR TUNING THE CORPUS. It exists because both
# recorded OOMs strike after generation has completed, so the failure destroys the
# cheap half of the run and preserves the expensive half. It skips TTS, real-clip
# copying and trimming, and still re-runs augmentation and training - so
# --training-steps, --layer-size and --max-negative-weight are all still live, while
# --samples-per-voice and friends are silently inert because the clips already exist.
# The unlike-mWW part: it also means no TTS server starts at all, which is the
# cleanest possible answer to the VRAM contention that caused the OOM.

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
MODEL="output/${SAFE_NAME}/oww/${SAFE_NAME}.onnx"
CORPUS="data/corpus/${SAFE_NAME}/oww"
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

# Rebuild first, unless told not to.
#
# WHAT STILL NEEDS A REBUILD, now that docker-compose bind-mounts train/ and
# scripts/ over the copies in the image: the openwakeword PATCHES, requirements.txt,
# and the Dockerfiles. Those are applied or installed at build time and cannot be
# mounted over. Editing train.py no longer needs one - the mount shadows the copy -
# which is a change from when this line was written.
#
# The reason it is still here by default is the patches. A patch edit that is not
# rebuilt runs the previous version silently, which is how a validation-batching fix
# appeared to have no effect and the OOM recurred.
#
# SKIP_BUILD=1 bypasses it. Worth using when the build cache has been pruned - the
# rebuild is then cold, costs a re-download of the CUDA base and torch, and
# repopulates tens of GB of cache. On a tight disk that is the opposite of what you
# want before a run that needs room for the corpus.
if [[ "${SKIP_BUILD:-}" == "1" ]]; then
    echo "=== $(date '+%H:%M:%S')  SKIP_BUILD=1 - using the existing image"
    echo "    Patches, requirements.txt and Dockerfile changes will NOT be picked up."
else
    echo "=== $(date '+%H:%M:%S')  building trainer image"
    docker compose build oww-trainer
fi

# Piper, only when the run actually asks for it. Unlike Kokoro it is NOT stopped
# before training: it runs CPU-only (--use-cuda measured 2.5x slower, see
# docker-compose.yml), so it holds no CUDA context and none of the VRAM that the
# GPU-resident feature patch needs. That is the whole reason the Kokoro dance below
# exists, and it does not apply here.
WANTS_PIPER=""
for arg in "$@"; do
    [[ "$arg" == --piper-fraction* ]] && WANTS_PIPER=1
done
# Neither TTS engine is needed when the corpus is being reused, so do not pay for
# either. Kokoro in particular is the point: not starting it is strictly better than
# starting it and stopping it again in a race, which is how the 16.09 GiB OOM
# happened. A SKIP_CORPUS=1 re-run has the whole card from the first instruction.
if [[ "${SKIP_CORPUS:-}" == "1" ]]; then
    WANTS_PIPER=""
fi
if [[ -n "$WANTS_PIPER" ]]; then
    echo "=== $(date '+%H:%M:%S')  starting Piper"
    docker compose up -d piper
fi

if [[ "${SKIP_CORPUS:-}" == "1" ]]; then
    # Down, not merely unstarted: a Kokoro left up by an earlier run holds ~8 GiB
    # and this path has no watcher to stop it, because there is no generation stage
    # to watch for.
    echo "=== $(date '+%H:%M:%S')  SKIP_CORPUS=1 - reusing $CORPUS, no TTS needed"
    docker compose stop kokoro kokoro2 >/dev/null 2>&1 || true
else
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
fi

# WAIT FROM INSIDE THE COMPOSE NETWORK, NOT FROM THE HOST. Piper speaks Wyoming over
# TCP rather than HTTP, so readiness is a connect check - and a host-side connect to
# localhost:10200 is a FALSE POSITIVE. Docker's port proxy accepts as soon as the
# container starts, before wyoming-piper has loaded its default voice and bound the
# port inside it. That check passed on its first attempt, printed "Piper ready" in
# the same second as "starting Piper", and the corpus container then died on
# "Connection refused" dialling piper:10200.
#
# The Kokoro loop above is unaffected: it issues a real HTTP request, which the proxy
# cannot answer on the app's behalf.
if [[ -n "$WANTS_PIPER" ]]; then
    docker compose run --rm --no-deps --entrypoint python3 oww-trainer -c "
import socket, sys, time
for _ in range(90):
    try:
        socket.create_connection(('piper', 10200), 2).close()
        sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit('piper did not start listening on piper:10200 within 180s')
"
    echo "=== $(date '+%H:%M:%S')  Piper ready"
fi

# Generation and feature computation. Kokoro is needed for the first, and the GPU
# headroom it occupies is harmless until training starts.
echo "=== $(date '+%H:%M:%S')  training (log: $LOG)"
: > "$LOG"

# Stop Kokoro the moment feature computation finishes, freeing its VRAM before
# openwakeword allocates. Started before training so the marker cannot be missed.
#
# THIS STOP IS ASYNCHRONOUS AND CANNOT BE RELIED ON ALONE. It is a 2 s poll, then
# SIGTERM, then two container shutdowns - racing a torch.empty() in the trainer that
# runs in milliseconds after the marker is printed. It lost, and the run died on
# "Tried to allocate 16.09 GiB ... 15.34 GiB is free" with ~8 GiB still held by
# Kokoro. train.oww.train therefore BLOCKS on the ports closing before it allocates;
# this loop is what makes that wait terminate, not what makes it safe.
#
# ~8 GiB across the two containers, measured against the public 0.8.1 image. An
# earlier ~2.4 GiB figure here predated that image and understated it by 3x, which
# is why the sizing is quoted with the image it was measured on.
# Poll the log rather than `tail -f | grep -q`. That pipeline is fragile in two
# ways that both fail SILENTLY, leaving Kokoro running and reproducing the OOM this
# exists to prevent: with pipefail inherited, grep -q exiting on a match kills
# tail -f with SIGPIPE and the pipeline reports failure, so the `&&` never runs; and
# BSD grep buffers stdin, so it may never process a line until EOF, which tail -f
# never sends. A polling loop has neither problem.
#
# Not started under SKIP_CORPUS=1: Kokoro was stopped before the run began, so there
# is nothing to wait for and a watcher would only sit there until the run ends.
WATCH_PID=""
if [[ "${SKIP_CORPUS:-}" != "1" ]]; then
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
fi

# Build the command with each argument quoted, so it survives being passed to
# `script` as a single string.
CMD="docker compose run --rm oww-trainer python -m train.oww.train"
# /app/data/external, NOT /app/data. The third-party corpora moved into
# data/external/ and train.py builds rir_paths/background_paths/feature_data_files
# by joining this prefix - so the old value points at directories that no longer
# exist, and openWakeWord augments with no impulse responses and no background audio
# rather than erroring.
CMD="$CMD --wake-word $(printf '%q' "$WAKE_WORD") --data-dir /app/data/external"
[[ "${SKIP_CORPUS:-}" == "1" ]] && CMD="$CMD --skip-corpus"
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
if [[ -n "$WATCH_PID" ]]; then
    { kill "$WATCH_PID" 2>/dev/null; pkill -P "$WATCH_PID" 2>/dev/null; \
      wait "$WATCH_PID"; } 2>/dev/null || true
fi

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

# Name the output by the code AND the audio that produced it - see
# train/provenance.py. Computed HERE, after training, on purpose: the synthetic
# corpus is built by the run itself, so hashing it beforehand would name the model
# after the previous run's audio.
TAG="$(python3 -m train.provenance --wake-word "$WAKE_WORD" --tag --fallback "$STAMP")"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
TAGGED="output/${SAFE_NAME}/oww/${SAFE_NAME}_${TAG}.onnx"
mkdir -p "$(dirname "$TAGGED")"
cp "$MODEL" "$TAGGED"

# The .tflite gets THE SAME TAG, from the same run. train.py converts it straight
# after export, so it is derived from exactly this .onnx - and a tagged .onnx beside
# an untagged .tflite is how a model and its conversion drift apart, which is
# precisely the mix-up eval/backends.py warns about when it says the two are not
# guaranteed to agree. Absent if the conversion failed; that is not fatal here, the
# .onnx is the artifact everything else works from.
TFLITE="${MODEL%.onnx}.tflite"
TAGGED_TFLITE="${TAGGED%.onnx}.tflite"
if [[ -f "$TFLITE" ]]; then
    cp "$TFLITE" "$TAGGED_TFLITE"
else
    echo "    NOTE: no $TFLITE - conversion did not run or failed. Produce it with"
    echo "          docker compose run --rm oww-trainer \\"
    echo "              python -m train.oww.onnx2tflite $TAGGED"
fi

echo "=== $(date '+%H:%M:%S')  DONE"
echo "    $TAGGED  ($(du -h "$TAGGED" | cut -f1), md5 ${AFTER_SUM:0:8})"
[[ -f "$TAGGED_TFLITE" ]] && \
    echo "    $TAGGED_TFLITE  ($(du -h "$TAGGED_TFLITE" | cut -f1), verified against the .onnx)"
[[ -n "$DIRTY" ]] && echo "    NOTE: working tree was dirty - the code half of $TAG is not reproducible"
echo
echo "    scp to the eval machine, then:"
echo "      docker compose run --rm eval python -m eval.compare_models \\"
echo "          --models <new> <previous-best>"
