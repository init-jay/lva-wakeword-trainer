#!/usr/bin/env bash
#
# Run a full training pass on the trainer host.
#
# Usage:
#   ./src/scripts/run-oww-training.sh "hey seeree"
#   ./src/scripts/run-oww-training.sh "hey seeree" --samples-per-voice 400 --training-steps 100000
#   SKIP_CORPUS=1 ./src/scripts/run-oww-training.sh "hey seeree"
#
# Any extra arguments are passed through to train.py.
#
#   SKIP_BUILD=1     use the existing image (see below for when)
#   SKIP_CORPUS=1    reuse data/corpus/<wake>/oww/ instead of regenerating it
set -euo pipefail

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.py args...]" >&2
    exit 2
fi

# A WAKE WORD IS WORDS. Anything else is a mistyped command line, and this check
# exists because one got through: a pasted multi-line invocation whose `\`
# continuation collapsed passed the environment assignments as ARGUMENTS, so
# KOKORO_EXTERNAL=1 was never set and the run started on the wake word
#
#     hey seereeKOKORO_EXTERNAL=1
#
# It named its outputs after that, took a minute to fail, and failed somewhere
# unrelated - in train.py's argument parser, reporting a missing .onnx. Everything
# downstream here derives paths from this string, so a bad one is cheap to catch
# now and confusing to diagnose later.
#
# Letters, spaces, apostrophes and hyphens only. Deliberately narrow: the wordlists
# and the TTS engines both take plain text, and no legitimate wake word has needed
# more. Widen it if a real one does, not to make an error message go away.
if [[ ! "$WAKE_WORD" =~ ^[A-Za-z][A-Za-z\'’-]*([[:space:]]+[A-Za-z][A-Za-z\'’-]*)*$ ]]; then
    echo "ERROR: '$WAKE_WORD' does not look like a wake word." >&2
    echo "       Expected words only - letters, spaces, apostrophes, hyphens." >&2
    if [[ "$WAKE_WORD" == *=* ]]; then
        echo >&2
        echo "       It contains '='. Environment assignments must come BEFORE the" >&2
        echo "       script, and a pasted line continuation often loses them:" >&2
        echo "         export KOKORO_EXTERNAL=1 KOKORO_URL=tcp://<box>:8899" >&2
        echo "         $0 \"hey seeree\"" >&2
    fi
    exit 2
fi
shift

# The REPO ROOT, not this script's directory - it moved to scripts/ in the reorg
# and every docker compose call below needs the compose file in the working dir.
cd "$(dirname "$0")/../../"

# tr rather than ${x,,} so this does not need bash 4 (macOS ships 3.2).
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
MODEL="output/${SAFE_NAME}/oww/${SAFE_NAME}.onnx"
CORPUS="data/corpus/${SAFE_NAME}/oww"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p src/logs
LOG="src/logs/training-${SAFE_NAME}-${STAMP}.log"

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
    [[ -z "${KOKORO_EXTERNAL:-}" ]] && docker compose stop kokoro kokoro2 >/dev/null 2>&1
    true
elif [[ -n "${KOKORO_EXTERNAL:-}" ]]; then
    # KOKORO IS SOMEONE ELSE'S PROBLEM. Start nothing, stop nothing, and take
    # KOKORO_URL exactly as given.
    #
    # THE CASE THIS EXISTS FOR IS METAL. Docker Desktop passes no Metal device
    # through, so a Metal Kokoro has to be a HOST process. On a Mac that host
    # engine is the in-process mlx one on 8900, and it takes precedence: the
    # Mac does not run docker kokoro at all - which is also why no compose
    # service publishes 8900 (docker-compose.yml).
    #
    # Measured on an M1 Max, same Kokoro-FastAPI v0.8.1 install throughout,
    # only DEVICE_TYPE changed:
    #
    #     host, DEVICE_TYPE=mps    8.02 clips/s   120 ms median
    #     host, DEVICE_TYPE=cpu    3.56 clips/s   283 ms median
    #     docker, CPU image        ~4    clips/s
    #
    # 2.25x, and the host/container difference is nil (3.56 against ~4) - the gain
    # is Metal, not the environment. A SECOND MPS INSTANCE BUYS NOTHING (9.07 vs
    # 8.45 clips/s): they share one GPU and serialise on it. Point this at one
    # server.
    #
    # It generalises past Metal: any Kokoro this script did not start - one on
    # another box, one already warm - works the same way. Note the repo's own warning
    # before reaching for a remote one: adding two REMOTE servers to two local ones
    # measured SLOWER, because batching amortises latency and not the ~640 KB a
    # batch of 16 sends back.
    #
    # From inside the compose network the host is `host.docker.internal`.
    # KOKORO_URL is a protocol spec (tcp://host:port), not an HTTP URL - since the
    # tts-service split every server, local or external, speaks the protocol - so a
    # host engine (the in-process mlx one on a Mac, say) is:
    #
    #     KOKORO_EXTERNAL=1 KOKORO_URL=tcp://host.docker.internal:8900 \
    #         ./src/scripts/run-oww-training.sh "hey seeree"
    if [[ -z "${KOKORO_URL:-}" ]]; then
        echo "ERROR: KOKORO_EXTERNAL=1 but KOKORO_URL is unset." >&2
        echo "       Nothing will be started, so there is nothing to fall back to." >&2
        exit 2
    fi
    echo "=== $(date '+%H:%M:%S')  KOKORO_EXTERNAL=1 - using $KOKORO_URL, starting nothing"

    # Probe from INSIDE the network, not the host. The whole point of this path is
    # that the server is somewhere compose did not put it, so a host-side check can
    # succeed against a URL the trainer cannot resolve - host.docker.internal being
    # exactly that case. The probe speaks the protocol itself (a `voices` op):
    # a server that answers that with the right capabilities is the only kind
    # the corpus generator can use.
    docker compose run --rm --no-deps --entrypoint python3 oww-trainer -c "
import sys
sys.path.insert(0, '/app/src/tts-service/tts_protocol')
from tts_protocol import TtsClient
for spec in '${KOKORO_URL}'.split(','):
    c = TtsClient(spec)
    voices = c.voices()
    print(f'  {spec}: engine={getattr(c, \"server_engine\", None)} {len(voices)} voices')
print('  reachable from the trainer')
"
else
    echo "=== $(date '+%H:%M:%S')  starting Kokoro"
    docker compose up -d kokoro kokoro2

    # Wait for readiness rather than assuming: the GPU image spends a while loading
    # voices, and train.py's probe would otherwise fail on a container that is up but
    # not yet serving. The check is a TCP connect to the PROTOCOL port, not the raw
    # API port: the wrapper's serve() only starts listening after its own available()
    # check passed, so a listener IS the readiness signal.
    for name in kokoro:8899 kokoro2:8901; do
        port="${name##*:}"
        for _ in $(seq 1 60); do
            (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null && { exec 3>&-; break; }
            sleep 2
        done
    done
    echo "=== $(date '+%H:%M:%S')  Kokoro ready"
fi

# WAIT FROM INSIDE THE COMPOSE NETWORK, NOT FROM THE HOST. The piper image now
# runs the same in-process engine as the Mac - the protocol server IS the
# container's main process - so readiness is again a plain connect check, but
# it must still be done from inside the network. A host-side connect to the
# published port is a FALSE POSITIVE: Docker's port proxy accepts as soon as
# the container starts, before the engine has bound the port inside it. That
# was the old failure mode (the Wyoming server bound late, the host check
# passed instantly, and the corpus container died on "Connection refused"),
# and the in-network dialling of piper:8898 is the same check the trainer's
# own client performs - if it connects, the render will too.
if [[ -n "$WANTS_PIPER" ]]; then
    docker compose run --rm --no-deps --entrypoint python3 oww-trainer -c "
import socket, sys, time
for _ in range(90):
    try:
        socket.create_connection(('piper', 8898), 2).close()
        sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit('piper did not start listening on piper:8898 within 180s')
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
# ~8 GiB across the two containers, measured against the public 0.8.1 image (an
# earlier ~2.4 GiB figure here predated it and understated by 3x - hence quoting the
# image).
# Poll the log rather than `tail -f | grep -q`. That pipeline fails SILENTLY in two
# ways, leaving Kokoro running and reproducing the OOM: with pipefail inherited,
# grep -q exiting on a match kills tail -f with SIGPIPE and the pipeline reports
# failure, so the `&&` never runs; and BSD grep buffers stdin, so it may never
# process a line until EOF, which tail -f never sends. A polling loop has neither
# problem.
#
# Not started under SKIP_CORPUS=1: Kokoro was stopped before the run began, so there
# is nothing to wait for.
#
# Not under KOKORO_EXTERNAL=1 either, for a stronger reason than "pointless":
# stopping a server this script did not start is out of bounds - the host MPS
# process is the user's, and on a shared box the URL may be someone else's entirely.
# The VRAM argument for stopping does not apply anyway: an external Kokoro is not
# on the training GPU, which is the whole point of it being external.
WATCH_PID=""
if [[ "${SKIP_CORPUS:-}" != "1" && -z "${KOKORO_EXTERNAL:-}" ]]; then
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
CMD="docker compose run --rm"
# -e OVERRIDES THE SERVICE'S OWN KOKORO_URL, and without it this whole path is inert.
# docker-compose.yml sets KOKORO_URL=tcp://kokoro:8899,tcp://kokoro2:8899 in the
# oww-trainer service, and a value in `environment:` beats the one inherited from the
# shell - so exporting KOKORO_URL alone would be silently ignored and the run would
# dial the containers KOKORO_EXTERNAL=1 deliberately did not start.
[[ -n "${KOKORO_EXTERNAL:-}" ]] && CMD="$CMD -e KOKORO_URL=$(printf '%q' "$KOKORO_URL")"
CMD="$CMD oww-trainer python -m train.oww.train"
# /app/data/external, NOT /app/data. The third-party corpora moved into
# data/external/ and train.py builds rir_paths/background_paths/feature_data_files
# by joining this prefix - so the old value points at directories that no longer
# exist, and openWakeWord augments with no impulse responses and no background audio
# rather than erroring.CMD="$CMD --wake-word $(printf '%q' "$WAKE_WORD") --data-dir /app/data/external"
[[ "${SKIP_CORPUS:-}" == "1" ]] && CMD="$CMD --skip-corpus"
for arg in "$@"; do CMD="$CMD $(printf '%q' "$arg")"; done

# Run under `script` so the container gets a pty. Piping to tee otherwise denies
# docker a TTY, and tqdm has no terminal to draw on - the progress bar disappears
# entirely. `script -e` propagates the child's exit status.
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
# as a failure would discard a good run (README: "TFLite conversion error at end").
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
# train/provenance.py. train.py files the tag itself - corpus half from the
# manifest written this run, config half from the resolved config - so the
# archive and the <tag>.config.json beside it are named by one number. Reading it
# back here rather than recomputing it in the shell is what keeps the two from
# drifting apart; the fallback only runs if the run died before filing it.
TAG="$(cat "output/${SAFE_NAME}/oww/.last_run_tag" 2>/dev/null)"
if [[ -z "$TAG" ]]; then
    echo "    NOTE: no run tag filed by train.py - computing it here (corpus half only)"
    TAG="$(PYTHONPATH=src python3 -m train.provenance --wake-word "$WAKE_WORD" --target oww --tag --fallback "$STAMP")"
fi
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
TAGGED="output/${SAFE_NAME}/oww/${SAFE_NAME}_${TAG}.onnx"
mkdir -p "$(dirname "$TAGGED")"
cp "$MODEL" "$TAGGED"

# The .tflite gets THE SAME TAG, from the same run. train.py converts it straight
# after export, so it is derived from exactly this .onnx - and a tagged .onnx beside
# an untagged .tflite is how a model and its conversion drift apart, the mix-up
# src/eval/src/backends.py warns about when it says the two are not guaranteed to agree.
# Absent if the conversion failed; not fatal - the .onnx is the artifact everything
# else works from.
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
echo "      cd src/eval && docker compose run --rm eval python -m eval.compare_models \\"
echo "          --models <new> <previous-best>"
