#!/usr/bin/env bash
#
# Run a full microWakeWord training pass on the trainer host.
#
# The openWakeWord equivalent is run-oww-training.sh. This one wraps four stages
# rather than one, because mWW builds its corpus, its features and its model as
# separate commands - and each stage silently produces nothing usable if the one
# before it did not run:
#
#   1  train.mww.corpus     WAVs        -> data/corpus/<wake>/mww/{positives,negatives}
#   2  train.mww.features   spectrograms-> data/corpus/<wake>/mww/features/<split>/
#   3  train.mww.train      model       -> output/<wake>/mww/<tag>/
#   4  train.mww.manifest   ESPHome json-> beside the model
#      + collection: the three shipped files, commit-tagged, in output/<wake>/mww/
#
# Usage:
#   ./scripts/run-mww-training.sh "hey seeree"
#   ./scripts/run-mww-training.sh "hey seeree" --training-steps 20000
#
# Extra arguments are passed through to train.mww.train.
#
#   SKIP_BUILD=1     use the existing image (see run-oww-training.sh for when)
#   SKIP_CORPUS=1    reuse data/corpus/<wake>/mww/ - the expensive TTS stage
#   SKIP_FEATURES=1  reuse the spectrograms - only valid if the corpus is unchanged
#   MAX_FAPH=0.2     false-accepts-per-hour budget for the manifest cutoff
#
# WHY NO KOKORO DANCE. run-oww-training.sh stops Kokoro before training because the
# GPU-resident feature patch needs the VRAM its CUDA contexts hold. Nothing here
# does that: mWW's corpus comes from PIPER, which runs CPU-only by design, so it
# holds no CUDA context and can stay up throughout.

set -euo pipefail

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.mww.train args...]" >&2
    exit 2
fi

# A WAKE WORD IS WORDS - same check, same reason, as run-oww-training.sh, where a
# collapsed line continuation passed environment assignments as arguments and a run
# started on "hey seereeKOKORO_EXTERNAL=1". Every path below derives from this
# string. See that script's copy for the full account.
if [[ ! "$WAKE_WORD" =~ ^[A-Za-z][A-Za-z\'’-]*([[:space:]]+[A-Za-z][A-Za-z\'’-]*)*$ ]]; then
    echo "ERROR: '$WAKE_WORD' does not look like a wake word." >&2
    echo "       Expected words only - letters, spaces, apostrophes, hyphens." >&2
    if [[ "$WAKE_WORD" == *=* ]]; then
        echo >&2
        echo "       It contains '='. Environment assignments must come BEFORE the" >&2
        echo "       script; a pasted line continuation often loses them." >&2
    fi
    exit 2
fi
shift

# The REPO ROOT, not this script's directory - every docker compose call below needs
# the compose file in the working directory.
cd "$(dirname "$0")/.."

# tr rather than ${x,,} so this does not need bash 4 (macOS ships 3.2).
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="training-mww-${SAFE_NAME}-${STAMP}.log"
CORPUS="data/corpus/${SAFE_NAME}/mww"
OUT_DIR="output/${SAFE_NAME}/mww"
MAX_FAPH="${MAX_FAPH:-0.2}"

run() { echo "=== $(date '+%H:%M:%S')  $*"; }

# --- image ----------------------------------------------------------------------

if [[ "${SKIP_BUILD:-}" == "1" ]]; then
    run "SKIP_BUILD=1 - using the existing mww-trainer image"
else
    run "building mww-trainer image"
    docker compose build mww-trainer
fi

# --- ambient sets ----------------------------------------------------------------
#
# Discovered rather than hardcoded, so adding a set to data/external/mww_ambient/
# is enough. THE *_eval SET IS NOT OPTIONAL: only it carries validation_ambient and
# testing_ambient, and model selection maximises average_viable_recall computed from
# false accepts per hour on exactly those splits. Without them the metric reads 0.000
# at every step, the best checkpoint never improves on anything, and the exported
# model is whichever happened to be current - while accuracy, recall and precision
# all still look excellent.

AMBIENT=()
for d in data/external/mww_ambient/*/; do
    [[ -d "$d" ]] && AMBIENT+=("${d%/}")
done
if [[ ${#AMBIENT[@]} -eq 0 ]]; then
    echo "ERROR: no ambient sets under data/external/mww_ambient/" >&2
    echo "       ./scripts/download-external-data.sh mww" >&2
    exit 1
fi
HAVE_EVAL=""
for d in "${AMBIENT[@]}"; do
    [[ -d "$d/validation_ambient" || -d "$d/testing_ambient" ]] && HAVE_EVAL=1
done
if [[ -z "$HAVE_EVAL" ]]; then
    echo "ERROR: no ambient set provides validation_ambient/testing_ambient." >&2
    echo "       Model selection cannot work without them - average_viable_recall" >&2
    echo "       would read 0.000 at every step while the model looks fine." >&2
    echo "       The *_eval archives carry those splits." >&2
    exit 1
fi
run "ambient sets: ${AMBIENT[*]}"

# --- 1. corpus -------------------------------------------------------------------

if [[ "${SKIP_CORPUS:-}" == "1" ]]; then
    run "SKIP_CORPUS=1 - reusing $CORPUS"
else
    run "starting Piper"
    docker compose up -d piper

    # WAIT FROM INSIDE THE COMPOSE NETWORK, NOT FROM THE HOST. Piper speaks Wyoming
    # over TCP, so readiness is a connect check - and a host-side connect to
    # localhost:10200 is a FALSE POSITIVE: Docker's port proxy accepts as soon as the
    # container starts, before wyoming-piper has bound the port inside it. That check
    # passed on its first attempt, printed "Piper ready" in the same second as
    # "starting Piper", and the corpus container then died on "Connection refused"
    # dialling piper:10200. Polling the same DNS name the real command uses is the
    # only check that means anything.
    #
    # (Kokoro's check in run-oww-training.sh is unaffected: it issues a real HTTP
    # request, which the proxy cannot answer on the app's behalf.)
    docker compose run --rm --no-deps --entrypoint python3 mww-trainer -c "
import socket, sys, time
for _ in range(90):
    try:
        socket.create_connection(('piper', 10200), 2).close()
        sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit('piper did not start listening on piper:10200 within 180s')
"
    run "Piper ready"

    run "building corpus (log: $LOG)"
    : > "$LOG"
    # --kokoro-fraction 0 is EXPLICIT, not the module default, on purpose: the
    # compose mww-trainer service has no KOKORO_URL and no kokoro service dependency
    # (unlike oww-trainer), so the moment someone wires those in, this line is where
    # the mix should change - and a silent default change here would make container
    # corpora drift from each other without a visible diff. The Apple Silicon route
    # (run-mww-training-applesilicon.sh) already runs 0.3.
    docker compose run --rm mww-trainer python -m train.mww.corpus \
        --wake-word "$WAKE_WORD" --piper-url piper:10200 --kokoro-fraction 0 2>&1 | tee -a "$LOG"
fi

# --- 2. features -----------------------------------------------------------------

if [[ "${SKIP_FEATURES:-}" == "1" ]]; then
    run "SKIP_FEATURES=1 - reusing $CORPUS/features"
else
    # --clean WHENEVER THE CORPUS WAS REBUILT THIS RUN. features.py skips a split
    # whose _mmap directory already exists, which makes it resumable but also means
    # spectrograms built from the PREVIOUS corpus survive a corpus rebuild - and
    # nothing downstream can tell stale features from fresh ones. Training would
    # then quietly use the old audio while the tag names the new.
    CLEAN=(--clean)
    [[ "${SKIP_CORPUS:-}" == "1" ]] && CLEAN=()
    run "generating spectrogram features ${CLEAN[*]:-(reusing what exists)}"
    docker compose run --rm mww-trainer python -m train.mww.features \
        --wake-word "$WAKE_WORD" "${CLEAN[@]}" 2>&1 | tee -a "$LOG"
fi
# --- the run tag -----------------------------------------------------------------
#
# Computed HERE, after the corpus exists and before training starts. Both halves of
# that matter. The corpus is part of the tag (train/provenance.py hashes it), so it
# has to be on disk first; and model_train_eval refuses to train into an existing
# directory, so the tag has to be decided before the run rather than after it. That
# is the opposite of the openWakeWord side, where the corpus is built by the training
# command itself and the tag is therefore computed at the end.
TAG="$(python3 -m train.provenance --wake-word "$WAKE_WORD" --tag --fallback "$STAMP")"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
run "run tag: $TAG"

# --- 3. train --------------------------------------------------------------------

run "training"
docker compose run --rm mww-trainer python -m train.mww.train \
    --wake-word "$WAKE_WORD" --tag "$TAG" \
    --ambient "${AMBIENT[@]}" "$@" 2>&1 | tee -a "$LOG"

RUN_DIR="${OUT_DIR}/${TAG}/tflite_stream_state_internal_quant"
MODEL="${RUN_DIR}/stream_state_internal_quant.tflite"
ROC="${RUN_DIR}/tflite_streaming_roc.txt"

if [[ ! -f "$MODEL" ]]; then
    echo "=== TRAINING FAILED - $MODEL does not exist. See $LOG" >&2
    exit 1
fi

# --- 4. manifest -----------------------------------------------------------------
#
# A manifest failure does NOT fail the run: the model is trained and valid, and the
# cutoff can be chosen later from the ROC that is already on disk. It is separated
# out because choosing probability_cutoff is a judgement, not a build step - the
# default budget of 0.0 has no measured row on most ROCs and is meant to refuse
# rather than silently pick the synthetic frr=1.0 terminator, which is how a
# manifest once shipped with a model that could not fire.

run "cutting the ESPHome manifest (--max-faph $MAX_FAPH)"
if docker compose run --rm mww-trainer python -m train.mww.manifest \
        --wake-word "$WAKE_WORD" --run "$TAG" --max-faph "$MAX_FAPH" 2>&1 | tee -a "$LOG"
then
    MANIFEST="${RUN_DIR}/${SAFE_NAME}.json"
else
    MANIFEST=""
    echo "    NOTE: manifest not written. The model is fine; pick a cutoff from"
    echo "          $ROC and rerun:"
    echo "          docker compose run --rm mww-trainer python -m train.mww.manifest \\"
    echo "              --wake-word \"$WAKE_WORD\" --run $TAG --max-faph <budget>"
fi

# --- collection ------------------------------------------------------------------
#
# The three shipped files, commit-tagged, directly under output/<wake>/mww/ - the
# same shape run-oww-training.sh produces for the .onnx, and what eval/paths.py
# documents as the collected form. The per-run directory stays where it is; this is
# a copy, not a move, because model_train_eval owns that directory's layout.
#
# THE MANIFEST IS REWRITTEN, NOT JUST COPIED. Its "model" key is a bare sibling
# filename - `stream_state_internal_quant.tflite` - so a manifest copied beside a
# renamed model points at a file that is not there. ESPHome would fail to load it,
# and the JSON would look correct while doing so.
TAGGED_MODEL="${OUT_DIR}/${SAFE_NAME}_${TAG}.tflite"
TAGGED_MANIFEST="${OUT_DIR}/${SAFE_NAME}_${TAG}.json"
TAGGED_ROC="${OUT_DIR}/tflite_streaming_roc_${TAG}.txt"

cp "$MODEL" "$TAGGED_MODEL"
[[ -f "$ROC" ]] && cp "$ROC" "$TAGGED_ROC"
if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    python3 - "$MANIFEST" "$TAGGED_MANIFEST" "$(basename "$TAGGED_MODEL")" <<'PY'
import json, sys
src, dst, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
manifest = json.load(open(src))
manifest["model"] = model_name          # must name its sibling, post-rename
json.dump(manifest, open(dst, "w"), indent=2)
open(dst, "a").write("\n")
PY
fi

# --- report ----------------------------------------------------------------------

echo
run "DONE"
echo "    $TAGGED_MODEL  ($(du -h "$TAGGED_MODEL" | cut -f1))"
[[ -f "$TAGGED_MANIFEST" ]] && echo "    $TAGGED_MANIFEST  (model: $(basename "$TAGGED_MODEL"))"
[[ -f "$TAGGED_ROC" ]] && echo "    $TAGGED_ROC"
[[ -n "$DIRTY" ]] && echo "    NOTE: working tree was dirty - the code half of $TAG is not reproducible"
echo
echo "    Confirm the cutoff against held-out recordings before deploying it - the"
echo "    ROC is scored on ambient sets, not on this repo's adversarial negatives:"
echo "      docker compose run --rm eval python -m eval.compare_models \\"
echo "          --models $TAGGED_MANIFEST"
