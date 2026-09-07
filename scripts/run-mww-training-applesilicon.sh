#!/usr/bin/env bash
#
# Run the microWakeWord trainer on the HOST, on Apple Silicon. The container-free
# counterpart to run-mww-training.sh: the same four stages, the same
# train/mww/*.py, the same pinned tensorflow 2.21.0 - with the corpus stage's
# Piper on the host instead of in the compose network.
#
# WHY IT IS A SEPARATE SCRIPT rather than a flag on run-mww-training.sh: that
# script is docker-compose orchestration - a network that does not exist on a
# Mac, a piper service name that does not resolve, an image build to wait for.
# Threading a host mode through it would leave both paths harder to read than
# two scripts, the same reasoning the openWakeWord pair documents.
#
# WHY IT IS FASTER THAN THE CONTAINER, MEASURED (2026-09-07, this machine):
#     stage        container            host
#     corpus       9.13 clips/s Piper   21.66 clips/s Piper (2.4x)
#     training     36.21 ms/step        30.86 ms/step (1.17x)
# The corpus stage is the longest in a full run, so it carries most of the
# end-to-end gain. tools/tf_probe.py and tools/bench_tts.py are the probes;
# the numbers and their caveats are in apple-port.md, phase 3.
#
#     ./scripts/setup-mww-applesilicon-trainer.sh                     # once
#     ./scripts/start-piper-host.sh                                   # in another terminal
#     ./scripts/run-mww-training-applesilicon.sh "hey seeree"
#
# SKIP_CORPUS=1 / SKIP_FEATURES=1 behave exactly as in run-mww-training.sh.
#
# The corpus stage needs a Piper server. This script starts nothing: the host
# one is scripts/start-piper-host.sh (uv venv, 127.0.0.1:10200). PIPER_URL or
# --piper-url reach any other one; a piper:PORT value is rewritten to
# 127.0.0.1:PORT, because the compose service name resolves to nothing here.

set -euo pipefail

cd "$(dirname "$0")/.."
ENV_DIR="train-mww-applesilicon"
CLONE="microwakeword"
# Same pin as scripts/setup-mww-applesilicon-trainer.sh - one value, two files,
# keep them in lockstep when the fork moves.
MWW_COMMIT="4665173cd35f1cff9a61e06fc427f124766c488e"

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [extra train.py args...]" >&2
    exit 2
fi

# Same guard as run-mww-training.sh, same reason: a collapsed line continuation
# once passed environment assignments as arguments and a run trained on the
# wake word "hey seereeMWW_REF=main".
if [[ ! "$WAKE_WORD" =~ ^[A-Za-z][A-Za-z\'’-]*([[:space:]]+[A-Za-z][A-Za-z\'’-]*)*$ ]]; then
    echo "ERROR: '$WAKE_WORD' does not look like a wake word." >&2
    echo "       Expected words only - letters, spaces, apostrophes, hyphens." >&2
    exit 2
fi
shift

# THE ENVS ARE SPLIT ON NUMPY, so is the failure mode. If the microwakeword
# package is missing from THIS venv, the run dies with ModuleNotFoundError in
# the features stage; if numpy is <2, model_train_eval's RaggedMmap path dies
# later still. Both are setup-script territory, so say that here, now, in the
# message a tired reader will actually read.
if [[ ! -x "$ENV_DIR/.venv/bin/python" ]]; then
    echo "ERROR: $ENV_DIR/.venv missing. Run ./scripts/setup-mww-applesilicon-trainer.sh" >&2
    exit 2
fi
"$ENV_DIR/.venv/bin/python" - <<'PY' || exit 2
import sys, numpy, microwakeword  # noqa: F401
if numpy.__version__ < "2":
    print(f"numpy {numpy.__version__} in {sys.executable} - mWW needs >=2.", file=sys.stderr)
    print("A different venv is probably activated; re-run the setup script.", file=sys.stderr)
    sys.exit(1)
PY

# The clone's HEAD is the sentinel, the way the oww run script checks for its
# applied patches: a `git pull` or a checkout inside microwakeword/ moves what
# the editable install resolves to, and the failure would surface two stages
# in, after the corpus is already generated. Read-only; the repair is the
# setup script, which is idempotent.
if [[ ! -d "$CLONE/.git" ]]; then
    echo "ERROR: no $CLONE/ clone. Run ./scripts/setup-mww-applesilicon-trainer.sh" >&2
    exit 2
fi
if [[ "$(git -C "$CLONE" rev-parse HEAD)" != "$MWW_COMMIT" ]]; then
    echo "ERROR: $CLONE/ is at $(git -C "$CLONE" rev-parse --short HEAD), not $MWW_COMMIT." >&2
    echo "       The venv's editable install resolves to whatever HEAD is. Re-run" >&2
    echo "       ./scripts/setup-mww-applesilicon-trainer.sh (idempotent) to pin it," >&2
    echo "       then start this run again." >&2
    exit 2
fi

# PIPER_URL
#
# Same contract as the oww host script: this pipeline is 100% Piper (there is
# no Kokoro half of the mWW corpus yet - train/mww/corpus.py documents the gap),
# so unlike there the probe is unconditional unless SKIP_CORPUS.
#
# piper:PORT is a COMPOSE-ONLY name: it resolves inside the compose network
# and to nothing from a host process. The intent is unambiguous - the same
# server, reached locally - so rewrite rather than fail. Any other
# host:port (a Piper on the LAN, a box on the network) passes through.
PIPER_PORT_DEFAULT=10200
if [[ "${PIPER_URL:-}" == piper:* ]]; then
    PIPER_PORT="${PIPER_URL#piper:}"
    [[ -z "$PIPER_PORT" || ! "$PIPER_PORT" =~ ^[0-9]+$ ]] && PIPER_PORT=$PIPER_PORT_DEFAULT
    PIPER_URL="127.0.0.1:${PIPER_PORT}"
    export PIPER_URL
    echo "=== note: rewrote PIPER_URL to $PIPER_URL - the compose service name"
    echo "          only resolves inside the compose network. For a local server:"
    echo "          ./scripts/start-piper-host.sh"
elif [[ -z "${PIPER_URL:-}" ]]; then
    PIPER_URL="127.0.0.1:${PIPER_PORT_DEFAULT}"
    export PIPER_URL
fi

# THE CONTAINER MAY OWN THESE FILES. Both paths write data/corpus/ and output/,
# and the trainer images run as root - train/ownership.py hands output/ back
# afterwards, but data/corpus/ is left as root wrote it. A host run then fails
# on permissions somewhere unhelpful, so check here where the fix is obvious.
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
for d in "data/corpus/${SAFE_NAME}/mww" "output/${SAFE_NAME}/mww"; do
    if [[ -e "$d" && ! -w "$d" ]]; then
        echo "ERROR: $d is not writable by $(whoami) - a container run probably made it." >&2
        echo "       sudo chown -R \"$(whoami)\" $d" >&2
        exit 2
    fi
done

# A dead or non-Wyoming Piper fails the corpus stage at clip 1 of thousands,
# and the log does not say why. A Describe round trip, not a TCP connect: a
# bound port owned by a dead or non-Wyoming listener passes a connect check
# and still fails the corpus stage. This asks the question the corpus stage
# asks.
if [[ "${SKIP_CORPUS:-}" != "1" ]]; then
    if ! "$ENV_DIR/.venv/bin/python" - "$PIPER_URL" <<'PYEOF'
import sys
sys.path.insert(0, ".")
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
        echo "  first render, not now. Start the host server in another terminal:" >&2
        echo "  ./scripts/start-piper-host.sh   (it downloads voices on first use)" >&2
        echo "  or point PIPER_URL / --piper-url at an existing one." >&2
        exit 1
    fi
fi

run() {
    echo
    echo "=== $(date '+%H:%M:%S')  $1"
    shift
    set +e
    "$@" 2>&1 | tee -a "$LOG"
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "=== $(date '+%H:%M:%S')  FAILED ($1 exited $rc); see $LOG" >&2
        exit $rc
    fi
}

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="training-${SAFE_NAME}-macos-${STAMP}.log"

# === 1. corpus ====================================================================
#
# The one stage this script reaches differently: PIPER_URL is the host server,
# never the compose service. generate_piper_samples here is the same code the
# container runs - the 2.4x comes from the server process, not the client.
if [[ "${SKIP_CORPUS:-}" == "1" ]]; then
    echo
    echo "=== $(date '+%H:%M:%S')  corpus (skipped)"
else
    run "corpus (Piper ${PIPER_URL})" \
        "$ENV_DIR/.venv/bin/python" -m train.mww.corpus \
            --wake-word "$WAKE_WORD" --piper-url "$PIPER_URL" --piper-speakers 12
fi

# === 2. features ===================================================================
#
# The --clean decision is the container script's, unchanged: corrupt on-disk
# state fails in the middle of a run, and the only question worth asking is
# whether the corpus changed since the features were built.
FEATURES_DIR="data/corpus/${SAFE_NAME}/mww/features"
if [[ "${SKIP_FEATURES:-}" == "1" ]]; then
    echo
    echo "=== $(date '+%H:%M:%S')  features (skipped)"
elif [[ ! -d "$FEATURES_DIR" ]]; then
    run "features" \
        "$ENV_DIR/.venv/bin/python" -m train.mww.features \
            --wake-word "$WAKE_WORD" --clean
else
    NEWEST_CORPUS="$(find "data/corpus/${SAFE_NAME}/mww" -path "*/features" -prune -o -type f -newer "$FEATURES_DIR" -print -quit 2>/dev/null || true)"
    if [[ -n "$NEWEST_CORPUS" ]]; then
        echo "=== $(date '+%H:%M:%S')  corpus changed since features - rebuilding"
        run "features" \
            "$ENV_DIR/.venv/bin/python" -m train.mww.features \
                --wake-word "$WAKE_WORD" --clean
    else
        echo
        echo "=== $(date '+%H:%M:%S')  features (corpus unchanged)"
    fi
fi

# === 3. train ======================================================================
#
# The tag is computed HERE, after the corpus exists and before training starts -
# the same reason run-mww-training.sh documents: the corpus is part of the tag, and
# model_train_eval refuses to train into an existing directory. The checksum guard
# in train/mww/train.py still applies - it is in the code, not in the shell.
TAG="$("$ENV_DIR/.venv/bin/python" -m train.provenance --wake-word "$WAKE_WORD" --tag --fallback "$STAMP")"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
run "run tag: $TAG"

# The ambient sets live under data/external/mww_ambient/ (download-external-data.sh),
# in the same directory as the augmentation impulse/background sets the features
# stage already used. Pass every subdirectory present: mWW's model_train_eval
# treats them as separate ambient sets and reports per-set false accepts.
AMBIENT_ARGS=()
if [[ -d data/external/mww_ambient ]]; then
    while IFS= read -r dir; do
        AMBIENT_ARGS+=("$dir")
    done < <(find data/external/mww_ambient -mindepth 1 -maxdepth 1 -type dir | sort)
fi

run "training"
set +e
"$ENV_DIR/.venv/bin/python" -m train.mww.train \
    --wake-word "$WAKE_WORD" --tag "$TAG" \
    --ambient "${AMBIENT_ARGS[@]+"${AMBIENT_ARGS[@]}"}" "$@" 2>&1 | tee -a "$LOG"
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
    echo "=== $(date '+%H:%M:%S')  TRAINING FAILED (exit $STATUS). See $LOG" >&2
    exit $STATUS
fi

OUT_DIR="output/${SAFE_NAME}/mww"
RUN_DIR="${OUT_DIR}/${TAG}/tflite_stream_state_internal_quant"
MODEL="${RUN_DIR}/stream_state_internal_quant.tflite"
ROC="${RUN_DIR}/tflite_streaming_roc.txt"

# A zero exit with no model file is the failure mode the tag exists to catch: an
# earlier run's directory, a crashed conversion. The container script checks
# exactly this.
if [[ ! -f "$MODEL" ]]; then
    echo "=== TRAINING FAILED - $MODEL does not exist. See $LOG" >&2
    exit 1
fi

# === 4. manifest ===================================================================
#
# The same stage with the same non-fatal semantics as run-mww-training.sh: a
# manifest failure does NOT fail the run, because choosing probability_cutoff is
# a judgement, not a build step. Default budget 0.2 there, same here.
MAX_FAPH="${MAX_FAPH:-0.2}"
run "cutting the ESPHome manifest (--max-faph $MAX_FAPH)"
MANIFEST=""
if "$ENV_DIR/.venv/bin/python" -m train.mww.manifest \
        --wake-word "$WAKE_WORD" --run "$TAG" --max-faph "$MAX_FAPH" 2>&1 | tee -a "$LOG"
then
    MANIFEST="${RUN_DIR}/${SAFE_NAME}.json"
else
    echo "    NOTE: manifest not written. The model is fine; pick a cutoff from"
    echo "          $ROC and rerun:"
    echo "          $ENV_DIR/.venv/bin/python -m train.mww.manifest \\"
    echo "              --wake-word \"$WAKE_WORD\" --run $TAG --max-faph <budget>"
fi

# === collect =======================================================================
#
# The three shipped files, commit-tagged, in output/<wake>/mww/ - the same shape
# run-oww-training.sh produces and what eval/paths.py documents. Verbatim from
# run-mww-training.sh, including the manifest rewrite: its "model" key is a bare
# sibling filename, so a manifest copied beside a renamed model points at a file
# that is not there.
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

# === report =========================================================================

echo
run "DONE  (host trainer)"
echo "    $TAGGED_MODEL  ($(du -h "$TAGGED_MODEL" | cut -f1))"
[[ -f "$TAGGED_MANIFEST" ]] && echo "    $TAGGED_MANIFEST  (model: $(basename "$TAGGED_MODEL"))"
[[ -f "$TAGGED_ROC" ]] && echo "    $TAGGED_ROC"
[[ -n "$DIRTY" ]] && echo "    NOTE: working tree was dirty - the code half of $TAG is not reproducible"
echo
echo "    Compare the wall time against the container's 26m06s (this machine, 2026-09-06)"
echo "    and the per-stage numbers against tools/tf_probe.py and tools/bench_tts.py."
echo
echo "    Evaluate:  ./scripts/eval-models.sh   (Docker, unchanged) - confirm the cutoff"
echo "    against held-out recordings before deploying it."
