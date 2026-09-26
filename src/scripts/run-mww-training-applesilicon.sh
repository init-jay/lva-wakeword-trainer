#!/usr/bin/env bash
#
# Run the microWakeWord trainer on the HOST, on Apple Silicon. 
#
#
#     ./src/scripts/setup-mww-applesilicon-trainer.sh                     # once
#     uv run --project src/tts-service/engines/piper python -m piper_engine --port 8898   # another terminal
#     uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900  # another, for the default 30% mix
#     ./src/scripts/run-mww-training-applesilicon.sh "hey seeree"
#
# KOKORO_FRACTION (default 0.3, or --kokoro-fraction on the command line) is the
# share of the PHRASE-ALONE positive budget Kokoro renders instead of Piper -
# substitution, not addition: the total clip count, the real-clip share, and the
# Piper-only negative set all stay fixed, mirroring the 30% its openWakeWord
# sibling already runs (engines swapped). 0.3 needs the Kokoro engine above;
# KOKORO_FRACTION=0 (or --kokoro-fraction 0) runs all-Piper, the historical corpus,
# and needs only Piper. The module document: train/mww/corpus.py.
#
# --samples-per-voice N (default 60, the corpus module's) sets the corpus DEPTH
# and is consumed by this script, applied to the corpus stage only - the train
# stage does not take it and would reject it. Scaling the data the model actually
# SEES needs --training-steps too (that one flows to the train stage): steps
# default to 10000, so a 2x corpus at 10000 steps trains on the same total number
# of examples and tests nothing about data size.
#
# --negatives-per-voice N (default 12, the corpus module's) sets how many
# adversarial clips each Piper voice renders - the REJECTION training, as opposed
# to the positive budget above. The 2026-09-08 doubled-depth runs showed what an
# underfed rejection set does: with 984 adversarial clips against 8-15k positives,
# one of the two produced a firehose that fired on everything Piper-ish and lost
# its FAPH operating point entirely. Doubling this to 24 (1,968 clips, ~2 minutes
# of TTS) is the cheap, single-variable counter-test; the measurement lives in
# train/mww/corpus.py, point 5.
#
# SKIP_CORPUS=1 / SKIP_FEATURES=1 behave exactly as in run-mww-training.sh.
#
# The corpus stage needs a Piper engine, and - at the default 30% mix - a Kokoro
# one too. This script starts nothing: the engines are the uv projects in
# src/tts-service/engines/ (commands above) - in-process piper-tts on 8898 and
# in-process kokoro-mlx on 8900, both speaking the TTS protocol. PIPER_URL /
# KOKORO_URL or --piper-url / --kokoro-url reach any other tcp:// server; a bare
# host:port value is coerced to tcp:// (the protocol client accepts only that
# form), because the old raw forms used to mean different backends with
# different audio.
#
# PIPER FLEET - the fast path for the corpus stage. One Piper instance is one
# serial lane: the engine holds one model resident and takes every call under
# one lock, so client threads queue instead of run, and the single instance
# measures 21.66 clips/s in src/scripts/bench_tts.py, against the ~16 the mww corpus
# stage ran at against it (improvement.md P2.1 - the corpus stage is the
# serial wall of a Mac run, ~5 of its ~14 measured minutes). Throughput scales
# with PROCESSES: PIPER_URLS takes the comma-joined list scripts/start-tts-fleet.sh
# prints (it starts N instances on 8898+ in the background and waits for each
# voices round trip), and the corpus shards the fleet BY VOICE - each model
# pinned to one instance for the whole run, so an instance loads each of its
# models once, not per request (corpus/piper.py, PiperFleet):
#
#     PIPER_URLS="$(./src/scripts/start-tts-fleet.sh 4)" \
#         ./src/scripts/run-mww-training-applesilicon.sh "hey seeree"
#
# One server stays PIPER_URL, unchanged; PIPER_URLS wins over it when both are
# set, because a comma list is an explicit statement and a bare PIPER_URL left
# exported from another context is not.
#
# SMOKE=1: a few-minute end-to-end check that a changed train/ tree still runs the
# whole pipeline: the corpus through the --skip path (no TTS servers; a
# pre-manifest corpus is reused as-is), the pre-built features, 200-step
# training, real tflite conversion. It implies SKIP_CORPUS=1 and appends --smoke
# to both the tag computation and the train stage, so both resolve the same
# smoke-<stamp> run directory (train/mww/train.py --smoke). The smoke-named model
# files in output/<wake>/mww/ cannot be mistaken for a real run's archive:
#
#     SMOKE=1 ./src/scripts/run-mww-training-applesilicon.sh "hey seeree"

set -euo pipefail

cd "$(dirname "$0")/../../"
ENV_DIR="src/train/train-mww-applesilicon"
CLONE="microwakeword"
CLONE_DIR="src/train/microwakeword"
# Same pin as scripts/setup-mww-applesilicon-trainer.sh - one value, two files,
# keep them in lockstep when the fork moves.
MWW_COMMIT="4665173cd35f1cff9a61e06fc427f124766c488e"

WAKE_WORD="${1:-}"
if [[ -z "$WAKE_WORD" ]]; then
    echo "usage: $0 \"wake word\" [--kokoro-fraction F] [--samples-per-voice N] [--negatives-per-voice N] [extra train.py args...]" >&2
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
# package is missing from THIS venv, the run dies with ModuleNotFoundError in the
# features stage; if numpy is <2, model_train_eval's RaggedMmap path dies later
# still. Both are setup-script territory, so say that here, now, in the message a
# tired reader will actually read.
if [[ ! -x "$ENV_DIR/.venv/bin/python" ]]; then
    echo "ERROR: $ENV_DIR/.venv missing. Run ./src/scripts/setup-mww-applesilicon-trainer.sh" >&2
    exit 2
fi
"$ENV_DIR/.venv/bin/python" - <<'PY' || exit 2
import sys, numpy, microwakeword  # noqa: F401
# __file__ is None when the clone ROOT shadows the package as a namespace
# package (no editable install): the import above passes and the failure
# surfaces two stages in, after the corpus TTS. 2026-09-22 smoke run.
assert microwakeword.__file__, "microwakeword resolved as a namespace package"
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
if [[ ! -d "$CLONE_DIR/.git" ]]; then
    echo "ERROR: no $CLONE_DIR/ clone. Run ./src/scripts/setup-mww-applesilicon-trainer.sh" >&2
    exit 2
fi
if [[ "$(git -C "$CLONE_DIR" rev-parse HEAD)" != "$MWW_COMMIT" ]]; then
    echo "ERROR: $CLONE/ is at $(git -C "$CLONE_DIR" rev-parse --short HEAD), not $MWW_COMMIT." >&2
    echo "       The venv's editable install resolves to whatever HEAD is. Re-run" >&2
    echo "       ./src/scripts/setup-mww-applesilicon-trainer.sh (idempotent) to pin it," >&2
    echo "       then start this run again." >&2
    exit 2
fi

# The pin above does NOT cover the patch. per-clip-stream-reset.py is a
# working-tree edit at the pinned commit, so the same `git checkout .` in the
# clone that moved nothing at HEAD undoes it - and the in-run streaming ROC
# then silently measures the persistent-stream condition again (the blind
# condition data/lva-mww-cause/report.md documents), a failure that surfaces
# only at the gate. The oww run script guards its patches this way; do the
# same. Read-only; the repair is the setup script, which re-applies.
if ! grep -q 'model = Model(model_path, stride=stride)' "$CLONE_DIR/microwakeword/test.py"; then
    echo "ERROR: the microWakeWord clone is missing its per-clip-stream-reset patch -" >&2
    echo "       the working tree was probably reset (e.g. a git checkout in $CLONE/)." >&2
    echo "       Re-run ./src/scripts/setup-mww-applesilicon-trainer.sh (idempotent)" >&2
    echo "       to re-apply it, then start this run again." >&2
    exit 2
fi

# KOKORO_FRACTION: the share of the PHRASE-ALONE budget Kokoro renders instead
# of Piper (see the header). It is consumed HERE, not passed to the training
# stage, because the corpus and train stages are separate processes and only
# the corpus takes it. KOKORO_FRACTION on the environment wins over nothing -
# a command-line --kokoro-fraction wins over the environment, which wins over
# the 0.3 default, the same precedence the oww script gives its --piper-fraction.
# --samples-per-voice is consumed here for the same reason: the train stage
# does not take it and would reject it.
KOKORO_FRACTION="${KOKORO_FRACTION:-0.3}"
SAMP_PER_VOICE=""
NEGATIVES_PER_VOICE=""
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kokoro-fraction)
            [[ $# -ge 2 ]] || { echo "ERROR: --kokoro-fraction needs a value" >&2; exit 2; }
            KOKORO_FRACTION="$2"; shift 2 ;;
        --kokoro-fraction=*)
            KOKORO_FRACTION="${1#*=}"; shift ;;
        --samples-per-voice)
            [[ $# -ge 2 ]] || { echo "ERROR: --samples-per-voice needs a value" >&2; exit 2; }
            SAMP_PER_VOICE="$2"; shift 2 ;;
        --samples-per-voice=*)
            SAMP_PER_VOICE="${1#*=}"; shift ;;
        --negatives-per-voice)
            [[ $# -ge 2 ]] || { echo "ERROR: --negatives-per-voice needs a value" >&2; exit 2; }
            NEGATIVES_PER_VOICE="$2"; shift 2 ;;
        --negatives-per-voice=*)
            NEGATIVES_PER_VOICE="${1#*=}"; shift ;;
        *)
            TRAIN_ARGS+=("$1"); shift ;;
    esac
done
if [[ -n "$SAMP_PER_VOICE" && ! "$SAMP_PER_VOICE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --samples-per-voice must be a positive integer, got $SAMP_PER_VOICE." >&2
    exit 2
fi
if [[ -n "$NEGATIVES_PER_VOICE" && ! "$NEGATIVES_PER_VOICE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --negatives-per-voice must be a positive integer, got $NEGATIVES_PER_VOICE." >&2
    exit 2
fi
if [[ ! "$KOKORO_FRACTION" =~ ^([0-9]+(\.[0-9]+)?|\.[0-9]+)$ ]]; then
    echo "ERROR: KOKORO_FRACTION='$KOKORO_FRACTION' is not a number." >&2
    exit 2
fi
if awk -v f="$KOKORO_FRACTION" 'BEGIN { exit !(f >= 0.0 && f < 1.0) }'; then
    : # [0, 1) - 1 is excluded on purpose: the negatives are Piper-only, so Piper
    # must stay in the corpus (train/mww/corpus.py enforces the same bound).
else
    echo "ERROR: KOKORO_FRACTION must be in [0, 1) - got $KOKORO_FRACTION." >&2
    echo "       (1 is not allowed: the adversarial negatives are Piper-only.)" >&2
    exit 2
fi
# "0" and "0.0" both mean off; normalize so the probes and labels can test one
# value. 0 = the historical all-Piper corpus, and the only setting that needs
# no Kokoro server at all.
if [[ "$KOKORO_FRACTION" == "0" || "$KOKORO_FRACTION" == "0.0" ]]; then
    KOKORO_FRACTION=""
fi
set -- "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"

# SMOKE=1 (header): the corpus stage goes through --skip and --smoke is added to
# the train stage. It must land BEFORE the TTS probes (they are gated on
# SKIP_CORPUS - a smoke run needs no servers) and BEFORE the tag is computed
# (--print-tag has to resolve the same smoke-<stamp> directory the train stage
# will claim, or the run would die on 'directory already exists').
if [[ "${SMOKE:-}" == "1" ]]; then
    SKIP_CORPUS=1
    set -- "$@" --smoke
fi

# PIPER_URL / PIPER_URLS
#
# PIPER_URL: same contract as the oww host script: this is the protocol URL of
# the in-process Piper engine (src/tts-service/engines/piper, port 8898). The old
# piper:PORT compose form is rewritten - the compose service name resolves to
# nothing from a host process - and any bare host:port is coerced to tcp://,
# the only form the protocol client accepts.
#
# PIPER_URLS: a comma-separated list of them (start-tts-fleet.sh prints
# exactly this). The corpus shards it BY VOICE (see the header), and every
# element is normalised the same way as a PIPER_URL below.
PIPER_PORT_DEFAULT=8898
PIPER_RAW="${PIPER_URLS:-${PIPER_URL:-}}"
if [[ -z "$PIPER_RAW" ]]; then
    PIPER_RAW="127.0.0.1:${PIPER_PORT_DEFAULT}"
fi
IFS=',' read -r -a PIPER_PARTS <<< "$PIPER_RAW"
PIPER_URL=""
for part in "${PIPER_PARTS[@]}"; do
    part="${part// /}"
    [[ -z "$part" ]] && continue
    if [[ "$part" == piper:* ]]; then
        PIPER_PORT="${part#piper:}"
        [[ -z "$PIPER_PORT" || ! "$PIPER_PORT" =~ ^[0-9]+$ ]] && PIPER_PORT=$PIPER_PORT_DEFAULT
        part="127.0.0.1:${PIPER_PORT}"
        echo "=== note: rewrote PIPER_URL(S) element to tcp://$part - the compose service name"
        echo "          only resolves inside the compose network. For a local engine:"
        echo "          uv run --project src/tts-service/engines/piper python -m piper_engine --port $PIPER_PORT"
    fi
    [[ "$part" != tcp://* ]] && part="tcp://$part"
    PIPER_URL="${PIPER_URL:+$PIPER_URL,}$part"
done
[[ -n "$PIPER_URL" ]] || { echo "ERROR: PIPER_URL / PIPER_URLS resolved to nothing ($PIPER_RAW)." >&2; exit 2; }
export PIPER_URL

# KOKORO_URL: same treatment, with one exception. The Mac's Kokoro is the
# in-process kokoro-mlx engine (src/tts-service/engines/kokoro_mlx, port 8900);
# the old mlx:// form named that same engine, so it is still coerced. The old
# http:// form is NOT: it meant a raw Kokoro-FastAPI host server, which does
# not speak the protocol - coercing it to tcp:// just moved the failure from
# connect time to every single render. Reject it and name the two real
# options instead of guessing a port.
KOKORO_URL="${KOKORO_URL:-tcp://127.0.0.1:8900}"
if [[ "${KOKORO_URL}" == *host.docker.internal* ]]; then
    KOKORO_URL="${KOKORO_URL//host.docker.internal/127.0.0.1}"
    echo "=== note: rewrote KOKORO_URL to $KOKORO_URL - host.docker.internal"
    echo "          only resolves inside a container."
fi
if [[ "${KOKORO_URL}" == http://* ]]; then
    echo "ERROR: KOKORO_URL=${KOKORO_URL} is a raw http:// Kokoro-FastAPI URL." >&2
    echo "       The protocol client speaks only tcp://, and a raw FastAPI port" >&2
    echo "       does not speak the protocol - no port rewrite can fix that." >&2
    echo "       Point KOKORO_URL at one of:" >&2
    echo "         tcp://127.0.0.1:8900   the mlx engine, in-process on this Mac" >&2
    echo "           (uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900)" >&2
    echo "         tcp://<box>:8899       the Docker kokoro wrapper (docker-compose.yml)" >&2
    exit 1
fi
if [[ "${KOKORO_URL}" == mlx://* ]]; then
    KOKORO_URL="tcp://${KOKORO_URL#*//}"
    echo "=== note: rewrote KOKORO_URL to $KOKORO_URL - mlx:// named the mlx"
    echo "          engine, which is now the protocol server on 8900"
fi
[[ "${KOKORO_URL}" != tcp://* ]] && KOKORO_URL="tcp://${KOKORO_URL}"
export KOKORO_URL

# THE CONTAINER MAY OWN THESE FILES. Both paths write data/corpus/ and output/,
# and the trainer images run as root - train/ownership.py hands output/ back
# afterwards, but data/corpus/ is left as root wrote it. A host run then fails on
# permissions somewhere unhelpful, so check here where the fix is obvious.
SAFE_NAME="$(printf '%s' "$WAKE_WORD" | tr ' [:upper:]' '_[:lower:]')"
for d in "data/corpus/${SAFE_NAME}/mww" "output/${SAFE_NAME}/mww"; do
    if [[ -e "$d" && ! -w "$d" ]]; then
        echo "ERROR: $d is not writable by $(whoami) - a container run probably made it." >&2
        echo "       sudo chown -R \"$(whoami)\" $d" >&2
        exit 2
    fi
done

# Dead servers fail the corpus stage at clip 1 of thousands, and the log does
# not say why. Both probes are the round trip the corpus stage actually makes,
# not a TCP connect: a bound port owned by a dead or foreign listener passes a
# connect check and still fails the corpus stage. Piper is probed when it will
# render anything (always, at any fraction in [0,1)) - EVERY URL of a PIPER_URLS
# fleet, since the corpus sends each voice to a different one; Kokoro when it will.
if [[ "${SKIP_CORPUS:-}" != "1" ]]; then
    if ! "$ENV_DIR/.venv/bin/python" - "$PIPER_URL" <<'PYEOF'
import sys
sys.path.insert(0, "src")
import train.corpus  # noqa: F401  (sys.path bootstrap for tts_protocol)
from tts_protocol.client import TtsClient
urls = [u.strip() for u in sys.argv[1].split(",") if u.strip()]
bad = []
for url in urls:
    try:
        pairs = TtsClient(url).voices(languages=["en_US", "en_GB"], max_speakers=0)
    except Exception as e:
        print(f"  Piper unreachable at {url}: {e}", file=sys.stderr)
        bad.append(url)
        continue
    print(f"  Piper probe OK: {len(pairs)} (voice, speaker) pairs at {url}")
sys.exit(1 if bad else 0)
PYEOF
    then
        echo "  No reachable Piper at $PIPER_URL - the corpus stage would fail at its" >&2
        echo "  first render, not now. Start the engine in another terminal:" >&2
        echo "  uv run --project src/tts-service/engines/piper python -m piper_engine --port 8898" >&2
        echo "  (it uses the voices under data/external/piper) or point PIPER_URL / --piper-url at an existing one." >&2
        echo "  A fleet:  PIPER_URLS=\"\$(./src/scripts/start-tts-fleet.sh 4)\"" >&2
        exit 1
    fi

    if [[ -n "${KOKORO_FRACTION:-}" ]]; then
        if ! "$ENV_DIR/.venv/bin/python" - "$KOKORO_URL" <<'PYEOF'
import sys
sys.path.insert(0, "src")
import train.corpus  # noqa: F401  (sys.path bootstrap for tts_protocol)
from tts_protocol.client import TtsClient
url = sys.argv[1]
try:
    c = TtsClient(url)
    voices = c.voices()
except Exception as e:
    print(f"  Kokoro unreachable: {e}", file=sys.stderr)
    sys.exit(1)
engine = c.server_engine or "?"
ts = "word timestamps yes" if c.supports_timestamps else "word timestamps NO"
print(f"  Kokoro probe OK: engine={engine}, {len(voices)} voices, {ts} at {url}")
PYEOF
        then
            echo "  No reachable Kokoro at $KOKORO_URL - the corpus stage would fail at" >&2
            echo "  its first Kokoro render, not now. Start the engine in another" >&2
            echo "  terminal:  uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900" >&2
            echo "  or run all-Piper:  KOKORO_FRACTION=0  (or --kokoro-fraction 0)" >&2
            exit 1
        fi
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
mkdir -p src/logs
LOG="src/logs/training-${SAFE_NAME}-macos-${STAMP}.log"

# === 1. corpus ====================================================================
#
# The one stage this script reaches differently: PIPER_URL and KOKORO_URL are
# host servers, never compose service names. generate_piper_samples and
# generate_kokoro_samples here are the same code the container runs - the 2.4x
# Piper gap comes from the server process, not the client.
#
# The optional depth overrides go through an array: a double-quoted
# "${SAMP_PER_VOICE:+--samples-per-voice $SAMP_PER_VOICE}" is ONE word to bash -
# no splitting inside the quotes - and argparse rejects it, as this script's
# first 820-voice run found out.
CORPUS_EXTRA=()
[[ -n "$SAMP_PER_VOICE" ]] && CORPUS_EXTRA+=(--samples-per-voice "$SAMP_PER_VOICE")
[[ -n "$NEGATIVES_PER_VOICE" ]] && CORPUS_EXTRA+=(--negatives-per-voice "$NEGATIVES_PER_VOICE")
if [[ "${SKIP_CORPUS:-}" == "1" ]]; then
    # --skip validates the corpus.json manifest against the shaping this stage
    # would request and exits with a diff on a mismatch - SKIP_CORPUS used to be
    # a blind reuse, and a corpus built with different depth than the one a
    # resumed run thinks it asked for is the measurement that goes wrong
    # silently. Same flags the build branch would use (CORPUS_EXTRA included),
    # because those are exactly what the check has to see.
    SKIP_CORPUS_ARGS=(--wake-word "$WAKE_WORD" --piper-url "$PIPER_URL" --piper-speakers 12 --skip)
    [[ -n "${KOKORO_FRACTION:-}" ]] && SKIP_CORPUS_ARGS+=(--kokoro-url "$KOKORO_URL" --kokoro-fraction "$KOKORO_FRACTION")
    run "verifying existing corpus (data/corpus/${SAFE_NAME}/mww)" \
        env PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.corpus \
            "${SKIP_CORPUS_ARGS[@]}" "${CORPUS_EXTRA[@]+"${CORPUS_EXTRA[@]}"}"
elif [[ -n "${KOKORO_FRACTION:-}" ]]; then
    run "corpus (Piper ${PIPER_URL} + Kokoro ${KOKORO_URL}, fraction ${KOKORO_FRACTION})" \
        env PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.corpus \
            --wake-word "$WAKE_WORD" --piper-url "$PIPER_URL" --piper-speakers 12 \
            --kokoro-url "$KOKORO_URL" --kokoro-fraction "$KOKORO_FRACTION" \
            "${CORPUS_EXTRA[@]+"${CORPUS_EXTRA[@]}"}"
else
    run "corpus (Piper ${PIPER_URL}, all-Piper mix)" \
        env PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.corpus \
            --wake-word "$WAKE_WORD" --piper-url "$PIPER_URL" --piper-speakers 12 \
            "${CORPUS_EXTRA[@]+"${CORPUS_EXTRA[@]}"}"
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
        env PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.features \
            --wake-word "$WAKE_WORD" --clean
else
    NEWEST_CORPUS="$(find "data/corpus/${SAFE_NAME}/mww" -path "*/features" -prune -o -type f -newer "$FEATURES_DIR" -print -quit 2>/dev/null || true)"
    if [[ -n "$NEWEST_CORPUS" ]]; then
        echo "=== $(date '+%H:%M:%S')  corpus changed since features - rebuilding"
        run "features" \
            env PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.features \
                --wake-word "$WAKE_WORD" --clean
    else
        echo
        echo "=== $(date '+%H:%M:%S')  features (corpus unchanged)"
    fi
fi

# === 3. train ======================================================================
#
# The tag is computed HERE, after the corpus exists and before training starts -
# the same reason run-mww-training.sh documents: the corpus is part of the tag,
# and model_train_eval refuses to train into an existing directory. The checksum
# guard in train/mww/train.py still applies - it is in the code, not in the shell.
#
# Computed THROUGH train.mww.train --print-tag, not through train.provenance: the
# tag's config half (-h) is a hash of the resolved hyperparameters, which only
# exists where the config is built. Computing it twice (here and in train.py) is
# how the archive and the run directory would drift apart. Same arguments as the
# real run below, so the tag is exactly the one the run will get.
#
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

TAG="$(PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.train \
    --wake-word "$WAKE_WORD" --print-tag \
    --ambient "${AMBIENT_ARGS[@]+"${AMBIENT_ARGS[@]}"}" "$@" 2>&1 | tail -1)"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
run "run tag: $TAG"

run "training"
set +e
PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.train \
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
if PYTHONPATH=src "$ENV_DIR/.venv/bin/python" -m train.mww.manifest \
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
# run-oww-training.sh produces and what src/eval/src/paths.py documents. Verbatim from
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
if [[ "${SMOKE:-}" == "1" ]]; then
    echo "    SMOKE RUN: the model is smoke-named, the results are not measurable,"
    echo "    and no real run's archive was touched. Delete it when done:"
    echo "      rm -rf output/${SAFE_NAME}/mww/${TAG}"
else
echo
echo "    Compare the wall time against the container's 26m06s (this machine, 2026-09-06)"
echo "    and the per-stage numbers against src/scripts/tf_probe.py and src/scripts/bench_tts.py."
echo
echo "    Evaluate (host, P2.4; Docker still works):"
echo "      src/eval/.venv/bin/python src/eval/src/eval_model.py --model $TAGGED_MODEL"
echo "      # or: cd src/eval && docker compose run --rm eval python -m eval.eval_model \\ --model ..."
echo "    Confirm the cutoff against the held-out recordings before deploying it;"
echo "    the per-speaker rows are the ones that decide it."
fi
