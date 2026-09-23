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
# this script starts nothing: the engines are the uv projects in
# tts-service/engines/, each in its own venv -
#
#     uv run --project tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine \
#         --port 8900        # Kokoro, in-process MLX, protocol port 8900
#     uv run --project tts-service/engines/piper python -m piper_engine \
#         --port 8898        # Piper, in-process piper-tts, protocol port 8898
#
# in another terminal, and the trainer talks to them over the TTS protocol
# (tcp:// URLs; KOKORO_URL below defaults to the mlx one). --piper-fraction N
# adds Piper-rendered phrase-alone clips to that corpus; it needs the piper
# engine above (PIPER_URL defaults to tcp://127.0.0.1:8898). For the measurement
# a full run is the wrong thing to include anyway: TTS is the same work either
# way and would bury the difference under 49 minutes of noise.
#
# PIPER FLEET (the fast path for the corpus stage): one Piper instance is one
# serial lane - the engine holds one model resident and takes every call under
# one lock, so client threads queue instead of run (21.66 clips/s measured for
# the single instance in tools/bench_tts.py, improvement.md P2.1). PIPER_URLS
# takes the comma-joined list scripts/start-tts-fleet.sh N prints, and the
# corpus shards the fleet BY VOICE - each model pinned to one instance for the
# whole run (corpus/piper.py, PiperFleet):
#
#     PIPER_URLS="$(./scripts/start-tts-fleet.sh 4)" \
#         ./scripts/run-oww-training-applesilicon.sh "hey seeree" --piper-fraction 0.3
#
# SMOKE=1: a few-minute end-to-end check that a changed train/ tree still runs the
# whole pipeline: corpus reuse, feature recompute, 200-step training, real tflite
# conversion. No TTS server - the corpus is the held-fixed input and the engines'
# health is probed at the start of a normal run anyway. The model lands in
# output/<wake>/oww/smoke-<stamp>/ and the canonical model, the .last_run_tag and
# the archive stay untouched (train/oww/train.py --smoke):
#
#     SMOKE=1 ./scripts/run-oww-training-applesilicon.sh "hey seeree"

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
#     TtsProtocolError: tcp://host.docker.internal:8900 unreachable: ...
#     [probe] no usable KOKORO server
#
# On the host the same server is simply localhost. Rewrite rather than fail: the
# intent is unambiguous, and the alternative is an error about DNS for what is really
# a leftover environment variable.
if [[ "${KOKORO_URL:-}" == *host.docker.internal* ]]; then
    KOKORO_URL="${KOKORO_URL//host.docker.internal/127.0.0.1}"
    export KOKORO_URL
    echo "=== note: rewrote host.docker.internal -> 127.0.0.1 in KOKORO_URL"
    echo "          ($KOKORO_URL) - that name only resolves inside a container."
fi

# The protocol client accepts ONLY tcp:// specs (it used to read three different URL
# forms as three different backends, and a silent misread rendered a corpus from the
# wrong engine - see tts_protocol/client.py). Coerce the two bare host:port forms a
# Mac user is likely to type, and say what was done.
if [[ -n "${KOKORO_URL:-}" && "${KOKORO_URL}" != tcp://* ]]; then
    KOKORO_URL="tcp://${KOKORO_URL}"
    export KOKORO_URL
    echo "=== note: rewrote KOKORO_URL to tcp:// form: $KOKORO_URL"
fi

# KOKORO_URL
#
# The Mac's Kokoro is the in-process kokoro-mlx engine (its own uv project,
# tts-service/engines/kokoro_mlx) on protocol port 8900. Export a default only
# when unset: an explicit --kokoro-url on the command line still wins, and a
# pre-set KOKORO_URL (coerced to tcp:// above) is respected.
if [[ -z "${KOKORO_URL:-}" ]]; then
    KOKORO_URL="tcp://127.0.0.1:8900"
    export KOKORO_URL
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
# On a Mac the Piper engine is in-process (tts-service/engines/piper: piper-tts
# 1.7.0 loaded directly, no Wyoming process at all) on protocol port 8898.
# Bare host:port input is coerced to tcp:// like KOKORO_URL above. A
# comma-separated value is a fleet (start-tts-fleet.sh) and is normalised
# element by element below, where the probe lives.
if [[ -n "${PIPER_URL:-}" && "${PIPER_URL}" != tcp://* && "${PIPER_URL}" != *,* ]]; then
    PIPER_URL="tcp://${PIPER_URL}"
    export PIPER_URL
    echo "=== note: rewrote PIPER_URL to tcp:// form: $PIPER_URL"
fi

# Fail before the corpus stage rather than during it. Both TTS backends are
# reachable from here as nothing but sockets, so a server that is down or not
# a TTS server at all would otherwise fail at clip 1 of thousands, and the
# log would not say why.
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
        --piper-fraction) want="--piper-fraction" ;;
        --piper-fraction=*) PIPER_FRACTION="${arg#*=}" ;;
        --piper-url) want="--piper-url" ;;
        --piper-url=*) PIPER_URL_ARG="${arg#*=}" ;;
    esac
done

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
    # Precedence: --piper-url (the command line, the strongest statement),
    # then PIPER_URLS (a comma list - an explicit fleet), then PIPER_URL,
    # then the default engine port. Each element of a list is coerced to
    # tcp:// individually: "a,b" must become tcp://a,tcp://b, not tcp://a,b.
    if [[ -n "$PIPER_URL_ARG" ]]; then
        PIPER_URL="${PIPER_URL_ARG}"
    elif [[ -n "${PIPER_URLS:-}" ]]; then
        PIPER_URL="${PIPER_URLS}"
    elif [[ -z "${PIPER_URL:-}" ]]; then
        PIPER_URL="tcp://127.0.0.1:8898"
    fi
    IFS=',' read -r -a PURLS <<< "$PIPER_URL"
    PIPER_URL=""
    for part in "${PURLS[@]}"; do
        part="${part// /}"
        [[ -z "$part" ]] && continue
        [[ "$part" != tcp://* ]] && part="tcp://$part"
        PIPER_URL="${PIPER_URL:+$PIPER_URL,}$part"
    done
    [[ -n "$PIPER_URL" ]] || { echo "ERROR: PIPER_URL / PIPER_URLS resolved to nothing." >&2; exit 2; }
    export PIPER_URL
    # A voices round trip over the protocol, not a TCP connect: a bound port
    # owned by a dead or non-protocol listener passes a connect check and
    # still fails the corpus stage. This asks the question the corpus stage
    # asks - against EVERY URL of a fleet, since the corpus sends each voice
    # to a different one.
    if ! "$ENV_DIR/.venv/bin/python" - "$PIPER_URL" <<'PYEOF'
import sys
sys.path.insert(0, ".")
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
        echo "  first phrase-alone render, not now. Start the engine in another" >&2
        echo "  terminal:  uv run --project tts-service/engines/piper python -m piper_engine --port 8898"
        echo "  (it uses the voices under data/external/piper) or point PIPER_URL / --piper-url at an existing one." >&2
        echo "  A fleet:  PIPER_URLS=\"\$(./scripts/start-tts-fleet.sh 4)\"" >&2
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
mkdir -p logs
LOG="logs/training-${SAFE_NAME}-macos-${STAMP}.log"

# SMOKE=1 (header): the smoke directory is named HERE, not in train.py, because
# this script's success check has to look at exactly the file the run writes -
# train.py's own default would pick a stamp this shell cannot see. --smoke goes
# into the arg list after the wake word, where it lands in "extra train.py args".
SMOKE_OUTPUT=""
if [[ "${SMOKE:-}" == "1" ]]; then
    SMOKE_OUTPUT="output/${SAFE_NAME}/oww/smoke-${STAMP}"
    set -- "$@" --smoke --smoke-output "$SMOKE_OUTPUT"
fi
[[ -n "$SMOKE_OUTPUT" ]] && MODEL="$SMOKE_OUTPUT/${SAFE_NAME}.onnx"

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
if [[ -n "$SMOKE_OUTPUT" ]]; then
    echo "    SMOKE RUN: the model is in the smoke directory, the results are not"
    echo "    measurable, and nothing in the archive changed. Delete it when done:"
    echo "      rm -rf $SMOKE_OUTPUT"
else
    echo "    Compare the training rate against the container run's 26 it/s - that is"
    echo "    the number this environment exists to beat. Then convert and evaluate:"
    echo "      $ENV_DIR/.venv/bin/python -m train.oww.onnx2tflite $MODEL"
fi
