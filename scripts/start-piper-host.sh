#!/usr/bin/env bash
#
# Run Piper TTS on the HOST, as a Wyoming server on port 10200.
#
# STATUS: this is now a DEBUGGING server, not a training or auditing one. Training
# corpora are rendered by the in-process engine in tts-service/engines/piper (protocol
# port 8898), and tools/audit_voices.py and bench_tts.py speak the protocol, not
# Wyoming - to audit the piper engine, point them at that same 8898 server, since
# it IS the engine. What this host process is for now is poking at the raw Wyoming
# protocol directly (and any external consumer that needs one). Keep the pins in
# lockstep with the in-process engine's pyproject
#
# MIRRORS start-kokoro-host.sh ON PURPOSE:
#   - a uv venv under data/external/ with pinned packages, not a uv project
#     (no pyproject, no lockfile) - same shape as the Kokoro host env;
#   - voices in data/external/piper/voices - regenerable third-party
#     downloads, in data/external for the same reason as Kokoro's model dir;
#   - one foreground process in its own terminal; the run script preflights
#     the port and points here if the server is missing.
#
# THE PINS MATCH EVERY OTHER PIPER IN THE REPO, AND THAT IS NOT COSMETIC.
# piper-tts 1.7.0 is what tts-service/engines/piper (the Mac training engine)
# pins and what docker/Dockerfile.piper pip-installs (checked 2026-09-07, when
# it was still the rhasspy/wyoming-piper:2.4.3 image's /usr/src/.venv). A fresh
# install today resolves 1.8.0 - a different G2P release - and the
# MISPRONOUNCING/UNAUDITED_PIPER_VOICES exclusion lists in train/corpus/piper.py
# were audited against this voice set and this G2P (piper.py header). Bump all
# three together: this script, the engine's pyproject, and Dockerfile.piper.
#
#   ./scripts/start-piper-host.sh                  # run (foreground)
#   ./scripts/start-piper-host.sh --venv-only      # (re)build the venv and stop
#   PIPER_PORT=10300 ./scripts/start-piper-host.sh # a different port
#
# The port defaults to 10200 (the historical Wyoming port; nothing in the repo
# speaks to it by default any more). If the compose `piper` service is up on
# this machine at the same time, its published 8898 does not collide, but pick
# a different PIPER_PORT anyway if you have other listeners around.
#
# VOICES. The default voice (en_US-lessac-medium) is fetched during setup,
# below, and every other voice is fetched by the server on first use into
# data/external/piper/voices (~60MB per single-speaker model; multi-speaker
# ones like libritts_r are much larger). The server logs each download, so a
# first corpus run visibly fills the directory; reruns are download-free.

set -euo pipefail

cd "$(dirname "$0")/.."
ENV_DIR="data/external/piper"
VOICES_DIR="$ENV_DIR/voices"
VENV_PY="$ENV_DIR/.venv/bin/python"
PYTHON_VERSION="3.12"
DEFAULT_VOICE="en_US-lessac-medium"
PIPER_PORT="${PIPER_PORT:-10200}"

# Match tts-service/engines/piper and Dockerfile.piper. Bump all three together.
PIP_VERSIONS=(wyoming-piper==2.4.3 piper-tts==1.7.0)

if [[ ! -x "$VENV_PY" || "${1:-}" == "--venv-only" ]]; then
    # Only .venv is rebuilt; voices/ is kept - refetching a populated
    # multi-GB directory is exactly what this layout exists to avoid.
    echo "=== $(date '+%H:%M:%S')  (re)building $ENV_DIR/.venv (python $PYTHON_VERSION)"
    rm -rf "$ENV_DIR/.venv"
    mkdir -p "$VOICES_DIR"
    uv venv "$ENV_DIR/.venv" --python "$PYTHON_VERSION"
    uv pip install --python "$VENV_PY" "${PIP_VERSIONS[@]}"

    # Fetch the default voice so the preflight below - and the server's
    # required --voice - have something on disk. Other voices are
    # downloaded by the server on first synthesize.
    "$VENV_PY" - "$VOICES_DIR" "$DEFAULT_VOICE" <<'EOF'
import sys
from wyoming_piper.download import ensure_voice_exists, get_voices
voices_dir, voice = sys.argv[1], sys.argv[2]
ensure_voice_exists(voice, [voices_dir], voices_dir, get_voices(voices_dir))
print(f"=== voice ready: {voice}")
EOF

    # PREFLIGHT: synthesize once, offline, before the server exec. A
    # Describe round trip (the wyoming health check) never touches the TTS
    # backend, so a server can happily bind the port and report healthy
    # while still failing on the FIRST synthesize - e.g. if a future
    # espeakbridge build for this platform is broken. The failure we want
    # here is loud and at startup, not two hours into the corpus stage.
    #
    # This is piper's own CLI, not a wyoming request: the same G2P and the
    # same ONNX session, minus the server, so the only thing it does not
    # cover is the wyoming handler itself - which the run script's probe
    # covers, against the running server.
        TMP_TXT="$(mktemp -t piper-preflight.XXXXXX.txt)"
    TMP_WAV="$(mktemp -t piper-preflight.XXXXXX.wav)"
    echo 'wake word preflight' > "$TMP_TXT"
    if "$VENV_PY" -m piper \
        -m "$DEFAULT_VOICE" --data-dir "$VOICES_DIR" \
        -i "$TMP_TXT" --output-file "$TMP_WAV" > /dev/null 2> /tmp/piper-preflight.err; then
        bytes=$(wc -c < "$TMP_WAV")
        if [[ "$bytes" -lt 1000 ]]; then
            echo "  Piper preflight FAILED: synthesized wav is only $bytes bytes."
            cat /tmp/piper-preflight.err
            rm -f "$TMP_TXT" "$TMP_WAV" /tmp/piper-preflight.err
            exit 1
        fi
        echo "  Piper preflight OK: synthesized $((bytes / 1024))KB of audio."
    else
        echo "  Piper preflight FAILED (G2P/ONNX cannot synthesize on this machine):"
        cat /tmp/piper-preflight.err
        rm -f "$TMP_TXT" "$TMP_WAV" /tmp/piper-preflight.err
        exit 1
    fi
    rm -f "$TMP_TXT" "$TMP_WAV" /tmp/piper-preflight.err

    if [[ "${1:-}" == "--venv-only" ]]; then
        echo "=== venv ready (voices in $VOICES_DIR/). Start the server without --venv-only."
        exit 0
    fi
fi

if [ -n "$(lsof -ti ":$PIPER_PORT" 2> /dev/null)" ]; then
    echo "Port $PIPER_PORT is already in use."
    echo "If that is another Piper (e.g. the compose 'piper' service), stop it first -"
    echo "or run this one on PIPER_PORT=<other> and set PIPER_URL=<host>:<other>"
    echo "for the training script."
    exit 1
fi

echo "=== $(date '+%H:%M:%S')  Starting wyoming-piper on tcp://127.0.0.1:$PIPER_PORT"
echo "    voices: $VOICES_DIR (missing ones download on first use)"
# --voice is required by wyoming-piper even though voices download on demand:
# without it the server exits 2 before binding the port (Dockerfile.piper).
# train/corpus/piper.py always names a voice in its synthesize requests anyway.
exec "$VENV_PY" -m wyoming_piper \
    --uri "tcp://127.0.0.1:$PIPER_PORT" \
    --data-dir "$VOICES_DIR" \
    --download-dir "$VOICES_DIR" \
    --voice "$DEFAULT_VOICE"
