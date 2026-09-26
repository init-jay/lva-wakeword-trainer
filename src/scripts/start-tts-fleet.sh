#!/usr/bin/env bash
#
# N in-process Piper engines (src/tts-service/engines/piper), one per port,
# 8898 .. 8897+N-1, sharing the voices under data/external/piper/voices -
# the horizontal-scaling fast path for the corpus stage.
#
# WHY INSTANCES AT ALL: the Piper engine holds ONE model resident and takes
# every engine call under one lock (src/tts-service/engines/piper's docstring,
# tts_protocol/server.py), so client threads against one instance queue, they
# do not run. Throughput scales with PROCESSES: the single instance measures
# 21.66 clips/s in src/scripts/bench_tts.py, and the mww corpus stage ran at ~16
# clips/s aggregate against it (improvement.md, P2.1 - the corpus stage is
# the serial wall of a Mac run). N instances should land near N x that, and
# src/scripts/bench_tts.py is the instrument to confirm it.
#
# WHAT THE CONSUMER DOES WITH IT: the corpus layer (src/train/corpus/piper.py,
# PiperFleet) shards the fleet BY VOICE - each model, all of its speakers,
# pinned to one instance for the whole run. Never round-robin: a request
# naming a model the instance does not hold costs a 0.6 s reload, and
# round-robin would make every instance reload on most requests - more
# reloads than the single instance would have done.
#
#     PIPER_URLS="$(./src/scripts/start-tts-fleet.sh 4)" \
#         ./src/scripts/run-mww-training-applesilicon.sh "hey seeree"
#
# The run scripts take the list as PIPER_URLS (or --piper-url): only this
# script's LAST line is on stdout - the comma-joined tcp:// URL list - and
# every status line goes to stderr, so $(...) captures exactly the list.
#
# THE FLEET OUTLIVES THIS SCRIPT - that is the point: the run script probes
# and renders against it, so a fleet that died when the launcher exited would
# die before its first clip. The trap kills the children this script STARTED
# on failure and on Ctrl-C (a port that fails to come up takes the rest down
# with it - a half fleet renders a half corpus, and the run script's probe
# would have reported exactly this anyway). On a CLEAN exit the trap is
# removed first and the PIDs go to src/logs/tts-fleet/pids, which is also how you
# stop the fleet later:
#
#     kill $(cat src/logs/tts-fleet/pids)
#
# IDEMPOTENT: a port that already answers a voices round trip is NOT started
# again (a second process on it would fail with EADDRINUSE anyway) - the
# script says which one, and all-already-up is a no-op success that just
# re-prints the list, so running the same command twice is safe. A port that
# answers a bare connect but not the protocol - or one answered by a
# non-piper engine - is refused, naming the port: the fleet is sharded by
# voice, and a lane that does not serve piper's catalog cannot take a shard.
#
#   ./src/scripts/start-tts-fleet.sh 4            # start/refresh four, print the list
#
# All instances must serve the SAME voices directory - the corpus probe
# (PiperFleet.probe) refuses a fleet whose catalogs disagree, because a
# sharded voice an instance lacks fails per-clip and shrinks the corpus
# silently. This script passes the shared default to every instance.

set -euo pipefail

cd "$(dirname "$0")/../../"

N="${1:-}"
if [[ -z "$N" || ! "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 N   (N = number of Piper instances, ports 8898 .. 8897+N-1)" >&2
    exit 2
fi

BASE_PORT=8898
PROJECT="src/tts-service/engines/piper"
VOICES_DIR="data/external/piper/voices"
LOG_DIR="src/logs/tts-fleet"
# 2 s x 120: covers a cold uv sync plus the engine's startup catalog; a dead
# process fails the kill -0 check fast and never burns the whole window.
WAIT_ROUNDS=120

mkdir -p "$LOG_DIR"
if [[ ! -d "$VOICES_DIR" ]] || ! ls "$VOICES_DIR"/*.onnx >/dev/null 2>&1; then
    echo "ERROR: no Piper voices under $VOICES_DIR - run ./src/scripts/download-external-data.sh mww first." >&2
    exit 1
fi

# A voices round trip over the PROTOCOL, not a TCP connect: the server binds
# the port before the engine has loaded, and a connect check passes for a
# listener that then fails its first render. The probe is the same question
# the run scripts' preflight asks. The piper project's own venv carries the
# protocol client (pure stdlib) - no extra dependency, and uv syncs it if
# the venv has not been built yet.
probe() { # probe <port> -> "UP <engine>" on stdout, exit 1 with "DOWN <why>" on stderr otherwise
    local port="$1" out
    if out="$(uv run --project "$PROJECT" python - "$port" <<'PY' 2>/dev/null
import sys
from tts_protocol.client import TtsClient
try:
    c = TtsClient(f"tcp://127.0.0.1:{sys.argv[1]}")
    c.voices()
except Exception as e:
    print(f"DOWN {type(e).__name__}: {e}")
    sys.exit(1)
print(f"UP {c.server_engine or '?'}")
PY
    )"; then
        printf '%s\n' "$out"
        return 0
    fi
    # uv run failed (or the engine died between the probe and now): carry the
    # cause - the python side prints "DOWN <why>" to stdout even on exit 1.
    if [[ -n "${out:-}" && "$out" == DOWN* ]]; then
        printf '%s\n' "$out" >&2
    else
        printf 'DOWN probe failed (uv run: %s)\n' "${out:-no output}" >&2
    fi
    return 1
}

PIDS=()           # children this script started (the trap kills exactly these)
STARTED_PORTS=()  # parallel to PIDS
URLS=()

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "  fleet: stopping the ${#PIDS[@]} instance(s) this script started: ${PIDS[*]}" >&2
        kill "${PIDS[@]}" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

for ((i = 0; i < N; i++)); do
    # 8898 DOWN (8898, 8897, ...), matching this header's "8898 .. 8897+N-1":
    # the Kokoro engines occupy the ports ABOVE 8898 on a co-located Mac
    # (8899/8901 docker, 8900 in-process mlx - src/tts-service/README.md), so a
    # fleet that counted UP would collide with them on its second instance.
    PORT=$((BASE_PORT - i))
    URL="tcp://127.0.0.1:$PORT"
    out="$(probe "$PORT" 2>/dev/null)" || out="DOWN probe failed"

    if [[ "$out" == UP* ]]; then
        ENGINE="${out#UP }"
        if [[ "$ENGINE" != "piper" ]]; then
            echo "REFUSING: port $PORT already answers, and it is engine '$ENGINE', not piper." >&2
            echo "       The fleet is sharded by voice, so a lane must serve piper's catalog." >&2
            echo "       Stop whatever owns $PORT and try again (or pick a free base port)." >&2
            exit 1
        fi
        echo "  port $PORT already answers (piper) - reusing it, not starting another" >&2
    else
        if nc -z "127.0.0.1" "$PORT" 2>/dev/null; then
            echo "REFUSING: port $PORT is owned by a process that does not answer the TTS" >&2
            echo "       protocol ($out) - it cannot take a voice shard. Stop it and try again." >&2
            exit 1
        fi
        echo "  starting piper_engine on $PORT  (log $LOG_DIR/piper-$PORT.log)" >&2
        uv run --project "$PROJECT" python -m piper_engine --port "$PORT" \
            >>"$LOG_DIR/piper-$PORT.log" 2>&1 &
        PIDS+=("$!")
        STARTED_PORTS+=("$PORT")
    fi
    URLS+=("$URL")
done

if [[ ${#PIDS[@]} -gt 0 ]]; then
    # Wait for every instance this script started to answer the same voices
    # round trip the corpus stage makes. uv run prints the launch error to
    # the per-port log, so a dead process is read there, not guessed.
    for i in "${!PIDS[@]}"; do
        PORT="${STARTED_PORTS[$i]}"
        ok=""
        for ((t = 0; t < WAIT_ROUNDS; t++)); do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                echo "ERROR: piper_engine on $PORT died during startup." >&2
                tail -5 "$LOG_DIR/piper-$PORT.log" >&2 || true
                exit 1
            fi
            if out="$(probe "$PORT" 2>/dev/null)" && [[ "$out" == UP* ]]; then
                ok=1
                break
            fi
            sleep 2
        done
        if [[ -z "$ok" ]]; then
            echo "ERROR: piper_engine on $PORT did not answer a voices round trip" >&2
            echo "       within $((WAIT_ROUNDS * 2)) s - see $LOG_DIR/piper-$PORT.log." >&2
            exit 1
        fi
        echo "  port $PORT up: ${out#UP }  (after $((t + 1)) probe(s))" >&2
    done
fi

# CLEAN EXIT: the fleet outlives this script (see the header) - the trap is
# what keeps it alive after a failure or Ctrl-C, so remove it, and hand the
# PIDs over to the pidfile for the later `kill $(cat ...)`.
trap - EXIT INT TERM
if [[ -f "$LOG_DIR/pids" ]]; then
    # A scale-up (N grew since the last run) must not drop the instances the
    # pidfile already carries: keep the ones still alive, so the stop command
    # reaches the WHOLE fleet, not only what this run started.
    while IFS= read -r old; do
        [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null && PIDS+=("$old")
    done < "$LOG_DIR/pids"
fi
if [[ ${#PIDS[@]} -gt 0 ]]; then
    printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids"
    echo "  fleet is up (${#PIDS[@]} in the pidfile: ${#PIDS[@]} total across runs);" >&2
    echo "  stop it later with:  kill \$(cat $LOG_DIR/pids)" >&2
else
    echo "  fleet already fully up (nothing started here)" >&2
fi

# The one line on stdout: the comma-joined list, exactly what PIPER_URLS /
# --piper-url consume.
IFS=,
echo "${URLS[*]}"
