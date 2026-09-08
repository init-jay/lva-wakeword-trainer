# tts-service

The shared TTS layer of this repo: one package, four engines, one interface, one
TCP service.

Everything that renders text into 16 kHz mono int16 clips for this repo goes
through here:

  * the two trainers' corpus stages (`train/corpus/` are now thin shims that
    re-export it; `train/oww/train.py` and `train/mww/corpus.py` are untouched)
  * the eval corpus generators (`eval/src/generate_negatives.py`,
    `eval/src/generate_positives.py`)
  * the CLI below, for checking a machine or an engine in five seconds

## The engines

| engine | what it is | where it runs | client spec |
|---|---|---|---|
| `kokoro-mlx` | Kokoro in-process via the mlx fork (word timestamps) | Apple Silicon host, `uv` venv | `kokoro-mlx` or `mlx://` |
| `kokoro-http` | Kokoro-FastAPI (OpenAI-compatible API) | any box, Docker (CPU or CUDA overlay) or bare host | `http://kokoro:8880` (comma-separated for a pool) |
| `piper` | Wyoming Piper (serial, one voice resident) | any box, Docker (multi-arch, no GPU needed) | `piper://127.0.0.1:10200`, or bare `host:port` |
| `tts-service` | any of the above, behind a TCP port | wherever the server is run | `tcp://host:8899` (comma-separated for a pool) |

A **spec** is what every caller passes - never a class. `engines_from_spec`
(`tts_service/engines/__init__.py`) is the registry: a fourth engine is a new
`engines/` module plus one dispatch row, and nothing upstream changes. The
`tcp://` row is how a *new machine* shows up: it runs the one server below in
front of whatever model it has, and this side points at the port.

The engines' failure conventions are the ones the corpus layer has always had
(`tts_service/engine.py`): Kokoro returns `None` on a transient render miss (the
generator retries that clip alone); Piper raises, because its caller reports
*which* voice failed. The service carries both over the wire: a null clip is the
Kokoro convention, an error envelope is the Piper one.

## The service

```
python -m tts_service.server --engine <spec> [--host 127.0.0.1] [--port 8899]
```

The server hosts ONE engine (any spec in the table, including another service's
`tcp://` - not that that is useful) and answers the one-line protocol in
`tts_service/protocol.py`: one JSON line per direction, one exchange per
connection, audio as base64 wav. That is deliberately not HTTP: there is one
verb per `op` field, the reply is always one line, and the only real
requirement - a 2 MB payload - is met by a newline. The server takes every
engine call under one lock, because every engine it hosts is effectively
single-threaded (`server.py` says which measurements settle that).

### Launching it, per machine

Apple Silicon host (in-process MLX - no other process needed; the measured
faster route for mww on a Mac, see SPEED.md):

    cd tts-service
    uv run --extra mlx python -m tts_service.server --engine kokoro-mlx --host 0.0.0.0 --port 8899

Any Docker box, in front of the compose TTS services (the trainer images copy
and bind-mount `tts-service/` - see `docker-compose.yml`):

    # Kokoro-FastAPI behind it (CPU or CUDA overlay)
    docker compose run --rm oww-trainer bash -c \
        "cd /app/tts-service && exec python -m tts_service.server --engine http://kokoro:8880 --host 0.0.0.0 --port 8899"
    # Piper behind it
    docker compose run --rm mww-trainer bash -c \
        "cd /app/tts-service && exec python -m tts_service.server --engine piper:10200 --host 0.0.0.0 --port 8899"

Then any client on a reachable network uses `tcp://<that-host>:8899`. Comma-
separate several for a pool: `tcp://a:8899,tcp://b:8898` round-robins exactly
the way the old `KOKORO_URL=http://kokoro:8880,http://kokoro2:8880` did.

## The CLI

    # what an engine offers
    python -m tts_service --tts kokoro-mlx --list-voices
    python -m tts_service --tts tcp://127.0.0.1:8899 --list-voices

    # render: one clip to a file, or a batch into a directory
    python -m tts_service --tts kokoro-mlx --voice af_bella -o hey.wav "hey seeree"
    python -m tts_service --tts tcp://127.0.0.1:8899 --voice af_bella --batch 5 --out-dir out/ \
        "hey seeree" "hey serious" "hey series" "hey Sarah" "hey Cindy"

    # Piper takes (voice, speaker) pairs
    python -m tts_service --tts 127.0.0.1:10200 --voice en_US-libritts_r-medium --speaker 12 -o s12.wav "hey seeree"

A run prints one line per clip and a measured ms/clip summary - the same number
the corpus generator's bar shows - so a slow machine says so before an hour is
spent on a corpus.

## Measured, 2026-09-08 (Apple Silicon Mac, M-series; Docker Desktop)

| path | shape | ms/clip end to end |
|---|---|---|
| `kokoro-mlx` in-process | 88 short clips, serial | 101 |
| `kokoro-mlx` in-process | 88 short clips, batch 10 | 102 (no gain - see below) |
| `kokoro-mlx` in-process | 88 long clips (1.5-2.3 s), serial | 134 |
| `kokoro-mlx` in-process | 88 long clips, batch 10 | 103 (1.3x) |
| `tcp://` -> service hosting `kokoro-mlx` | 5 short clips, batch 5 | 37 |
| `tcp://` -> service hosting `piper` (10200) | single, 0.73 s @1.2x | 36 |
| `tcp://` -> service hosting `piper` | 3 clips, batch 3 (2.8 s total) | 33 |
| `piper` direct (10200) | single, 0.77 s @1.2x, cold connection | 711 |
| `kokoro-http` (8880, CPU image) | single, 0.97 s | 375 |
| `kokoro-http` (8880) | 5 short clips, batch 5 | 109 |

First call on a cold MLX process pays the model load (~6 s with a warm Hugging
Face cache; the 122-file fetch itself is <1 s). Piper's direct-connection number
includes the first connection and the WSOLA stretch; the via-service numbers are
warm.

**Why batch 10 did not beat serial for short MLX clips** (the one counterintuitive
row): the join/split algorithm saves *request* overhead - on the HTTP server that
is ~3/4 of a short request (see `tts_service/engine.py`, Engine.batch), and
batching measures 3.3-5x there. The in-process fork has almost no per-request
overhead to save, so for sub-second phrases joining buys nothing; the gain
appears when the joined text is long enough that the model's per-phoneme cost
dominates (the 1.3x row). The batch machinery is kept engine-agnostic because
the same code is the whole win over HTTP and costs nothing in-process.

## Layout

    tts_service/
      engine.py       the Engine interface + split_joined (the batch algorithm)
      audio.py        16 kHz helpers: to_int16, time_stretch (WSOLA)
      protocol.py     the service's one-line wire format
      server.py       python -m tts_service.server - hosts one engine behind TCP
      client.py       KokoroPool, probe_kokoro_servers, run_jobs, Client
                      (the calibrated corpus generator - see the file's header)
      engines/        kokoro_mlx, kokoro_http, piper, tcp + the spec registry
      __main__.py     the CLI

Callers that used to import the old locations keep working through the shims:
`train/corpus/kokoro.py`, `train/corpus/kokoro_mlx.py`, `train/corpus/piper.py`,
`train/corpus/augment.py` (time_stretch now lives in `audio.py`).
