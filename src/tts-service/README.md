# tts-service

The TTS **protocol** of this repo, and the engines that speak it.

Two kinds of code live here:

  * **`tts_protocol/`** - the protocol itself: a tiny package (pure stdlib
    plus the shared audio code) that defines the wire format and the client.
    Training and eval components depend on *this*, and on nothing else in the
    repo's TTS stack.
  * **`engines/`** - one uv project per engine, each an independent server that
    implements the protocol. The engines are NOT part of the protocol package
    and are NOT installed into the trainers: each runs in its own venv (or its
    own Docker image, for the CUDA box) because their dependency sets conflict
    with each other and with the trainers' (see below).

Everything that renders text into 16 kHz mono int16 clips for this repo goes
through the protocol:

  * the two trainers' corpus stages (`src/train/corpus/kokoro.py`,
    `src/train/corpus/piper.py`)
  * the eval corpus generators (`src/eval/src/generate_negatives.py`,
    `generate_positives.py`)
  * the CLI below, for checking a machine or an engine in five seconds

## The protocol

One TCP port per engine process. One JSON line per direction. One exchange per
connection (a per-request socket, no sessions, no pooling). Audio travels as
base64 WAV. The full contract, including the exact envelope shapes, is the
docstring of `tts_protocol/wire.py` - that document is the spec; this README
explains the shape of the system around it.

The client (`tts_protocol/client.py`, `TtsClient`) implements the `Engine`
interface (`tts_protocol/engine.py`), so the corpus generators program against
the interface and never name an engine class:

```
TtsClient("tcp://127.0.0.1:8900")     # one engine
TtsClient("tcp://box-a:8900,box-b:8900")   # a pool; the client load-balances
```

**`tcp://` is the only spec the client accepts.** Old raw forms (`http://...`,
a bare `host:port`, `mlx://`) are rejected at construction with an explanation.
That is deliberate: this code used to read the same string as three different
backends (HTTP service, raw Wyoming socket, in-process mlx), and a silent
misread once rendered a whole corpus from the wrong engine. A `tcp://` URL
cannot be misread.

Batching is the client's job, not the server's: `Engine.batch` joins a list of
texts into one utterance (`". ".join(t.rstrip(".") for t in texts) + "."`),
renders it once, and splits it back apart on the word timestamps. The join is
byte-identical to what the old in-process `kokoro_tts_batch` produced, so the
rendered audio of a batched corpus is what the 16 calibrated runs trained on.

Two capabilities ride on the `voices` reply, so a probe needs no test render:
`timestamps` (does the engine return word timestamps - run-on cuts depend on
this) and `speaker` (does it have the multi-speaker `(voice, speaker)` pairs).
The `engine` name is reported too, so a misconfigured port tells you what it
actually is.

## The engines

| engine | what it is | where it runs | protocol port | client spec |
|---|---|---|---|---|
| `engines/kokoro_mlx` | Kokoro-82M in-process via the mlx fork (word timestamps) | **Apple Silicon host** (the MLX runtime), `uv` project | 8900 | `tcp://<mac>:8900` |
| `engines/piper` | Piper in-process via piper-tts 1.7.0 (serial, one voice resident) | **Apple Silicon host AND Docker** - the SAME code in both | 8898 | `tcp://<box>:8898` |
| `docker/tts_engines/kokoro_http_engine` | Kokoro-FastAPI behind a protocol wrapper | **Docker** (CPU or CUDA), FastAPI on 8880 in the same container | 8899 | `tcp://<box>:8899` |

**Apple Silicon launch mode is the two uv projects.** No Docker, no Wyoming
process: the trainer's host run talks to `kokoro_mlx` (8900) and `piper` (8898)
over loopback. On a Mac the non-MLX Kokoro is not used at all - the mlx engine
is strictly better there (its own module document carries the measurements) -
which is why there is no non-MLX Kokoro project under `src/tts-service/engines/`
and the HTTP adapter lives in `docker/tts_engines/`.

**The CUDA box runs the two Docker images.** The piper image hosts the exact
same in-process engine as the Mac's project (`docker/Dockerfile.piper` bakes in
`src/tts-service/engines/piper`), so the two machines render from one audited code
path; the kokoro image runs FastAPI and its wrapper in one container
(compose's CMD starts both), which is the only place the HTTP adapter ever
runs. The trainers point at `tcp://kokoro:8899` / `tcp://piper:8898` on the
compose network.

### Why the engines are separate processes

Three reasons, and each one is load-bearing:

  1. **Dependencies conflict.** kokoro-mlx drags torch (+ the MLX runtime) and
     misaki; piper-tts drags onnxruntime; the Kokoro-FastAPI image drags torch
     CPU; the trainers need torch 2.5.1 (oww) or tensorflow (mww) - and
     oww's numpy 1.x and mww's numpy 2.x do not share an environment (the
     `train-applesilicon` / `train-mww-applesilicon` split exists for that).
     One process cannot hold all of these, so the engines are processes.
  2. **Isolation is a feature.** The trainer images therefore carry no TTS
     engine dependency at all; the worst a broken TTS stack can do is fail a
     corpus render, not corrupt a training environment.
  3. **Ports are per-engine**, so a dead or slow engine is visible and
     replaceable without touching the others, and the probes in the run
     scripts ask each engine the question its stage asks.

**Piper scales horizontally; Kokoro on a Mac mostly does not.** A Piper
instance holds one model resident and takes every call under one lock, so one
instance is one serial lane no matter how many client threads point at it -
throughput scales with PROCESSES. `src/scripts/start-tts-fleet.sh N` starts N of
them on 8898..8897+N-1 (all sharing `data/external/piper/voices`, all of them
probed with a voices round trip), and the run scripts take the comma-joined
list it prints as `PIPER_URLS`. The corpus layer (`src/train/corpus/piper.py`,
`PiperFleet`) shards the fleet BY VOICE - each model, all of its speakers,
pinned to one instance for the whole run - because round-robin would make
every instance reload a model on most of its requests, and a reload (0.6 s)
costs more than the synthesis it delays. Kokoro takes the same comma-list
form on the client (round-robin, `KokoroPool`) but is a much weaker candidate
on a Mac: MLX contends on one GPU, so measure before adding instances
(improvement.md, P2.1).

The engines' failure conventions survive the wire unchanged: Kokoro returns a
null clip on a transient render miss (the generator retries that clip alone);
Piper raises, because its caller reports *which* voice failed.

## The CLI

Check an engine in five seconds - any of the ports above, from anywhere:

```
uv run --project src/tts-service/tts_protocol python -m tts_protocol \
    --server tcp://127.0.0.1:8900 --list-voices
uv run --project src/tts-service/tts_protocol python -m tts_protocol \
    --server tcp://127.0.0.1:8898 --voice en_US-lessac-medium \
    --speaker 1 --speed 1.2 -o /tmp/test.wav
uv run --project src/tts-service/tts_protocol python -m tts_protocol \
    --server tcp://127.0.0.1:8900 --voice af_heart --batch \
    "hey seeree" "hey seeree, can you hear me" --out-dir /tmp/batch
```

The package's own venv (`tts_protocol/.venv`) has exactly the protocol - no
engine - so a failure to start one is unambiguously the engine's, never a
dependency accident on the client side.

## Versioning

The protocol is deliberately simple enough that its two sides (client in the
trainers, server in the engines) are pinned in this repo; both sides move in
one commit. The envelopes carry no version field because a mismatch fails
loudly and fast (an unknown `op`, a missing key) rather than silently - one
exchange per connection means there is no session state to desynchronize.
