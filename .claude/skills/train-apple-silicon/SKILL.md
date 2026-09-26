---
name: train-apple-silicon
description: Train wake-word models on an Apple Silicon Mac. openWakeWord runs natively on the host (uv venv, in-process Kokoro via MLX on the GPU, Piper TTS from a host server); microWakeWord runs natively on the host too (uv venv + host Piper, faster than the container) or in the multi-arch Docker CPU image. Use when the user wants to train or retrain a model on a Mac, set up the trainer environment, start the host TTS services, hits a failure in the Apple Silicon training scripts, or asks why training on a Mac differs from the CUDA box.
---

# Training on Apple Silicon

Step 2 of the pipeline.
## Setup, once per machine

```bash
./src/scripts/setup-applesilicon-trainer.sh
```

Idempotent; re-run it any time something downstream smells wrong. It clones
openWakeWord at the pinned commit, applies the patches in `src/train/patches/`, builds
`src/train/train-applesilicon/.venv` (Python 3.12), downloads the spacy model, and
verifies.

## The TTS engines

The Mac's launch mode is the two **uv projects in `src/tts-service/engines/`**, each
its own venv, each a protocol server in one terminal (see `src/tts-service/README.md`
for the protocol itself):

- **Kokoro, in-process MLX** (`kokoro_mlx`, port 8900, `uv sync`'d by the setup
  script): the same Kokoro-82M model renders inside that process, on the GPU.
  Word timestamps are exact (agreed with the FastAPI service to 3 ms on the same
  phrase - durations are predicted per phoneme and rendered from them, so run-on
  cuts come from measurement, not the estimation fallback). The trainer reaches
  it as `tcp://127.0.0.1:8900` - the run script's `KOKORO_URL` default.
- **Piper, in-process piper-tts** (`piper`, port 8898): piper-tts 1.7.0 loaded
  directly - no Wyoming process at all (the old host route was a Wyoming client
  to a separate server; the in-process engine keeps the identical pins and voice
  directory, `data/external/piper/voices/`, and the serial one-voice-resident
  behaviour). The `tts-protocol` client is what the trainer venv carries of this
  whole stack.
- **Which TTS a MWW run uses depends on its route and its mix.** MWW's corpus is
  Piper-majority with a 30% Kokoro mix by default on the host route (the mirror
  of OWW's 0.30 Piper fraction; `KOKORO_FRACTION=0` for all-Piper) - and the
  negatives are Piper-only on both routes, so Piper is always needed.

(`src/scripts/start-kokoro-host.sh` / `start-piper-host.sh` still exist but are for
raw-API debugging now - `src/scripts/audit_voices.py` and `src/scripts/bench_tts.py` speak the
protocol, against the same engines the corpus uses - not for training.)

## Running openWakeWord


```bash
uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900   # terminal 2
uv run --project src/tts-service/engines/piper python -m piper_engine --port 8898              # terminal 3, only if --piper-fraction > 0

./src/scripts/run-oww-training-applesilicon.sh "hey seeree" --piper-fraction 0.3
```

`KOKORO_URL` defaults to the mlx engine (`tcp://127.0.0.1:8900`); `--skip-corpus`
reuses the existing corpus. What the script checks before `train.py` ever starts, and why each
check exists:

- **espeak-ng (the mlx engine)**: `brew`'s espeak-ng data must be present or the
  ENGINE refuses to start with a message that says exactly that - misaki loads
  it through a wheel that otherwise hardcodes its build path.
- **Arg validation**: every custom flag must start with `--` and unknowns must
  not collide with `train.py`'s own flags. This has caught real typos,
  including a collapsed line continuation that turned the wake word into
  `"hey seereeKOKORO_EXTERNAL=1"`.
- **`KOKORO_URL` / env-var traps**: `host.docker.internal` is a
  container-only name that tends to linger in a shell after container-path
  work; the script rewrites it to `127.0.0.1` with a notice. A leftover
  `KOKORO_RUNON_URL` is simply ignored: run-ons render through the same pool
  now. `KOKORO_EXTERNAL` is unset, since here every Kokoro is external.
- **`PIPER_URL` rewrite**: `piper:10200` (the compose service name) never
  resolves on the host, so it is rewritten to a loopback protocol URL with a
  notice; the log records the URL actually used. The protocol client accepts
  only `tcp://` specs - the old bare `host:port` / `http://` forms are coerced
  by the script with a note, and rejected outright by the client, because they
  used to mean different backends with different audio.
- **TTS preflights**: a `voices` round trip against each engine that will render
  anything; on failure they print the exact start command and exit 1. Without
  them the corpus stage dies mid-run with a raw `ConnectionRefusedError`.
- **Ownership**: the Docker trainers run as root and leave `data/corpus/`
  root-owned; a host run then fails on permissions somewhere unhelpful, so the
  script checks writability up front and prints the `sudo chown -R` fix.
- **Patch sentinel**: greps the openWakeWord clone for a marker line of the
  piper patch. This matters twice: the augmentation stage **re-invokes the same
  file** (`openwakeword/train.py --augment_clips`) as a subprocess, and a run
  on a de-patched clone dies there with `KeyError: 'piper_sample_generator_path'`
  after the corpus was already built (measured on a real run). If the guard fires,
  do not hand-patch — re-run `./src/scripts/setup-applesilicon-trainer.sh`; nothing
  in the run script ever writes to the clone.

The log lands in the repo's **logs/** as `training-<word>-macos-YYYYMMDD-HHMMSS.log`
(three stages to watch: corpus generation, feature extraction, training).

**The real signal is the footer, not the exit code.** Whether the model was
*written, and changed since before the run* (md5), is what the script checks —
a file that came back unchanged is the **previous** model, and it says so
explicitly: do not evaluate or deploy it. Two known benign end states:
`train.py` exits 1 on its own tflite conversion, which this repo replaces with
`src/train/oww/onnx2tflite.py`; and your Ctrl+C — both surface as the generic
`TRAINING FAILED (exit …)` footer. Read the log to tell those from a real
failure.

## The tflite conversion (the one thing that aborts on the host)

At the end of a run, `train.py`'s built-in "Converting to tflite" step can
**abort the whole process** on macOS arm64 (measured: it hit with the training
itself having just completed):

```
libc++abi: terminating due to uncaught exception of type
  std::__1::system_error: mutex lock failed: Invalid argument
```

That is a C++ thread-state failure after a long torch/OpenMP run, not a model
problem; do not try to fix it inside the trainer venv. The `.onnx` is already
written at this point. This repo's wrapper runs the converter
in a SUBPROCESS (`src/train/oww/train.py: convert_to_tflite`), so the SIGABRT dies
with the child and the run still exits 0 with a WARNING — but the converter
still cannot succeed in this venv (the onnx2tf stack aborts even on a cold
import), so the .tflite still comes from Docker (multi-arch, native on Apple
Silicon — `docker compose build oww-trainer` first if the image is absent):

```bash
docker run --rm -v "$PWD:/work" -w /work lva-wakeword-trainer-oww-trainer:latest \
    python /work/src/train/oww/onnx2tflite.py /work/output/<word>/oww/<word>.onnx \
    -o /work/output/<word>/oww/<word>.tflite
```

The script self-verifies — it runs the tflite and the onnx on random inputs and
refuses to write if they disagree — so overwriting a stale `.tflite` is safe.
One quirk, checked and harmless: `onnx2tf` re-saves the input `.onnx`
(metadata only — a size change there means nothing).

## Running microWakeWord
```bash
./src/scripts/setup-mww-applesilicon-trainer.sh   # once per machine; idempotent
uv run --project src/tts-service/engines/piper python -m piper_engine --port 8898      # in another terminal
uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900  # another, for the default 30% mix
./src/scripts/run-mww-training-applesilicon.sh "hey seeree"
```

`KOKORO_FRACTION=0` (or `--kokoro-fraction 0`) runs the historical all-Piper
corpus and needs only the Piper engine.

The setup script pins the `src/train/microwakeword/` clone to one commit of
the fork, builds `src/train/train-mww-applesilicon/.venv` (Python 3.12, tensorflow
2.21.0 — the version the image installs — numpy 2, which is why this cannot
share the openWakeWord venv), and verifies the imports. It installs the clone
**editable `--no-deps`**; a non-editable build is a verified failure, because
`microwakeword/audio/` has no `__init__.py` and `find_packages()` silently
drops it from the wheel. The run script preflights a real `describe` / voices
round trip to each TTS server it will use before spending the run, checks the shared `data/corpus`/`output`
directories are writable (the Docker trainers run as root), and verifies the
clone is still at the pinned commit. Same knobs as the container path:
`SKIP_CORPUS=1`, `SKIP_FEATURES=1`, `MAX_FAPH=…`. The log lands in the repo's
logs/ dir as `training-<word>-macos-<stamp>.log`.


## What survives a run, and what doesn't

- The trainers `rmtree` their corpus **every run** — `data/corpus/` is
  generated material. `data/recordings/` (samples + holdout) is what the
  trainers read and is untouched. Never park personal data in `data/corpus/`.
- A tag is code + data: `<commit>-d<audio hash>`. The data half moves between
  runs even when nothing changed (TTS is not bit-reproducible). A different
  tag is not a different experiment.
- `output/<word>/oww/<word>.onnx` (and, after the container conversion,
  `.tflite`) is the artifact. If the run script told you the onnx is
  *unchanged*, it is the previous model, whatever the tag next to it says.

## Then

Evaluate — that is the `eval-models` skill, and it runs natively on the same
Mac. Before shipping anything to a phone, the preflight on a real microphone
needs a human; that part you hand over, you do not run.
