---
name: train-apple-silicon
description: Train wake-word models on an Apple Silicon Mac. openWakeWord runs natively on the host (uv venv, in-process Kokoro via MLX on the GPU, Piper TTS from a host server); microWakeWord runs natively on the host too (uv venv + host Piper, faster than the container) or in the multi-arch Docker CPU image. Use when the user wants to train or retrain a model on a Mac, set up the trainer environment, start the host TTS services, hits a failure in the Apple Silicon training scripts, or asks why training on a Mac differs from the CUDA box.
---

# Training on Apple Silicon

Step 2 of the pipeline.
## Setup, once per machine

```bash
./scripts/setup-applesilicon-trainer.sh
```

Idempotent; re-run it any time something downstream smells wrong. It clones
openWakeWord at the pinned commit, applies the patches in `patches/`, builds
`train-applesilicon/.venv` (Python 3.12), downloads the spacy model, and
verifies.

## The TTS services

- **Kokoro, in-process MLX**: `--kokoro-url mlx://` on the run
  script. No server, no `KOKORO_URL`; the same Kokoro-82M model renders inside
  the training process, on the GPU, and if any other URL is in the pool it is
  ignored. Word timestamps are exact (agreed with the server to 3 ms on the
  same phrase — durations are predicted per phoneme and rendered from them, so
  run-on cuts come from measurement, not the estimation fallback), which is
  what the run-on cuts are built on. 
- **Piper, host** The Wyoming `describe` event returns the **whole** embedded voice catalog
  (163 entries in the pinned `wyoming-piper` 2.4.3), not just what is
  downloaded — a fresh install answers a preflight instantly, then downloads
  each voice on demand (~7 s for the first of each, 0.1 s after). Voices persist
  in `data/external/piper/voices/` across rebuilds; `--venv-only` rebuilds the
  venv without touching them.
- **Which TTS a MWW run uses depends on its route and its mix.** MWW's corpus is
  Piper-majority with a 30% Kokoro mix by default on the host route (the mirror
  of OWW's 0.30 Piper fraction; `KOKORO_FRACTION=0` for all-Piper) - and the
  negatives are Piper-only on both routes, so Piper is always needed.


## Running openWakeWord


```bash
./scripts/start-piper-host.sh        # only if --piper-fraction > 0

./scripts/run-oww-training-applesilicon.sh "hey seeree" --kokoro-url mlx:// --piper-fraction 0.3
```

`--kokoro-url mlx://` selects the in-process MLX engine (28 voices, GPU, no
`KOKORO_URL`). `--skip-corpus` reuses the existing corpus. What the script checks before `train.py` ever starts, and why each
check exists:

- **espeak-ng (mlx path)**: `brew`'s espeak-ng data must be present or the run
  dies at clip 1 with a `/Users/runner/...` path that mentions neither TTS nor
  MLX — the script checks for it up front and says `brew install espeak-ng`.
- **Arg validation**: every custom flag must start with `--` and unknowns must
  not collide with `train.py`'s own flags. This has caught real typos,
  including a collapsed line continuation that turned the wake word into
  `"hey seereeKOKORO_EXTERNAL=1"`.
- **`KOKORO_URL` / env-var traps** (server path): `host.docker.internal` is a
  container-only name that tends to linger in a shell after container-path
  work; left, it fails with a DNS error ("no usable Kokoro servers") — the
  script rewrites it to `localhost` with a notice. A leftover
  `KOKORO_RUNON_URL` is simply ignored: run-ons render through the same pool
  now. `KOKORO_EXTERNAL` is unset, since here every Kokoro is either in-process
  or a server you pointed at.
- **`PIPER_URL` rewrite**: `piper:10200` (the compose service name) never
  resolves on the host, so it is rewritten to `127.0.0.1:10200` with a notice;
  the log records the URL actually used.
- **Piper preflight**: one real `describe` round-trip; on failure it prints the
  fix (start the service) and exits 1. Without it the corpus stage dies
  mid-run with a raw `ConnectionRefusedError`.
- **Ownership**: the Docker trainers run as root and leave `data/corpus/`
  root-owned; a host run then fails on permissions somewhere unhelpful, so the
  script checks writability up front and prints the `sudo chown -R` fix.
- **Patch sentinel**: greps the openWakeWord clone for a marker line of the
  piper patch. This matters twice: the augmentation stage **re-invokes the same
  file** (`openwakeword/train.py --augment_clips`) as a subprocess, and a run
  on a de-patched clone dies there with `KeyError: 'piper_sample_generator_path'`
  after the corpus was already built (measured 2026-09-07). If the guard fires,
  do not hand-patch — re-run `./scripts/setup-applesilicon-trainer.sh`; nothing
  in the run script ever writes to the clone.

The log lands in the **repo root** as `training-<word>-macos-YYYYMMDD-HHMMSS.log`
(three stages to watch: corpus generation, feature extraction, training).

**The real signal is the footer, not the exit code.** Whether the model was
*written, and changed since before the run* (md5), is what the script checks —
a file that came back unchanged is the **previous** model, and it says so
explicitly: do not evaluate or deploy it. Two known benign end states:
`train.py` exits 1 on its own tflite conversion, which this repo replaces with
`train/oww/onnx2tflite.py`; and your Ctrl+C — both surface as the generic
`TRAINING FAILED (exit …)` footer. Read the log to tell those from a real
failure.

## The tflite conversion (the one thing that aborts on the host)

At the end of a run, `train.py`'s built-in "Converting to tflite" step can
**abort the whole process** on macOS arm64 (measured 2026-09-07, the
12:43 run — training itself had just completed):

```
libc++abi: terminating due to uncaught exception of type
  std::__1::system_error: mutex lock failed: Invalid argument
```

That is a C++ thread-state failure after a long torch/OpenMP run, not a model
problem; do not try to fix it inside the trainer venv. The `.onnx` is already
written at this point. Convert in the Docker image that carries the full
tensorflow + onnx2tf stack (multi-arch, native on Apple Silicon —
`docker compose build oww-trainer` first if the image is absent):

```bash
docker run --rm -v "$PWD:/work" -w /work lva-wakeword-trainer-oww-trainer:latest \
    python /work/train/oww/onnx2tflite.py /work/output/<word>/oww/<word>.onnx \
    -o /work/output/<word>/oww/<word>.tflite
```

The script self-verifies — it runs the tflite and the onnx on random inputs and
refuses to write if they disagree — so overwriting a stale `.tflite` is safe.
One quirk, checked and harmless: `onnx2tf` re-saves the input `.onnx`
(metadata only — a size change there means nothing).

## Running microWakeWord
```bash
./scripts/setup-mww-applesilicon-trainer.sh   # once per machine; idempotent
./scripts/start-piper-host.sh                 # in another terminal
./scripts/start-kokoro-host.sh                # another, for the default 30% mix
./scripts/run-mww-training-applesilicon.sh "hey seeree"
```

`KOKORO_FRACTION=0` (or `--kokoro-fraction 0`) runs the historical all-Piper
corpus and needs only the Piper server.

The setup script pins the `microwakeword/` clone at repo root to one commit of
the fork, builds `train-mww-applesilicon/.venv` (Python 3.12, tensorflow
2.21.0 — the version the image installs — numpy 2, which is why this cannot
share the openWakeWord venv), and verifies the imports. It installs the clone
**editable `--no-deps`**; a non-editable build is a verified failure, because
`microwakeword/audio/` has no `__init__.py` and `find_packages()` silently
drops it from the wheel. The run script preflights a real `describe` / voices
round trip to each TTS server it will use before spending the run, checks the shared `data/corpus`/`output`
directories are writable (the Docker trainers run as root), and verifies the
clone is still at the pinned commit. Same knobs as the container path:
`SKIP_CORPUS=1`, `SKIP_FEATURES=1`, `MAX_FAPH=…`. The log lands in the repo
root as `training-<word>-macos-<stamp>.log`.


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
