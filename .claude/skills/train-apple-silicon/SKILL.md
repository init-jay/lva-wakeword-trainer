---
name: train-apple-silicon
description: Train wake-word models on an Apple Silicon Mac. openWakeWord runs natively on the host (uv venv, in-process Kokoro via MLX on the GPU, Piper TTS from a host server); microWakeWord runs natively on the host too (uv venv + host Piper, faster than the container) or in the multi-arch Docker CPU image. Use when the user wants to train or retrain a model on a Mac, set up the trainer environment, start the host TTS services, hits a failure in the Apple Silicon training scripts, or asks why training on a Mac differs from the CUDA box.
---

# Training on Apple Silicon

Step 2 of the pipeline. The two targets run on **two different machines**:

| target | where | why |
|---|---|---|
| openWakeWord | **on the host**, the `train-applesilicon/` uv venv | TTS is in-process Kokoro on the GPU — MLX, 25 ms/clip batched, against 88 for the same model behind the fastest server — and the trainer is torch on Accelerate, 6.3× faster steps than the container's wheel, on a model that is `Linear`×7 + one LSTM, so the one thing the macOS wheel lacks (oneDNN) costs it nothing |
| microWakeWord | **on the host**, the `train-mww-applesilicon/` uv venv (Docker `cpu` overlay also works) | Full run measured 14m14s on the host against 26m06s in the container (1.8×), same library versions: its corpus is 100% Piper, and the host Piper is 2.4× faster (21.66 vs 9.13 clips/s); the TF train stage is 1.7× faster. SPEED.md |

**The GPU does exactly one job here: Kokoro.** That is the measured layout,
not an assumption. In-process MLX is the fastest TTS path this machine has —
plain 25 ms/clip, run-ons 63 ms/clip (batch of 10, measured on this M1 Max
against the same model behind its fastest server configuration, 88 / 229) —
and it removed the old run-on/plain asymmetry, which is why the second Kokoro
server (`KOKORO_RUNON_URL`) no longer exists. Everything else is CPU
*demonstrably*: the torch trainer, because the macOS wheel is the 6.3× measured
win, and the Metal route is closed rather than pending (no device passes through
Docker, and tensorflow-metal does not pair with TF 2.21.0); and the MWW host route,
because its two measured gaps — the Piper corpus at 2.4× and the TF train step at
1.17× — are both host wins at the same versions (SPEED.md),
and neither wants a GPU. Docker Desktop passes **no Metal device through** —
which is why `docker-compose.mps.yml` is empty, why anything that wants the
GPU has to run on the host, and why the host route is this skill, not an
overlay. Two boundaries to keep straight: the *server's* torch-MPS mode loses
on batched (0.68×, its ISTFT layers stay on CPU) — history, which is what made
MLX the winner; and MLX as a *trainer* is not on this path at all — a different array framework
with no PyTorch/TensorFlow backend, so adopting it means porting the upstream
training loops that `patches/` already modifies in five places, one of them the only
reason `--augmentation-rounds > 1` produces data at all. MLX here is used the
healthy way: as the TTS engine inside the training process.

Measured on this machine (M1 Max, 10 cores, 64 GB): the full-corpus run of
2026-09-07 used in-process MLX — 25 ms/clip batched (plain), 63 (run-ons) — so
the Piper fraction (~1.55 s/clip at 0.7× speed) is now the slow TTS half, and
feature extraction is the stage the trainer itself calls "the slowest stage of
the run". A `--skip-corpus` re-run (features + 50,000 training steps, through
the tflite crash) took ~18 minutes on 2026-09-07. A full corpus is a long
afternoon, not an overnight.

## Setup, once per machine

```bash
./scripts/setup-applesilicon-trainer.sh
```

Idempotent; re-run it any time something downstream smells wrong. It clones
openWakeWord at the pinned commit, applies the patches in `patches/`, builds
`train-applesilicon/.venv` (Python 3.12), downloads the spacy model, and
verifies.

**The `VIRTUAL_ENV` trap (cost a training run, 2026-09-07):** if the *preflight*
venv is activated — or `VIRTUAL_ENV` points at any venv — bare `uv pip install`
resolves that environment first, *not* the project's `.venv` (`uv sync` is
project-anchored and unaffected). The symptom is subtle: the install succeeds,
the editable openWakeWord lands in the wrong venv, and the trainer venv silently
loses it. The setup script pins `VIRTUAL_ENV` to its own venv on every uv
invocation, so the script itself is safe — but any **ad-hoc** `uv pip ...` you
type outside it must unset `VIRTUAL_ENV` or point it at the venv you mean.

## The TTS services

- **Kokoro, in-process MLX — the default**: `--kokoro-url mlx://` on the run
  script. No server, no `KOKORO_URL`; the same Kokoro-82M model renders inside
  the training process, on the GPU, and if any other URL is in the pool it is
  ignored. Word timestamps are exact (agreed with the server to 3 ms on the
  same phrase — durations are predicted per phoneme and rendered from them, so
  run-on cuts come from measurement, not the estimation fallback), which is
  what the run-on cuts are built on. Three honest caveats, from
  `train/corpus/kokoro_mlx.py`, before using it for a real corpus:
  - **It offers 28 English voices against the server's 42.** Voice diversity is
    a corpus lever — the run prints the count; check it, don't assume.
  - **Its audio is not the server's**: bf16 weights
    (`mlx-community/Kokoro-82M-bf16`) and misaki G2P, against the server's own
    stack. Timing agrees; pronunciation is unverified — and this repo already
    excludes mispronouncing voices (`MISPRONOUNCING_VOICES`), so G2P
    differences are not cosmetic. A corpus generated this way needs an eval
    against one that was not, not an assumption.
  - It renders ~430 ms of leading silence the server does not. The timestamps
    account for it and `corpus/augment.py` trims it; only code mixing a
    pre-trim timestamp with post-trim audio would go wrong, and nothing does.
  The run script preflights what it needs: **`brew install espeak-ng`** (the
  misaki phonemiser loads it through a wheel that hardcodes its build path —
  without it the failure is a `/Users/runner/...` path error at clip 1, long
  after the run started, that mentions neither TTS nor MLX).
- **Kokoro, host server — the voice-set fallback**: `./scripts/start-kokoro-host.sh`
  (port 8880) plus `KOKORO_URL=http://localhost:8880`, when you want the 42
  voices. **CPU by default** — the fastest server configuration (88 ms/clip
  batched, 229 run-on; `--mps` batched was 161, its ISTFT layers staying on
  CPU). The old run-on-`--mps` split is history: MLX beats both cases, so the
  server is a voice-set option now, not a speed one. Do not run a second
  instance hoping to scale it on Metal: two processes share one GPU and
  serialise on it (measured 9.07 vs 8.45 clips/s) — the CUDA box runs
  `kokoro2` because there instances do scale; do not carry that habit across.
  Verify with
  `curl -s http://127.0.0.1:8880/v1/audio/speech -X POST -H 'Content-Type: application/json' -d '{"input":"hello","voice":"af_heart"}' | wc -c`
  (a few KB of audio back). The host and the Docker service render the same
  voices, so corpora are interchangeable.
- **Piper, host** — OWW only, and the one server a full run needs: 
  The Wyoming `describe` event returns the **whole** embedded voice catalog
  (163 entries in the pinned `wyoming-piper` 2.4.3), not just what is
  downloaded — a fresh install answers a preflight instantly, then downloads
  each voice on demand (~7 s for the first of each, 0.1 s after). Voices persist
  in `data/external/piper/voices/` across rebuilds; `--venv-only` rebuilds the
  venv without touching them.
- **The piper pin is audit parity, not preference**: `start-piper-host.sh` pins
  `piper-tts==1.7.0`, the exact version inside `rhasspy/wyoming-piper:2.4.3`
  (verified 2026-09-07), because the exclusion lists in `train/corpus/piper.py`
  were audited against it. A fresh install resolves 1.8.0, which changed G2P.
  Bump either the image tag or the pin only with a re-audit of `en_US` +
  `en_GB` in `piper.py`.
- **Which TTS a MWW run uses depends on its route and its mix.** MWW's corpus is
  Piper-majority with a 30% Kokoro mix by default on the host route (the mirror
  of OWW's 0.30 Piper fraction; `KOKORO_FRACTION=0` for all-Piper) - and the
  negatives are Piper-only on both routes, so Piper is always needed. The **host**
  route wants `./scripts/start-piper-host.sh` on 127.0.0.1:10200 and, at the
  default mix, `./scripts/start-kokoro-host.sh` on 127.0.0.1:8880, and
  `run-mww-training-applesilicon.sh` rewrites `piper:PORT` / host.docker.internal
  values (compose-only names) to 127.0.0.1 equivalents with a notice, then
  preflights a real round trip to each server it will use. The **Docker** route
  reaches the `piper` *compose service* on the compose network, where the host
  servers are irrelevant, and runs all-Piper (`--kokoro-fraction 0` is passed
  explicitly - the compose mww-trainer has no Kokoro service wired in yet).
  Either way TTS runs CPU-only by design, which is why the servers can stay up
  through training with no GPU-memory dance.

A full OWW run therefore needs **at most one server** (Piper, and only when
`--piper-fraction` is nonzero); `--skip-corpus` needs neither.

## Running openWakeWord

Kokoro is in-process — the script **starts no TTS servers**, and a full run
needs at most one (Piper):

```bash
./scripts/start-piper-host.sh        # only if --piper-fraction > 0

./scripts/run-oww-training-applesilicon.sh "hey seeree" --kokoro-url mlx:// --piper-fraction 0.3
```

`--kokoro-url mlx://` selects the in-process MLX engine (28 voices, GPU, no
`KOKORO_URL`); the host-server fallback above is `KOKORO_URL=http://localhost:8880`
with no `mlx://` argument. `--skip-corpus` reuses the existing corpus and needs
neither. What the script checks before `train.py` ever starts, and why each
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

**On the host** — the default on a Mac, because both measured stages win there
(full run 14m14s vs 26m06s in the container, 1.8×; SPEED.md):

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

**In Docker** — elsewhere, and the reference point for the phase-3 numbers:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml ./scripts/run-mww-training.sh "hey seeree"
```

Same four stages (corpus → features → train → manifest), then the script
**collects** the three shipped files, commit-tagged, directly into
`output/<word>/mww/`. Give Docker Desktop enough RAM first: the 17.28 GB
feature array is mmap'd, and running short of memory **page-faults rather
than erroring** — a Mac that feels like it is swapping for hours has hit
this. Knobs: `SKIP_BUILD=1` (reuse the image — needed after a
`docker builder prune`), `SKIP_CORPUS=1` (skip the expensive TTS stage),
`SKIP_FEATURES=1` (only if the corpus is unchanged), `MAX_FAPH=…` (the
false-accepts-per-hour budget the manifest cutoff is chosen from).
The log lands in the repo root as `training-mww-<word>-<stamp>.log`.

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
