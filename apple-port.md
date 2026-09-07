# Running the trainers on Apple Silicon

A note on what it would take, and why it has not been done. Written after asking
whether MLX has equivalents to the CUDA speed-ups the training images rely on.

**Phases 1a, 1b and 3 are measured; phase 2 is not started.** The numbers below
are from this Mac (M1 Max, 64 GB, 10 cores) unless they say they come from the
training VM (20 GB RAM, RTX 3090, 4 cores) or from baselines recorded in
`docker/Dockerfile.mww.cuda`. Where a measurement is still open, the section says
so.

## Status, 2026-09-04: the open questions are answered, and two of them flipped

The original recommendation was "do not port". It was correct on what was known then;
three things have changed since, and the plan at the bottom replaces it.

| Then | Now |
|---|---|
| Target Mac unknown; below ~32 GB the port is pointless | **M1 Max, 64 GB, 10 cores, macOS 26.5.2** - the good branch |
| tensorflow-metal / TF 2.21.0 pairing unknown | **Does not exist, and is not coming.** See below |
| A run took 83 minutes and thrashed swap | **oww 28m59s, mww 28m02s** on the 3090, both post-fix |

The 64 GB is the one that matters most, because it deletes work rather than adding
it: the swap problem `gpu-resident-features.py` exists to solve cannot occur, so
that patch is simply not applied. See "gpu-resident-features.py becomes unnecessary".

Disk is fine too - 148 GB free, against ~18 GB for the openWakeWord half of
`data/external/` (~17.2 GB ACAV100M plus ~730 MB shared).

**The result so far, and it is not the one this document expected.** microWakeWord
finished in **26m06s on the Mac against 28m03s on the RTX 3090** - CPU only, in
Docker, no Metal anywhere. The two things the GPU actually bought openWakeWord were
never compute (see "Whether it is worth doing at all"), and for a 25,537-parameter
model they were not worth much at all.

**And the biggest lever found so far is not a GPU question.** It is that the same
library is markedly faster outside a container than inside one on this hardware -
3.7x for Kokoro, 6.3x on a GEMM-shaped torch train step - because the macOS wheels
link Accelerate. That cuts the other way for convolutions, where the macOS build has
no oneDNN and is 12x slower. Phase 1b has the numbers; it is the section to read
before optimising anything here.

## MLX is not a drop-in

(Written when this was the main finding. It has been overtaken: the largest measured
effect on this hardware is the container/host library split in Phase 1b, which needs
no new framework at all. This section still stands on its own terms.)

MLX is a separate array framework. openWakeWord trains with **PyTorch** and
microWakeWord with **TensorFlow**, and neither has an MLX backend. Adopting MLX
means porting the training loops — and on the openWakeWord side that means
reimplementing the upstream `train.py` that `patches/` already modifies in five
places. The patches are not cosmetic: one of them is the only reason
`augmentation_rounds > 1` produces more data rather than discarding it.

So MLX is a rewrite, not a swap. The realistic path is the per-component one below.

## What the CUDA dependency actually consists of

| CUDA piece | What it buys | Apple counterpart |
|---|---|---|
| `torch torchaudio` cu121 | oww model training | PyTorch **MPS** backend |
| `onnxruntime-gpu` | melspectrogram + embedding inference | onnxruntime **CoreML** EP |
| `patches/gpu-resident-features.py` | 17.28 GB feature array out of swap | **unified memory** — see below |
| `tensorflow[and-cuda]==2.21.0` | mww training | **tensorflow-metal** — the weak link |

### feature-device-selection.py ports almost for free

That patch exists because upstream picked the feature-computation device by asking
*torch* about a decision belonging to *onnxruntime* — so on a CUDA box with the CPU
build of onnxruntime it requested a provider that did not exist AND set `ncpu=1`,
giving single-threaded CPU inference: 2.46 it/s, 36 minutes of an 83-minute run.

The fix keys off `onnxruntime.get_available_providers()` instead, which is
framework-agnostic by construction. Retargeting it to `CoreMLExecutionProvider` is a
change to the provider name, not to the logic.

### gpu-resident-features.py becomes unnecessary, or catastrophic

This is the one place where Apple Silicon is architecturally better rather than
merely adequate.

The patch exists because `mmap_batch_generator` walks a (5625000, 16, 96) float16
array — 17.28 GB — sequentially against 20 GB of RAM, so each pass evicts what the
next one needs and the kernel swaps to keep up: GPU at 14%, CPU 37% idle, 7.2 GB of
8 GB swap in use. Moving the bytes into VRAM removed the stall.

Apple Silicon has no separate VRAM. The array is simply in memory, reachable by both
CPU and GPU, so "move it to the GPU" is meaningless — and so is the problem it
solves, PROVIDED THE MACHINE HAS THE RAM. On a 36-48 GB Mac this is a real win: the
patch could be skipped entirely. On a 16 GB Mac it would thrash considerably worse
than the VM did, because there is no second 24 GB pool to escape into. There is no
middle ground here; it is a function of installed memory.

## Whether it is worth doing at all

The measurements already in this repo argue mostly no.

**microWakeWord barely benefits from a GPU as it is.** `docker/Dockerfile.mww.cuda`
records a CPU baseline of ~46 s per 500 steps on a 25,537-parameter model — about 15
minutes for a 10,000-step run — and notes that a model this small may not fill a
GPU. It will not fill an M-series GPU either. The CPU path is already acceptable,
which makes tensorflow-metal's weakness much less important than it first looks.

**openWakeWord's wins were not raw compute.** They were (1) not falling back to
single-threaded feature computation and (2) not swapping. Both are reachable on a
Mac with multi-threaded CPU onnxruntime and enough RAM, with no GPU framework
involved.

The TTS side is the standing warning about assuming a GPU helps. `train/oww/train.py`
records a Kokoro process pinned at 101.8% CPU — exactly one core — while the GPU sat
at 21% and VRAM at 1.5 of 24 GB: the bottleneck was a serialised stage, not compute,
so the accelerator was mostly idle. `docker/Dockerfile.piper` carries that forward and
declines to bake `--use-cuda` in, calling the GPU's value an open question rather than
an obvious win. The same caution applies to MPS and Metal: *"it initialised"* and
*"it helped"* are different claims, and only the second one matters.

## The practical blocker

Both images are `linux/amd64` CUDA bases (`nvidia/cuda:12.1.1-devel-ubuntu22.04`,
`tensorflow/tensorflow:2.21.0-gpu`). Running on a Mac means not using them at all —
a native host environment of the kind `record/` and `preflight/` already have, each
with its own `pyproject.toml` and uv lock. That is a third and fourth host
environment to maintain, not a compose overlay.

`docker-compose.cuda.yml` is not a precedent for this. It selects a different base
image for a service that is *pulled*, not built; it cannot make a CUDA training
image run natively on arm64.

## Answered: tensorflow-metal is dead, so microWakeWord gets no GPU on macOS

`tensorflow-metal` **1.2.0**, uploaded **2025-01-31**, is the newest release. The mww
image pins TensorFlow **2.21.0**, uploaded **2026-03-06** — five minor versions and
thirteen months later, with no tensorflow-metal release in that window.

Its metadata cannot be used to argue otherwise, and is worth naming because it looks
like it should settle the question:

    requires_dist: ['wheel~=0.35', 'six>=1.15.0']

It declares no TensorFlow dependency at all. That is not "compatible with every
version" — it is a plugin loaded through TensorFlow's `PluggableDevice` ABI, which
metadata does not describe. This repo already paid for reading `requires_dist` as
truth once, in `docker/requirements.txt`. The environment is the authority.

So the microWakeWord half is **CPU-only on macOS, permanently as far as anyone can
plan.** Per the section above, that is close to fine: ~46 s per 500 steps, about 15
minutes for a 10,000-step run, on a model of 25,537 parameters. Ten M1 Max cores
should beat the four-core VM figure that baseline came from.

## The plan

The shape follows from one fact the compose overlay already records: **Docker Desktop
does not pass the Metal device through.** Anything wanting the GPU must run on the
host. So the port is not an overlay, and `docker-compose.mps.yml` will never have
content — its eventual fate is deletion, replaced by a pointer to this document.

But only the *trainer* has to move. The split:

| Stage | Where it runs on a Mac | Work needed |
|---|---|---|
| Corpus generation (Kokoro, Piper) | Docker service, or a host uv venv - no GPU either way | Kokoro: the 3.7x host win below made the host route worthwhile; Piper: host server verified 2026-09-07 |
| Training | **host, uv env** | this document |
| Eval | **Docker, unchanged** | none |

Corpus generation needs no *port* because neither TTS server wants a GPU here
anyway: the compose default is `ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1` and Piper
is started without `--use-cuda` by choice. Both already run on arm64. A run is
agnostic about which of those it is - `--kokoro-url` / `--piper-url` point at
whatever answers.

What Phase 1b settled is the *where*. The same Kokoro runs 3.7x faster in a host
uv venv than in a container on this hardware, so the host trainer takes its corpus
from `scripts/start-kokoro-host.sh` rather than a container. Piper followed the same
shape on 2026-09-07: `scripts/start-piper-host.sh` is one uv venv pinned to the
same `wyoming-piper`/`piper-tts` as `docker/Dockerfile.piper`, voices downloading
on demand next to it. Verified on this Mac that day: describe enumerates the full
163-voice catalog (2,005 en_US/en_GB pairs), a first render lands in 1.2 s wall, a
warm one renders 1.55 s of audio in 0.1 s, and an unseen voice downloads and
renders in 7.2 s including the 63 MB download. Measured 2026-09-07, in the
README's timing table: the Kokoro corpus is 30m29s with the host FastAPI server
and 19m06s/24m54s in-process (`--kokoro-url mlx://`), and the README's estimate
now assumes the in-process route because it generates the corpus 1.2-1.6x
faster; the open remainder is its run-on regression, documented in the README's
Kokoro section.

### Phase 1 — CPU images, in Docker, and measure them

**Decided 2026-09-04: the baseline is Docker, not a host environment.** The repo's
stated design choice is Docker only, and CPU is reachable inside a container while
Metal is not — so the CPU baseline has no reason to leave. `docker/Dockerfile.oww.cpu`
and `docker/Dockerfile.mww.cpu`, selected by `docker-compose.cpu.yml`.

Do this on **CPU only** first and measure it. Not caution — sequencing. It is the
cheapest experiment in the document, and if a 10-core M1 Max is fast enough on CPU
then phases 2 and 3 are optional rather than load-bearing. Measuring MPS against a
CPU baseline you never took is how "it initialised" gets mistaken for "it helped".

Four things the CPU images had to solve that the plan did not anticipate. All four
are arm64-only, and all four were invisible on x86 — which is the general lesson:
"the same requirements file" does not mean "the same resolution".

* **`tensorflow/tensorflow:2.21.0` is amd64-only** — verified with
  `docker manifest inspect`, a single linux/amd64 manifest and no arm64 variant. So
  the mww image cannot keep its base at all; it builds on `python:3.12-slim` with
  TensorFlow from PyPI, which does publish `manylinux_2_27_aarch64` wheels.
* **Python 3.12 is an intersection, not a default.** tensorflow ships aarch64 wheels
  for cp310–cp313, ai-edge-litert for cp310–cp314, and `pymicro-features` as a single
  `cp39-abi3` aarch64 wheel. 3.12 is inside all three. That also settles the warning
  in `Dockerfile.mww.cuda` that `pymicro-features` might need upstream's macOS fork —
  on linux/arm64 it does not.
* **`pip install torch` on linux/aarch64 is not a CPU install.** It resolves the same
  CUDA-enabled wheel as on x86 and pulls the runtime with it — observed mid-build,
  `cuda_toolkit` plus `nvidia_cudnn_cu13` at 651 MB, none of which can ever load on a
  Mac. Those wheels exist for NVIDIA's own ARM machines. The `+cpu` wheels from
  PyTorch's CPU index carry no `nvidia-*` at all, and `Dockerfile.oww.cpu` now asserts
  that none are present rather than trusting it.
* **`numpy<2` and `tensorflow==2.21.0` between them make onnx2tf unresolvable on
  arm64.** Every onnx2tf 2.x pins either numpy 2 (2.3+) or tensorflow 2.19.0
  (2.0.23–2.2.2), so pip lands on 1.29.24, which pins `onnxsim==0.4.36` — the one
  package with no aarch64 wheel, whose sdist wants cmake. The resolution is not to
  install cmake: `train/oww/onnx2tflite.py` already runs `onnx2tf ... -nuo`, which is
  `--not_use_onnxsim`, so the dependency is dead weight here. It is installed
  `--no-deps` with `onnxsim-prebuilt` supplying the module from a wheel.

That last one is worth generalising. The repo already learned from
`docker/requirements.txt` that *declared* and *installed* are different things. arm64
adds a second gap: *installed on x86* and *installable on arm64* are different things
too, and neither `requires_dist` nor a working amd64 build predicts the other.

`train-applesilicon/` holds a resolved `pyproject.toml` and `uv.lock` for the host route. uv
resolved the pinned openWakeWord stack (speechbrain 0.5.14, datasets 2.14.6,
`numpy<2`) on macOS/CPython 3.12 in 143 packages, which settled the open question of
whether a host env was even possible.

It was kept for phase 2, on the assumption that only MPS would justify it. Phase 1b
below overtakes that: the host torch is **6.3x faster than the container's on a
GEMM-shaped train step**, on CPU, with no GPU involved at all. So this directory is
now the most likely home of the next real openWakeWord speedup rather than a
placeholder for a deferred one. Do not delete it.

The patch set survives this almost intact, which is the main way MPS differs from
the MLX rewrite analysed above — MPS keeps PyTorch, so the patches keep applying:

| patch | on macOS |
|---|---|
| `configurable-corpus-dir.py` | unchanged |
| `honour-augmentation-rounds.py` | unchanged — and load-bearing; it is the only reason `augmentation_rounds > 1` adds data |
| `skip-piper-import.py` | unchanged |
| `feature-device-selection.py` | one provider name, see phase 2 |
| `gpu-resident-features.py` | **not applied** — 64 GB makes it unnecessary |

Two of five touched, one of those by deletion.

### Phase 1b — the container is the wrong place for openWakeWord, and the right one for microWakeWord

**This is the finding that most changes the plan, and it arrived sideways.** Chasing
Kokoro throughput turned up a 3.7x gap between the same Kokoro-FastAPI version in a
container and on the host — native arm64 both times, no emulation. That is a *torch*
gap, and both trainers were built on the assumption that it does not exist.

Probing torch directly, identical script, container against `train-applesilicon/`:

| operation | container (linux/arm64) | host (macOS arm64) | |
|---|---|---|---|
| `matmul 2048³` | 53.05 ms | **17.42 ms** | host **3.0x** |
| train step, 1024 batch, MLP | 26.39 ms | **4.19 ms** | host **6.3x** |
| `conv1d 512×16×96` | **7.38 ms** | 90.16 ms | host **12x SLOWER** |

The conv row is not noise — three trials, 88.15 / 90.44 / 88.39 ms — and the cause
is one flag:

    threads 8 | mkldnn False        # host
    linear equiv: 0.48 ms           # same tensor through a GEMM path

**The macOS wheel ships without oneDNN, so convolutions fall back to a reference
kernel; the linux wheel has it. Meanwhile GEMM on the host goes to Accelerate.** Two
libraries, two ops, opposite winners. "Go native on Apple Silicon" is not a blanket
improvement, and anyone who asserts it without measuring will be right about half the
model and badly wrong about the other half.

Which half you land on is decided by architecture:

* **openWakeWord is `Linear` x7 and one `LSTM` — no convolutions at all** (counted in
  its own `train.py`). Pure GEMM, so the host's missing oneDNN costs nothing and
  Accelerate is a straight win. `docker/Dockerfile.oww.cpu` is therefore running the
  slower of the two torches for the one model it trains.
* **microWakeWord is mixednet — convolutional.** On host torch that is the 12x
  reference-kernel path. It is TensorFlow rather than torch so this probe does not
  transfer directly, and that turned out to matter: measured in TF (phase 3), the
  full mixednet step is 1.17x FASTER on the host, and the corpus stage's Piper is
  2.4x faster. The conv row above remains true for torch; it was the wrong
  framework to argue from.

**Before acting on the 6.3x**, two things it does not yet prove. The probe is a
synthetic MLP, not openWakeWord's real model — the LSTM is a third code path that
neither Accelerate nor oneDNN covers cleanly. And the training loop is a minority of
an openWakeWord run: TTS and onnxruntime feature computation dominate, and the
training stage itself is ~16 minutes. A 6.3x on a minority stage is worth much less
than it sounds.

The honest next step is the cheap one: run the actual openWakeWord model through both
environments and compare wall time on the training stage alone, with `SKIP_CORPUS=1`
so nothing else moves. `train-applesilicon/` exists for precisely this and is otherwise
unused.

### Phase 2 — MPS for the model, CoreML for the features, measured separately

Separately, because they are different mechanisms with different failure modes and
adopting them together makes a regression unattributable.

**Features (onnxruntime → CoreML).** `feature-device-selection.py` already keys off
`onnxruntime.get_available_providers()` rather than torch, which is exactly the
framework-agnostic form this needs; adding `CoreMLExecutionProvider` to the
preference list is the change. Verify the provider actually loads rather than
trusting the list — `report_onnx_providers()` in `train/oww/train.py` already makes
this distinction and explains why it matters, and that logic is portable as-is.

**Model (torch → MPS).** `torch.device("mps")`. No special wheel: the macOS arm64
PyTorch wheels carry MPS. The risk here is not speed, it is silent numerical
difference or an unimplemented operator falling back to CPU, so the check is that
the trained model still converts and still evaluates — the pipeline already has that
instrument, in `onnx2tflite.py`'s conversion scoring and in `eval/compare_models.py`.

### Phase 3 — microWakeWord: measured, and the host wins

Written as "same host-environment treatment, no Metal, expect the easy half". The
"easy half" was actually the hard measurement, because the phase 1b conv result
pointed the other way — and it was torch, so it did not transfer. Measured now,
2026-09-07, with `tools/tf_probe.py`: batch 128, the actual op shapes in
`train/mww/train.py` (first conv (5,1) s3, the three MDConv blocks, a 1024³ GEMM,
and a full train step forward+grad+update), the same `tensorflow==2.21.0` source
build in both environments.

| operation | container (linux/arm64) | host (macOS arm64) | |
|---|---|---|---|
| first conv (5,1) s3, 150 frames | 3.34 ms | 2.92 ms | host 1.14x |
| MDConv block 1 (32ch) | 3.50 ms | 2.72 ms | host 1.29x |
| MDConv block 4 (1024ch) | 8.37 ms | 7.26 ms | host 1.15x |
| GEMM 1024³ | 20.67 ms | 16.09 ms | host **1.29x** |
| full train step (fwd+grad+update) | **36.21 ms** | **30.86 ms** | host **1.17x** |

And `threading_options` made **both** sides slower (host 33.90, container 39.80
with 10 threads): both builds run one CPU per op at these shapes, so the win is
the BLAS in the GEMM-bound residual, the same Accelerate story as phase 1b's
matmul — smaller, because mixednet's convolutions are too skinny to feed a GEMM
kernel, which is why phase 1b's conv prior looked the way it did.

The bigger number is the corpus stage. `tools/bench_tts.py`, the same
`wyoming-piper` 2.4.3 / `piper-tts` 1.7.0 in both environments, 140 "hey seeree"
clips:

| | container | host | |
|---|---|---|---|
| Piper, sequential | 9.13 clips/s | **21.66 clips/s** | host **2.4x** |
| Piper, 8 client threads | 9.13 (flat) | 23.4 | still serialises |

Same server code, same model, 2.4x: `piper-tts` runs on onnxruntime, and the
macOS wheel links Accelerate while the linux/arm64 one does not — the identical
mechanism to every other number in this document. The corpus stage is the
longest in a full run (roughly 14 of the container run's 26m06s), so it carries
most of the end-to-end gain. Full run on this machine: the host's, 2026-09-07,
against the container's 2026-09-06 run - a day apart, so per the day rule in
the README's Kokoro section, read the total as directional and the same-day
stage probes as the evidence. Container stage times are that run's log
timestamps:

| stage | container | host | |
|---|---|---|---|
| corpus (4,920 + 984 clips) | ~14 min | **6m12s** | 2.3x |
| features | ~1 min | 1m22s | about the same |
| train, 10,000 steps + conversion + ROC | ~11 min | **6m40s** | 1.7x |
| **total** | **26m06s** | **14m14s** | **1.83x** |

The training stage beat its 1.17x probe, because the conversion and ROC
calibration that follow it run in the same process and carried the same gap.
The container route stays correct everywhere else; on this Mac the host route
is the faster one, measured, not projected.

What it added:

* `train-mww-applesilicon/` — the host uv environment: Python 3.12,
  `tensorflow==2.21.0` (the version `Dockerfile.mww.cpu` installs from PyPI),
  `numpy>=2`, `pymicro-features==2.0.2`. A fourth host environment because it
  cannot share a venv with `train-applesilicon/` (numpy 2 vs numpy<2 — the same
  split that keeps the two trainer images apart).
* `scripts/setup-mww-applesilicon-trainer.sh` — clones
  `OHF-Voice/micro-wake-word` at repo root, pinned to `4665173`, builds the venv.
  The clone installs **editable `--no-deps`**, and that is a verified failure
  mode, not a preference: `microwakeword/audio/` has no `__init__.py`, so a
  non-editable wheel build drops the whole subpackage (`find_packages()`
  silently skips it) and the features stage dies on `ModuleNotFoundError` two
  stages in.
* `scripts/run-mww-training-applesilicon.sh` — the same four stages, on the
  host, with `PIPER_URL` reaching the host Piper (`./scripts/start-piper-host.sh`
  on 127.0.0.1:10200; a `piper:PORT` value is the compose-only name and is
  rewritten, as the oww script does).

Verified on this machine: every import the stages touch; the features-stage
path against the real corpus (RaggedMmap + augmentation); a 600-step training
run through quantised streaming TFLite conversion and the ROC, both directly
and through the run script end to end; and a full run - the 14m14s row above.
Open: the host-vs-container comparison on any Mac but this one.

One caveat that would change the training-stage conclusion if it ever bites:
the container side was measured in this Docker Desktop VM, and its CPU share is
a VM setting, not a constant of the hardware. The TTS gap does not move with
that — it is a library link, not a schedule.

### What would make this not worth finishing

Written as "stop if phase 1 shows CPU training too slow for the feedback loop". Phase
1 has now answered that for microWakeWord — 26m06s, faster than the 3090 — so the
question survives only for openWakeWord, where the corpus stage rather than training
is the thing that hurts.

The bar was never the 3090's ~29 minutes. It is whatever keeps a change-one-thing
iteration inside a sitting, and the levers that move it most are the ones found by
accident: `SKIP_CORPUS=1`, and getting the TTS off the container's libraries.

The standing warning from the TTS side applies to every phase here: `train.py`
records a Kokoro process pinned at 101.8% CPU — one core — while the GPU sat at 21%
and VRAM at 1.5 of 24 GB. The bottleneck was a serialised stage, not compute, and
the accelerator was mostly idle. *"It initialised"* and *"it helped"* are different
claims, and only the second one matters.
