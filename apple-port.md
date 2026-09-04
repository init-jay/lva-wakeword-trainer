# Running the trainers on Apple Silicon

A note on what it would take, and why it has not been done. Written after asking
whether MLX has equivalents to the CUDA speed-ups the training images rely on.

**Nothing here has been measured on Apple Silicon.** Every number below comes from
the training VM (20 GB RAM, RTX 3090, 4 cores) or from the CPU baselines already
recorded in `docker/Dockerfile.mww.cuda`. Treat the whole document as a plan, not a
result.

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

## MLX is not a drop-in, and that is the main finding

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
| Corpus generation (Kokoro, Piper) | **Docker, unchanged** | none |
| Training | **host, uv env** | this document |
| Eval | **Docker, unchanged** | none |

Corpus generation needs no port because neither TTS server wants a GPU here anyway:
the compose default is `ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1` and Piper is
started without `--use-cuda` by choice. Both already run on arm64. That removes the
largest and slowest stage from the problem entirely.

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

`train-host/` holds a resolved `pyproject.toml` and `uv.lock` for the host route.
It is **not used by anything** and is kept only because phase 2 needs it: uv resolved
the pinned openWakeWord stack (speechbrain 0.5.14, datasets 2.14.6, `numpy<2`) on
macOS/CPython 3.12 in 143 packages, which was the open question about whether a host
env was even possible. Delete it if MPS is abandoned.

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

### Phase 3 — microWakeWord, host, CPU

Same host-environment treatment, no Metal, per the tensorflow-metal finding. Expect
this to be the easy half.

### What would make this not worth finishing

Stop and stay on the CUDA box if phase 1 shows CPU training so slow that the
feedback loop — step 3 back into step 2, which is the actual pipeline — becomes
painful. The bar is not the 3090's ~29 minutes; it is whatever keeps a
change-one-thing iteration inside a sitting.

The standing warning from the TTS side applies to every phase here: `train.py`
records a Kokoro process pinned at 101.8% CPU — one core — while the GPU sat at 21%
and VRAM at 1.5 of 24 GB. The bottleneck was a serialised stage, not compute, and
the accelerator was mostly idle. *"It initialised"* and *"it helped"* are different
claims, and only the second one matters.
