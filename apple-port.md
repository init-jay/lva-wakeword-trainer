# Running the trainers on Apple Silicon

A note on what it would take, and why it has not been done. Written after asking
whether MLX has equivalents to the CUDA speed-ups the training images rely on.

**Nothing here has been measured on Apple Silicon.** Every number below comes from
the training VM (20 GB RAM, RTX 3090, 4 cores) or from the CPU baselines already
recorded in `docker/Dockerfile.mww.cuda`. Treat the whole document as a plan, not a
result.

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

## Open questions, in the order they should be answered

1. **Does tensorflow-metal support TensorFlow 2.21.0?** The mww image pins that
   version to match its base image. If the pairing does not exist, the mww half has
   no GPU story on macOS at all — though per above, it may not need one.
2. **How much RAM does the target Mac have?** Below ~32 GB the openWakeWord path is
   worse on a Mac than on the VM, and no amount of framework work changes that.
3. **Is CPU-only fast enough?** Cheapest experiment by far, and it needs no port:
   run the mww trainer on CPU natively and compare against the 15-minute baseline.
   If that is tolerable, most of this document is moot.
4. Only then: MPS for the oww model, and the CoreML provider for feature
   computation, measured separately rather than adopted together.

## Recommendation

Do not port. Train on the Linux box, and keep the Mac for the two steps that already
run there natively and need a microphone — `record/` (step 1) and `preflight/`
(step 4). Evaluation already runs on the Mac too, in the `eval` image, which builds
native for arm64 by design.

Revisit if the training VM becomes unavailable, or if a Mac with 48 GB+ of unified
memory makes question 2 answer itself.
