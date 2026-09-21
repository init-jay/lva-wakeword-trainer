# Measured timings

The route recommendations in the README rest on the measurements in this file,
and this file is where they live: the route timings, and the model results those
routes produced. Two reading rules apply to everything below, because both
already produced a wrong conclusion here: treat any single Mac timing as ±30%
machine load (the full story is in the Kokoro section), and never compare two
configurations measured on different days - cross-day totals are directional,
same-day stage probes are the evidence.

## The full table

Three environments, all measured with `time` on the same wake word:

- **CUDA box** — RTX 3090, 20 GB RAM, 4 cores, `docker-compose.cuda.yml`
- **Mac, Docker** — M1 Max, 64 GB, 10 cores, `docker-compose.cpu.yml`, no GPU
- **Mac, host** — same Mac, no container: `scripts/*-applesilicon.sh`

| Step | CUDA box | Mac, Docker | Mac, host |
|---|---|---|---|
| Fetch external corpora | download-bound, ~43 GB for both targets | same | same |
| Record | human time, 20–50 clips per speaker | same | same |
| Train — microWakeWord | **28m02s** | **26m06s** | **14m14s** |
| ├ TTS corpus (Piper + Kokoro) | included above | included above | **6m12s** (all-Piper) |
| ├ Features | included above | included above | 1m22s |
| └ Training, 10,000 steps + TFLite | included above | included above | **6m40s** |
| Train — openWakeWord | **28m59s** | ~2h 15m | **~34–48m** |
| ├ Kokoro corpus, protocol server (`tcp://`; `mlx://` before the refactor) | included above | ~49 min | **~18–25m** |
| ├ Augmentation + features | included above | ~21 min | **~11–12m30s** |
| └ Training, 50k steps | ~16 min | **~37 min** | **~4m35s–6m** |
| Eval | minutes | minutes | minutes |
| Preflight | needs a mic | needs a mic | needs a mic |

The Mac-host openWakeWord row is three measured 22-voice runs: the two
in-process `mlx://` runs (37m and 47m43s) and, after the protocol refactor,
the same engine behind a TTS protocol server — 34m37s on 2026-09-09
(`training-hey_seeree-macos-20260909-073922.log`): corpus + trim 17m45s, then
augmentation + features + 50k steps in the remaining 16m48s, the training
loop itself ~6m from feature completion (08:08) to model write (08:13:55);
its run-on clips were cut at the engine's word timestamps with zero
estimate fallbacks, and the model is not yet evaluated. Not a sum of stages
- the ±30% machine-load caveat in the Kokoro section below applies. An
earlier 36-voice host-server run took ~1h15m; most of that difference is
corpus size, not speed.

Figures for whole scripts are full runs at defaults - `run-mww-training.sh` covers
corpus, features, training and manifest; `run-oww-training.sh` covers TTS generation,
augmentation, training and the tflite conversion. The two targets are independent.

Four things move these numbers more than the hardware does. `SKIP_CORPUS=1` skips
corpus generation on a re-run, which is the largest single stage. `KOKORO_EXTERNAL=1`
with `scripts/start-kokoro-host.sh` takes TTS out of the container, worth 3.7x on
that stage. Docker Desktop's memory limit decides whether the 17.28 GB array is
mmap'd or thrashed, and falling short page-faults rather than erroring. And on the
CUDA box, holding the card alone is the difference between finishing and not: a
Kokoro server left up cost a run a 16.09 GiB allocation with 15.34 GiB free.

## openWakeWord on the Mac: host vs. container

**On Apple Silicon, leaving the container is worth ~7x on the training stage.** 250
it/s on the host against 26 in the container — same corpus, same commit, same
torch 2.5.1, only the environment differs. The macOS wheels link Accelerate and the
linux/arm64 ones do not; the container also mmaps a 16 GB feature array across
Docker Desktop's filesystem boundary, which the host reads natively. Those two have
not been separated, so treat "7x" as the combined effect rather than a claim about
either one.

The component measurements behind it (same torch 2.5.1, identical script, native
arm64 in both environments):

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
improvement: it wins the GEMM half of a model and loses the convolution half, and
anyone who asserts a blanket result without measuring will be right about one half
and badly wrong about the other.

Which half you land on is decided by architecture: openWakeWord is `Linear` x7 and
one `LSTM` — no convolutions at all (counted in its own `train.py`) — so it is pure
GEMM, the missing oneDNN costs it nothing, and Accelerate is a straight win. That is
why `docker/Dockerfile.oww.cpu` is running the slower of the two torches for the one
model it trains. (microWakeWord is mixednet — convolutional — but it trains with
TensorFlow, so this torch probe does not transfer to it; the TF probe in the section
below is the one that does.)

**And it was not a trade against quality.** Evaluated on the same holdout, the
host-trained model beat the container-trained one at every matched false-accept
point and for every speaker - jay_runon, the largest sample at n=57, went 35% ->
51%. Two runs is not proof, and this repo has measured 10 points of run-to-run
variance before, but the direction was consistent everywhere.

## microWakeWord on the Mac: host vs. container

**microWakeWord is also faster on the host on a Mac, and the host is the default
route there.** The old reason to leave it in Docker was measured in the wrong
library: macOS torch ships without oneDNN and measured 12x slower on conv1d, and
mixednet is the convolutional model — a fair prior, but that probe was torch, and
mWW trains with TF 2.21.0. Measured in TF instead (`tools/tf_probe.py`): batch 128, the actual op shapes in `train/mww/train.py`
(first conv (5,1) s3, the three MDConv blocks, a 1024³ GEMM, and a full train step
forward+grad+update), the same `tensorflow==2.21.0` source build in both
environments:

| operation | container (linux/arm64) | host (macOS arm64) | |
|---|---|---|---|
| first conv (5,1) s3, 150 frames | 3.34 ms | 2.92 ms | host 1.14x |
| MDConv block 1 (32ch) | 3.50 ms | 2.72 ms | host 1.29x |
| MDConv block 4 (1024ch) | 8.37 ms | 7.26 ms | host 1.15x |
| GEMM 1024³ | 20.67 ms | 16.09 ms | host **1.29x** |
| full train step (fwd+grad+update) | **36.21 ms** | **30.86 ms** | host **1.17x** |

`threading_options` made **both** sides slower (host 33.90, container 39.80 with
10 threads): both builds run one CPU per op at these shapes, so the win is the
BLAS in the GEMM-bound residual, the same Accelerate story as the table in the
section above — smaller, because mixednet's convolutions are too skinny to feed a
GEMM kernel, which is why the torch conv prior looked the way it did.

The other half of a run is TTS, where most of a run goes, and the host servers are
faster for both engines in the corpus. `tools/bench_tts.py`, the same
`wyoming-piper` 2.4.3 / `piper-tts` 1.7.0 in both environments, 140 "hey seeree"
clips:

| | container | host | |
|---|---|---|---|
| Piper, sequential | 9.13 clips/s | **21.66 clips/s** | host **2.4x** |
| Piper, 8 client threads | 9.13 (flat) | 23.4 | still serialises |

Same server code, same model, 2.4x: `piper-tts` runs on onnxruntime, and the macOS
wheel links Accelerate while the linux/arm64 one does not — the identical mechanism
to every other number in this document. Kokoro is the same Kokoro-FastAPI v0.8.1
the openWakeWord host route already uses; the Kokoro section below documents its
host-vs-container wall time.

The full run on this machine: the host's, 2026-09-07, against the container's
2026-09-06 run — a day apart, so per the day rule, read the total as directional
and the same-day stage probes as the evidence. Container stage times are that
run's log timestamps:

| stage | container | host | |
|---|---|---|---|
| corpus (4,920 + 984 clips, all-Piper) | ~14 min | **6m12s** | 2.3x |
| features | ~1 min | 1m22s | about the same |
| train, 10,000 steps + conversion + ROC | ~11 min | **6m40s** | 1.7x |
| **total** | **26m06s** | **14m14s** | **1.83x** |

The corpus row is the all-Piper mix (`KOKORO_FRACTION=0`). The run script now
defaults to a 30% Kokoro mix — the mirror of the oWW corpus's 30% Piper fraction —
because the phrase-alone budget was single-voice-family and the oWW notebook's
two-engines-beat-one result (run 17) applied untested here. First mixed run on
this machine, same day (2026-09-07, tag `e4da6a3-dirty`): corpus **7m24s** against
the all-Piper 6m12s — the mix costs about a minute of Kokoro render, and the full
run stayed at 14m24s against 14m14s.

**What the runs bought** (matched-false-accept comparisons on the holdout,
`eval/src/compare_models.py`):

- **30% mix at 1x depth: diversity bought run-on, not yet generalisation.** Against
  the same-day all-Piper run (`333944a`): run-on 65% vs 34% at 4/32 adversarial
  false accepts (ceilings 93 vs 75), plain detection a wash (57 vs 55), and the
  per-speaker spread survived (jen 20% vs jay 69%, n=10/35) — generalisation still
  needs more jen/ryan recordings, which is a human task.
- **2x depth, 30% mix: collapse.** Per-voice phrase-alone budget 60 -> 120,
  training 10,000 -> 20,000 steps, 15,132 positives (tag `ecbf160-dirty-d4504462`):
  corpus **12m20s** — the near-linear scaling the stage table implies — full run
  **27m01s**. At 12/32 adversarial false accepts, plain 86 -> 59 and run-on 93 ->
  28, with 17/32 adversarial false accepts at the 0.5 reference: the worst of the
  four models compared.
- **2x depth, all-Piper: a firehose.** The same doubled budget at
  `KOKORO_FRACTION=0` (15,156 positives, `ecbf160-dirty-dcdbcc5c`, **24m35s**):
  held-out detection came **back** — the best of any run (84% plain and 87%
  run-on at the 0.5 reference; ceilings 86/90 at 12/32) — but the model no longer
  rejects its own training set (37.5% of the training negatives score above 0.99;
  training-ROC AUC 0.295, below a coin flip), so no cutoff meets the 0.2 FAPH
  deployment budget and the manifest stage refused to write. Read the two runs
  together: at doubled depth the 30% mix was what collapsed detection (all-Piper
  2x far ahead of mixed 2x), while Piper-only depth bought no deployable gain over
  1x and cost the operating point. The Kokoro share has a sweet spot at 1x;
  doubling depth in either configuration did not.
- **2x adversarial negatives: the lever that worked.** 1x depth, all-Piper,
  negatives 12 -> 24 per voice (1,968 total; `ecbf160-dirty-da01854d`, **15m54s**).
  It confirmed the diagnosis — the firehose was not overfitting, it was an
  underfed rejection set. At its calibrated 0.09 cutoff this is the first model in
  the repo to pass the extend+hey_other gate: 1/32 against 5-17/32 for every
  earlier run, with zero training false accepts at 0.81 and the best per-speaker
  plain numbers measured (jay 94 / jen 40 / ryan 83 at 4/32 matched). The price was
  recall, not rejection: run-on 37 (against 65-93 for the 1x runs),
  detection-with-command 57%, median latency 261 ms — the conservative model fires
  late — and the per-speaker wall (jen 20-40%) survived it, as it has every lever
  so far. Recorded as point 5 of `train/mww/corpus.py`.

The levers that remain are the ones depth cannot touch: more real recordings (the
per-speaker spread is the standing failure in every configuration, including jen at
0% on the firehose), and the run-on positives that take the word timestamps Kokoro
already provides (`train/mww/corpus.py` records both measurements). The training
stage beat its 1.17x probe, because the conversion and ROC calibration that follow
it run in the same process and carried the same gap. One caveat that would change
the training-stage conclusion if it ever bites: the container side was measured in
this Docker Desktop VM, and its CPU share is a VM setting, not a constant of the
hardware. The TTS gap does not move with that — it is a library link, not a
schedule. The Docker route stays correct for every other machine; on a Mac the host
route is the faster one, measured, not projected.

Do not read the mww result as "the GPU is pointless" either. It is a claim about
one 25,537-parameter model, too small to fill a 3090, in a run that is mostly not
training: `docker/Dockerfile.mww.cuda` records the CPU baseline at ~46 s per 500
steps — about 15 minutes for a 10,000-step run — and notes that a model this small
may not fill a GPU at all; and Piper corpus generation is CPU-only on both machines
by choice, since `--use-cuda` measured 2.5x *slower*.

The Metal question is closed, not pending, and this section is where it is
recorded. Docker Desktop passes no Metal device through to containers, so a
torch build that asks for `mps` inside one finds nothing and falls back to CPU -
silently, which is worse than failing - and no compose overlay can say otherwise
(there is no device reservation to write, unlike `docker-compose.cuda.yml`'s
`driver: nvidia`), so there is no `.mps` overlay at all. The host is the only
route to Metal; the MPS/CoreML training route there was evaluated and abandoned,
and tensorflow-metal does not pair with TF 2.21.0, so microWakeWord has no Metal
path at all. Kokoro is the one measured exception (below).

## CoreML for the oww feature stage: measured, loses (2026-09-22)

The question the Metal section above does NOT close: the feature stage runs the
melspectrogram + embedding ONNX models through onnxruntime on CPU (the largest
non-TTS host stage, ~12 min at 22,144 clips in the 2026-09-21 bar-test run),
and onnxruntime on this Mac ships a CoreMLExecutionProvider. `tools/coreml_probe.py`
ran the real models at the real batch shape (16 clips x 19,200 samples,
ncpu 5 as in the stage), CPU's threaded per-clip/per-window path (what runs
today) vs CoreML batched:

| stage | CPU (today) | CoreML | |
|---|---|---|---|
| melspectrogram, batch 16 | 2.9 ms | 10.5 ms | 0.28x |
| embedding, batch 16 | 27.1 ms | 67.4 ms | 0.40x |
| per-batch total | 30 ms | 78 ms | **0.39x** |

CoreML is 2.6-3.5x SLOWER on both stages (extrapolated: ~31 min of model time
where CPU measures ~12), and the outputs are not bit-identical - embedding
drift max 6.1e-02 on values ~13.5, which is above float noise for features
that are baked into every trained model. The graphs only partially convert
(11 of 18 melspec nodes, 44 of 65 embedding nodes), so the rest falls back to
CPU inside the same call. Verdict: do not add a CoreML branch to the feature
stage; the question is closed by measurement, the way the Metal section is.
(The probe is reproducible: `train-applesilicon/.venv/bin/python tools/coreml_probe.py`.)

## Kokoro TTS: three ways to run it, and the speed/quality tradeoff

Corpus generation is the largest stage, so the TTS engine matters more than the
trainer. Three configurations, all on the M1 Max, all generating the SAME corpus
(22 voices, 13,183 positives, 6,600 negatives) so the comparison is engine-only:

| | corpus TTS | whole run | run-on recall @ 6/32 FA |
|---|---|---|---|
| Kokoro-FastAPI in Docker | ~2h (est.) | not run | not run |
| **Kokoro-FastAPI on the host** | **30m29s** | 47m56s | **82%** |
| kokoro-mlx, in-process (`mlx://` at the time; `tcp://` since the refactor) | **19m06s** / 24m54s | 37m / 47m43s | 51% -> 61% |

The 09-09 protocol-server run (34m37s, the full-table note above) is not a row
here: it generated a different, larger corpus (15,317 positives / 7,260
negatives), so it would break the same-corpus premise of this table. It does
settle one thing for it - the `tcp://` transport costs nothing measurable
against in-process on the same engine and machine.

The in-container row is an estimate; the measured form of the same comparison came
sideways during the Apple Silicon port: chasing Kokoro throughput turned up the same
Kokoro-FastAPI version running **3.7x** faster in a host uv venv than in a
container on this hardware — native arm64 both times, no emulation. That is a torch
gap, the same Accelerate story as every table above, and it is what made
`scripts/start-kokoro-host.sh` worth building. The standing warning attached to
it: `train/oww/train.py` records the server pinned at 101.8% CPU — exactly one
core — while the GPU sat at 21% and VRAM at 1.5 of 24 GB. The bottleneck was a
serialised stage, not compute, so the accelerator was mostly idle, and "it
initialised" and "it helped" are different claims.

**MLX generates the corpus 1.2-1.6x faster and produces a worse corpus on
run-on; the timing table above assumes the MLX engine (in-process `mlx://`
at the time of those measurements, the `tcp://` protocol server since the
refactor), and the host FastAPI server is the alternative when the run-on
gap matters.** The regression is specific to
run-ons; plain positives are comparable.

TWO NUMBERS PER MLX CELL, BECAUSE THE SPREAD IS THE POINT. Identical corpus,
identical engine, two runs a day apart: TTS 19m06s then 24m54s, training 256 then
140 it/s. Nothing changed but what else the machine was doing. Treat any single
timing here as +/- 30%, and never compare two configurations measured on different
days - an earlier "MLX is only 9% faster" reading came from a run while the machine
was paging 540,000 times against 296 MB of free swap, and was simply wrong.

The `51% -> 61%` is a timestamp fix landing between the two MLX runs. Run-on clips
are the wake word plus a short tail of the following command, cut at a word boundary
the TTS engine reports, and kokoro-mlx was reporting that boundary ~34 ms early -
enough to leave almost no tail:

    run-on minus plain, which should be the tail:
      FastAPI   697 - 580 = 117 ms
      MLX       645 - 610 =  35 ms      RUNON_TAIL_MS is 150-300

A run-on with no tail is just a plain positive, so the model never learns to fire when
speech continues past the wake word.

Fixing it recovered run-on recall by +10 to +14 points at every operating point from
6/32 false accepts upward, which confirms the cut was the mechanism. It did not close
the gap: 84/61 against the host server's 90/82. Something in the audio itself - bf16
weights, misaki phonemisation - still costs run-on detection, and that is unexplained.
See `BUGREPORT-kokoro-mlx.md` and `tts-service/engines/kokoro_mlx/`.

**What this settled:** voice count is not the problem. The same FastAPI corpus at 22
voices scored BETTER than an earlier one at 36 (82/65 against 76/56 at 4/32), so the
13 voices MLX lacks cost nothing - they are `v0` legacy variants of speakers it
already has, not distinct ones. Those are now skipped by default for every engine,
which takes ~36% off the Kokoro clips: see `LEGACY_VOICE_MARKER` in
`train/corpus/negatives.py`.

MLX remains worth having for plain clips and negatives, which are 79% of the corpus
and show no regression. That is the split to build if the remaining run-on gap turns
out not to be closable.
