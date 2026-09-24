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

## openWakeWord stage costs on the current pipeline (measured 2026-09-22, Mac host)

The P0 work split the oww run into reuse-checked stages (corpus manifest,
features sidecar keyed on corpus digest + rounds + seed), so the question a
sweep plan actually asks is no longer "how long is a run" but "what does one
point cost once the corpus exists". Measured end to end on the 15-voice
post-reservation corpus (c348af7b, 17,241 clips, all-Kokoro, seed 55, 25k
steps, `--corpus rebuild`; file mtimes, `logs/corpus-rebuild.log`):

| stage | wall |
|---|---|
| TTS corpus + trim + manifest (17,241 clips, one Kokoro engine on 8900) | **12m12s** |
| Features, 3 augmentation rounds (sidecar miss: new seed) | **8m49s** |
| 25,000 steps + .onnx write | **3m50s** |
| **Total, corpus rebuild included** | **24m51s** |

A warm sweep point on the same corpus and seed is the features-skip case:
~4 minutes. A warm point at a new seed recomputes features: ~13 minutes.
That is the sub-hour repeatable pass the plan was after: a two-point grid
with two repeats is under two hours including the one-time corpus build.
The ±30% machine-load rule applies (load not recorded for this run);
the 22-voice 50k rows above stand for the bigger-corpus case.

`--smoke` (harness check, not a quality pass) measured the same day:
own 13.5 min end to end, mww ~1 minute (improvement.md, P2.5) - the oww
smoke is dominated by the feature recompute, which smoke deliberately
re-runs so a broken feature stage fails in minutes, not hours.

## First sweep: 25k vs 50k steps - not a win at matched false accepts (measured 2026-09-22)

`sweeps/oww-training-steps.yaml`: 25k vs 50k steps, two repeats each, on the
frozen cf9c065b corpus, seeds 42/43 (25k) and 1042/1043 (50k). The matched-FA
curves were built by hand - 40 evals driving `eval_model.py` across decision
thresholds 0.25-0.85 - because the automated matched-FA path (eval's
`threshold_sweep` + the ledger's `det@FA<=B` column) did not exist yet; that
gap is now closed, so the next sweep concludes itself.

In the comparable FA band (~2.2-2.5% total-negative FA over the 366-clip set)
detection is the same - 25k ~78-82%, 50k ~77-81% - overlapping inside the
2-6 point repeat spread and far under the 10-point run-to-run noise floor:
**NOT distinguishable at matched false accepts.** 50k only reaches its higher
detection ceiling (up to 90%) at 3.0-3.3% FA - it buys that extra detection
with more false triggers, not for free. Per speaker (never pooled): ryan is
the lowest-detection voice at every point and step count (17-33% in the
comparable band); jay 77-94% and jen 80-100% are solid for both. Doubling
steps does not lift the weak voice. Consistent with the corpus-depth result
(more of the training signal did not produce a deployable model).

**First clean voice-holdout baseline** (post-reservation corpus c348af7b, 15
voices, 2026-09-22): `81490a6-dirty-c348af7b-hc250775` (seed 55, 25k steps) -
**28/45 (62%) at threshold 0.5, median latency 94 ms** on the 45-clip
voice-disjoint ranking set; on the real holdout, 37/51 (73%) at 0.5 with ryan
1/6. The @0.5 readings are orientation, not comparisons; the contaminated-era
4/45 measured against cf9c065b models is NOT comparable to the 28/45
(different models at a fixed threshold, which the fixed-threshold rule
forbids). Caveats that survive to the next sweep: the cf9c065b rows have a
contaminated voice-holdout axis (the corpus predates the reservation), FA
quantises to 0.27%/clip at 366 negatives, and per-speaker n is small (ryan 6
clips).

## Current models (trained 2026-09-23)

Two fresh points, both on post-reservation corpora, latest code (42f8982):

- **oww `42f8982-c348af7b-hb9d1d75`** (seed 56, 25k steps, corpus c348af7b):
  43/51 (84%) at 0.5 - jay 33/35, jen 9/10, ryan 1/6; adversarial FA 5/298
  (2%); det@FA≤2.0% 86.3%; median latency 100 ms; voice-holdout set 35/45
  (78%) at 0.5, 66 ms. The second point on the clean corpus (seed 55: 37/51,
  28/45) - the delta is inside the 10-point seed noise, and ryan is 1/6 on
  both: the weak-voice result is not a seed effect.
- **mww `42f8982-ce1500f5-h5f6f353`** (10k steps, calibrated cutoff 0.62:
  88.74% recall at 0.187 FA/h on the training set) on the first post-
  reservation mww corpus - **ce1500f5**, 7,617 clips, 30% Kokoro mix, the
  held-out voices and 2 Piper pairs excluded and recorded in its manifest.
  Holdout at 0.62: 38/51 (75%) - jay 34/35, jen 4/10, **ryan 0/6**; adversarial
  FA 6/298 (2%); median latency 142 ms (the old candidate measured 261 ms);
  voice-holdout set 33/45 (73%) - its first clean reading (the old candidate
  predates the reservation). Against the staged candidate ecbf160 (73% at its
  0.09 cutoff: jay 31/35, jen 2/10, ryan 4/6) it wins jay and jen and loses
  ryan outright - not a per-speaker win, so per the deploy rule it does not
  replace the staged candidate.

Both engines miss the same two speakers - mww reads jen 4/10 and ryan 0/6, oww
reads ryan 1/6 with jen at 9/10 - and the fix for both is the same lever no
hyperparameter has moved: more real recordings of them.

## Real-clip VTLP: the dose-response is real, the trade is not worth it (swept 2026-09-23)

The first `corpus_axes` sweep (`sweeps/oww-real-vtlp.yaml`): VTLP variants
(1.15-1.30x formant shift) of ryan's 78 REAL clips at {0, 30, 60} variants
per clip, base 10x copies, 2 seeds per arm, each arm on its own TTS redraw
(corpus ids dc089d9 / 34e7640 / a90706a). Matched FA at the common budget
(adv FA <= 2.3%, extend+hey_other, each arm read on its own curve):

| arm | ryan (n=12) | jay | jen | det@FA<=2.3% | voice-holdout |
|---|---|---|---|---|---|
| none | 3/12 (25%) | 88.6% | 85.0% | **83.3%** | 80% |
| ryan=30 | 4/12 (33%) | **61.4%** | 80.0% | 67.6% | 73% |
| ryan=60 | **6/12 (50%)** | **62.9%** | 85.0% | 68.6% | 77% |

Ryan's gain is monotone in the dose and survives both repeats - this is the
first measured lever that moved him at all. But it is bought from the
adults: jay drops ~26 points in BOTH VTLP arms (43/70, 44/70 - not seed
noise) and overall detection at matched FA loses ~15 points against
baseline. The mechanism is visible in the row counts: 60 variants/clip makes
ryan ~36% of positive-class rows vs jay's 1600, and the positive class
re-balances toward him by dilution of attention, not by acquisition. Verdict:
**as configured, real-clip VTLP is a measured net negative for the clean-
detection gate** - it moves the right speaker and breaks the wrong ones.
A single point before the sweep (seed 58: ryan 3/6, jay 30/35 held) read as
the lever working; under 2x2 discipline on a different TTS redraw it did not
reproduce - which is precisely why the loop got its grid.

The retry for seed 2042 (arm ryan=60) died once in the known onnxruntime
recursive_mutex SIGABRT flake during feature computation (the same abort a
2026-09-22 re-run hit) and filed cleanly on re-run; arms now sit
2/2/2 repeats in the ledger.

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
`eval/src/compare_models.py`). These are records of past runs against the
adversarial set as it then was; the denominator changed on 2026-09-22 (32 ->
298 clips, P1.1), so every `x/32` figure below is not comparable to a modern
scorecard's `x/298` - the comparisons WITHIN this section are unaffected, the
comparison ACROSS the date is not:

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

## Piper fleet (N instances): no N>1 scaling on one machine (2026-09-22, re-measured the same day with repeats)

P2.1 in `improvement.md` ships `PiperFleet` (voice-pinned sharding across N
`piper_engine` processes, `scripts/start-tts-fleet.sh`) to scale the corpus
stage. The first measurement — one clean run per N, machine load not recorded
— read:

```
N = 1      2      4       6       8
   16.66  9.09  17.11   17.26   18.16   clips/s     (superseded: one trial each)
```

Non-monotonic (a 45% loss at N=2 that "recovers" at N=4), and this file's
±30% / same-day rules do not license a conclusion from a single reading, so it
was re-measured the same evening: `tools/bench_piper_fleet.py --trials 3`, the
same 1,440-clip oww workload at every N (25 models, 90 (voice, speaker) pairs
x 16 clips, built once and re-used, so a change between sizes is the fleet not
the workload; each model's one 0.6 s load is paid, not warmed away), three
trials per N back to back, the box's 1/5/15-min load average recorded at each
trial's start and end. Min/max per cell, the way the MLX table does:

| N | clips/s, min-max (n trials) | mean | 1-min load (trial starts) |
|---|---|---|---|
| 1 | 15.49 - 16.51 (6) | 16.14 | 4.5 - 9.4 |
| 2 | **8.59 - 8.96 (4)** | **8.71 (0.54x)** | 3.7 - 7.6 |
| 4 | 12.45 - 15.49 (3) | 13.46 (0.83x) | 4.1 - 11.4 |
| 6 | 13.73 - 16.28 (3) | 14.62 (0.91x) | 8.3 - 17.1 |
| 8 | 14.83 - 17.98 (3) | 15.95 (0.99x) | 12.9 - 20.1 |

Both findings are now supported by repeats, and both cut against the old
note's narrative:

**The N=2 dip is real, not the artefact it read like.** 8.59 / 8.64 / 8.65 on
three consecutive trials under the sweep's lightest loads, 0.54x, and the
fleet was not imbalanced (720/720 clips per instance, max-min spread 0 -
sharding was doing its job). A dip of exactly this shape is what survives
repetition.

**No N>1 scaling.** The old table called N=4/6/8 a tie with N=1; with repeats
N=4 sits at 0.75-1.00x (two of its three trials at 0.78x) and no N>1 *mean*
beats N=1's (16.14). The single trial that does clear N=1's best (17.98 vs
16.51, N=8) ran at the afternoon's dirtiest 1-min load (13-20); N=1 ran first,
at the lightest load of the sweep, so the one reading that flatters scaling is
also the one reading with the least room to be read as scaling. The ceiling
is about 18 clips/s for the box and one serial engine already delivers
15.5-16.5 of it.

**The mechanism - and where the old note was wrong.** `--sample-cpu` measures
each instance's CPU two ways over the trial: the exact cumulative-CPU-time
mean, and a 10 Hz peak poll (1 Hz aliases the ~60 ms request period):

| | exact mean CPU | 10 Hz peak | threads |
|---|---|---|---|
| N=1, one instance (3 sampled trials) | **462% of the box's 10 cores, identical to 1% across all three** | 520 - 539% | 11 |
| N=2, two instances combined | 370% | 647% | 9 + 8 |

The old note's mechanism — "one instance already used ~980% CPU, ten of this
box's ten cores" — was borrowed from `bench_tts.py`'s shorter "hey seeree"
workload and never measured on this one. It does not hold here: a single
instance averages 4.6 of the ten cores and peaks at ~5.3, and yet one
instance alone reaches the total throughput no fleet of four to eight can
pass. The wall is not a core count. And the N=2 dip is not core ownership
either — the two instances there burned only ~370% of the box on average
while every clip took twice as long as at N=1: each instance's onnxruntime
intra-op burst, sized for a 10-core box, time-shares cores with the other
process, so both slow down without either filling a core. That is
oversubscription, not saturation — the first pass's "~18-19 threads each
thrash" was a `top` guess, and the CPU times say the effect is less sharp
than that, but the direction holds. What the two processes contend on exactly
(memory bandwidth, cache, scheduler) is not measured here, and this file does
not record a mechanism it has not measured; the dip stands as a measured,
reproducible loss, its cause open.

Verdict: on this 10-core box, one Piper instance delivers what no fleet
delivers; N=2 is measurably worse, so a local fleet is out of the default
path. It is kept for the multi-machine case only — the `PiperFleet` docstring
states the disposition, and this section is where the measurement lives, the
way the CoreML section closes the CoreML question.

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
