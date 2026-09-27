# Pipeline efficiency: what the stages cost, and what moves them

The route recommendations in `README.md` and `CLAUDE.md` rest on the measurements
here. This file is the *efficiency* half of what was measured: which route to take,
what each stage costs, and which avenues are closed by measurement rather than by
opinion. It is not the record of any one wake word's model quality — detection and
false-accept results, per-speaker scores, sweep verdicts and the staged candidate
live on branch `train/hey_seeree` in `SPEED.md` and `deploy/`. Those numbers are a
property of a campaign; the ones below are properties of the pipeline and a machine.

**Conditions, not constants.** Everything measured here was measured on two
machines: an M1 Max (64 GB, 10 cores, no GPU, Docker Desktop) and an RTX 3090 box
(20 GB VRAM, 20 GB RAM, 4 cores). The corpora were the `hey seeree` ones in use at
the time — 22,144 clips at 22 voices for the oww route tables, 17,241 clips at 15
voices (digest `c348af7b`) for the stage-cost table, 4,920 + 984 clips for the mww
route table. A different word, voice count or machine moves every figure. What
transfers is the *shape*: which stage dominates, which lever is largest, and which
directions are dead ends.

**Two reading rules, because both produced a wrong conclusion here.** Treat any
single Mac timing as ±30% machine load (the spread is measured, below). And never
compare two configurations measured on different days: cross-day totals are
directional, same-day stage probes are the evidence.

## The full table

Three environments, all measured with `time` on the same wake word:

- **CUDA box** — RTX 3090, 20 GB RAM, 4 cores, `docker-compose.cuda.yml`
- **Mac, Docker** — M1 Max, 64 GB, 10 cores, `docker-compose.cpu.yml`, no GPU
- **Mac, host** — same Mac, no container: `src/scripts/*-applesilicon.sh`

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

The Mac-host openWakeWord row is three measured 22-voice runs: two in-process
`mlx://` runs (37m and 47m43s) and, after the protocol refactor, the same engine
behind a TTS protocol server — 34m37s on 2026-09-09, corpus + trim 17m45s then
augmentation + features + 50k steps in the remaining 16m48s, the training loop
itself ~6m from feature completion to model write. Not a sum of stages: the ±30%
rule applies. An earlier 36-voice host-server run took ~1h15m, and most of that
difference is corpus size, not speed.

Whole-script figures are full runs at defaults — `run-mww-training.sh` covers
corpus, features, training and manifest; `run-oww-training.sh` covers TTS
generation, augmentation, training and the tflite conversion. The two targets are
independent, so a `make` of one is not a claim about the other's cost.

**Four things move these numbers more than the hardware does.** `SKIP_CORPUS=1`
skips corpus generation on a re-run, and the corpus is the largest single stage.
`KOKORO_EXTERNAL=1` with `src/scripts/start-kokoro-host.sh` takes TTS out of the
container, worth 3.7x on that stage. Docker Desktop's memory limit decides whether
the ~17 GB feature array is mmap'd or thrashed, and falling short page-faults
rather than erroring. And on the CUDA box, holding the card alone is the difference
between finishing and not: a Kokoro server left up cost a run a 16.09 GiB
allocation with 15.34 GiB free.

## What one sweep point costs, once the corpus exists

The oww run is split into reuse-checked stages (a corpus manifest, and a features
sidecar keyed on corpus digest + augmentation rounds + seed), so the question a
sweep plan asks is not "how long is a run" but "what does one point cost".
Measured end to end on the 15-voice, 17,241-clip corpus (`c348af7b`, all-Kokoro,
seed 55, 25k steps, `--corpus rebuild`, file mtimes):

| stage | wall |
|---|---|
| TTS corpus + trim + manifest | **12m12s** |
| Features, 3 augmentation rounds (sidecar miss: new seed) | **8m49s** |
| 25,000 steps + .onnx write | **3m50s** |
| **Total, corpus rebuild included** | **24m51s** |

A warm point on the same corpus and seed is the features-skip case: ~4 minutes. A
point at a new seed recomputes features: ~13 minutes. That is what makes a small
grid affordable — two points with two repeats came under two hours including the
one-time corpus build. `--smoke` (a harness check, not a quality pass) measured
13.5 min for oww and ~1 min for mww on the same day: the oww smoke is dominated by
the feature recompute, which smoke deliberately re-runs so a broken feature stage
fails in minutes rather than hours.

The corollary for anyone planning sweeps: **depth is not a free lever, and it is
not a quality lever either.** Doubling corpus depth costs the near-linear amount
the stage table implies (~14 min of corpus at 2x on the mww route) and produced no
deployable model in either engine mix; the runs are on branch `train/hey_seeree`.

## Route choice on Apple Silicon: measure the op, not the platform

**Leaving the container is worth ~7x on the oww training stage** — 250 it/s on the
host against 26 in the container, same corpus, same commit, same torch 2.5.1. Two
causes are tangled in that figure and have not been separated: the macOS wheels
link Accelerate and the linux/arm64 ones do not, and the container mmaps a 16 GB
feature array across Docker Desktop's filesystem boundary. The components:

| operation (torch 2.5.1, identical script) | container linux/arm64 | host macOS arm64 | |
|---|---|---|---|
| `matmul 2048³` | 53.05 ms | **17.42 ms** | host **3.0x** |
| train step, 1024 batch, MLP | 26.39 ms | **4.19 ms** | host **6.3x** |
| `conv1d 512×16×96` | **7.38 ms** | 90.16 ms | host **12x SLOWER** |

The conv row is not noise — three trials, 88.15 / 90.44 / 88.39 ms — and the cause
is one flag: the macOS wheel ships without oneDNN, so convolutions fall back to a
reference kernel, while GEMM on the host goes to Accelerate. Two libraries, two
ops, opposite winners. **"Go native on Apple Silicon" is not a blanket
improvement**: it wins the GEMM half of a model and loses the convolution half, and
anyone who asserts the blanket result without measuring will be right about one
half and badly wrong about the other.

Which half you land on is decided by architecture. openWakeWord is `Linear` ×7 and
one `LSTM` — no convolutions at all (counted in its own `train.py`) — so it is pure
GEMM, the missing oneDNN costs it nothing, and Accelerate is a straight win. That is
why `docker/Dockerfile.oww.cpu` runs the slower of the two torches for the one model
it trains. microWakeWord is mixednet — convolutional — but it trains with TensorFlow,
so a torch probe does not transfer; `src/scripts/tf_probe.py` measures the same
question in TF 2.21.0 at the real op shapes, and the host wins everything by 1.14x
to 1.29x (full train step 36.21 ms container vs 30.86 ms host, host 1.17x).
`threading_options` made **both** sides slower: both builds run one CPU per op at
these shapes, so the win is the BLAS in the GEMM-bound residual — the same
Accelerate story, smaller, because mixednet's convolutions are too skinny to feed a
GEMM kernel, which is why the torch conv prior looked the way it did.

The rest of an mww run is TTS, where most of it goes, and the host servers are
faster for both engines: `src/scripts/bench_tts.py`, same `wyoming-piper` 2.4.3 /
`piper-tts` 1.7.0 both sides, 140 clips — Piper 9.13 clips/s in the container
against 21.66 on the host (**2.4x**, same server code and model, because
`piper-tts` runs on onnxruntime and only the macOS wheel links Accelerate), and
flat at 9.13 with 8 client threads, so it serialises either way.

Full mww run on this machine, host (2026-09-07) against container (2026-09-06) — a
day apart, so per the day rule read the total as directional and the same-day stage
probes as the evidence: corpus ~14 min → **6m12s**, features ~1 min → 1m22s, train
+ conversion + ROC ~11 min → **6m40s**, total **26m06s → 14m14s (1.83x)**. One
caveat that would change the training-stage conclusion if it bites: the container
side was measured in this Docker Desktop VM, and its CPU share is a VM setting, not
a constant of the hardware. The TTS gap does not move with that — it is a library
link, not a schedule. **The Docker route stays correct for every other machine**; on
a Mac the host route is the faster one, measured, not projected.

Nor is the host result a claim that the GPU is pointless. It is about one
25,537-parameter model, too small to fill a 3090, in a run that is mostly not
training: `docker/Dockerfile.mww.cuda` records the CPU baseline at ~46 s per 500
steps (~15 min for 10,000) and notes a model this small may not fill a GPU at all;
and Piper corpus generation is CPU-only on both machines by choice, since
`--use-cuda` measured 2.5x *slower*.

## Closed avenues

These are dead ends settled by measurement, kept so nobody spends an afternoon
re-deriving them.

**CoreML for the oww feature stage: loses, 0.39x.** The feature stage runs the
melspectrogram + embedding ONNX models through onnxruntime on CPU (the largest
non-TTS host stage, ~12 min at 22,144 clips), and onnxruntime on this Mac does
expose a `CoreMLExecutionProvider`. `src/scripts/coreml_probe.py` ran the real
models at the real batch shape (16 × 19,200 samples, ncpu 5, as in the stage),
CPU's threaded per-clip path against batched CoreML: melspectrogram 2.9 ms vs
10.5 ms, embedding 27.1 ms vs 67.4 ms. CoreML is 2.6–3.5x slower on both stages,
**and** the outputs are not bit-identical — embedding drift max 6.1e-02 on values
~13.5, which is above float noise for features baked into every trained model.
The graphs only partially convert (11 of 18 melspec nodes, 44 of 65 embedding
nodes), so the rest falls back to CPU inside the same call. Do not add a CoreML
branch to the feature stage.

**A local Piper fleet: no N>1 scaling on one machine.** `PiperFleet`
(`src/scripts/start-tts-fleet.sh`) shards voice-pinned work across N
`piper_engine` processes. Measured with repeats — `src/scripts/bench_piper_fleet.py
--trials 3`, the same 1,440-clip oww workload at every N, built once and re-used so
a change between sizes is the fleet and not the workload, load average recorded at
each trial:

| N | clips/s, min–max (trials) | mean | 1-min load at trial starts |
|---|---|---|---|
| 1 | 15.49 – 16.51 (6) | 16.14 | 4.5 – 9.4 |
| 2 | **8.59 – 8.96 (4)** | **8.71 (0.54x)** | 3.7 – 7.6 |
| 4 | 12.45 – 15.49 (3) | 13.46 (0.83x) | 4.1 – 11.4 |
| 6 | 13.73 – 16.28 (3) | 14.62 (0.91x) | 8.3 – 17.1 |
| 8 | 14.83 – 17.98 (3) | 15.95 (0.99x) | 12.9 – 20.1 |

The N=2 dip is real (8.59 / 8.64 / 8.65 on three consecutive trials, sharding
balanced 720/720, spread 0) and no N>1 mean beats N=1. The one trial that clears
N=1's best (17.98 at N=8) ran at the sweep's dirtiest load. Mechanism, measured
with `--sample-cpu`: one instance averages **462% of the box's 10 cores** —
identical to 1% across three trials — and peaks ~5.3 cores, yet it alone reaches
the ceiling of about 18 clips/s that no fleet of 2–8 passes. So the wall is not a
core count, and the N=2 loss is not saturation either: at 370% combined each
instance's intra-op burst, sized for a 10-core box, time-shares cores with the
other process. **Oversubscription, not saturation.** What the two processes contend
on (memory bandwidth, cache, scheduler) is not measured here, and this document
does not record a mechanism it has not measured — the dip stands as a reproducible
loss, its cause open. A fleet remains for the multi-machine case only.

An earlier note here claimed one instance "used ~980% CPU, ten of ten cores". That
figure was borrowed from a shorter workload and never measured on this one; the
`--sample-cpu` numbers above replaced it. Keep the lesson with the correction: the
mechanism paragraph was written from a `top` reading of a different run.

**Metal inside a container: closed, not pending.** Docker Desktop passes no Metal
device through, so a torch build asking for `mps` inside one finds nothing and
falls back to CPU — silently, which is worse than failing — and no compose overlay
can say otherwise (there is no device reservation to write, unlike
`docker-compose.cuda.yml`'s `driver: nvidia`). Hence no `.mps` overlay at all. The
host is the only route to Metal; MPS/CoreML *training* there was evaluated and
abandoned, and `tensorflow-metal` does not pair with TF 2.21.0, so microWakeWord
has no Metal path at all. Kokoro is the one measured exception, below.

## TTS engine choice: the largest stage, and the one with a quality cost

Corpus generation is the largest stage, so the engine matters more than the
trainer. Three configurations, all on the M1 Max, all generating the SAME corpus
(22 voices, 13,183 positives, 6,600 negatives) so the comparison is engine-only:

| | corpus TTS | whole run |
|---|---|---|
| Kokoro-FastAPI in Docker | ~2h (estimate, not run) | not run |
| **Kokoro-FastAPI on the host** | **30m29s** | 47m56s |
| kokoro-mlx, in-process (`mlx://` then; `tcp://` since the refactor) | **19m06s** / 24m54s | 37m / 47m43s |

MLX generates 1.2–1.6x faster **and produced a worse corpus on run-on clips** —
plain positives are comparable. The per-run recall figures are a model-quality
measurement and live on branch `train/hey_seeree`; what belongs here is the
mechanism that was found, because it is checkable on any engine:

    run-on minus plain duration, which should be the spoken tail:
      FastAPI   697 - 580 = 117 ms
      MLX       645 - 610 =  35 ms      RUNON_TAIL_MS is 150-300

Run-on clips are the wake word plus a short tail of the following command, cut at a
word boundary the TTS engine reports; MLX was reporting that boundary ~34 ms early,
leaving almost no tail. A run-on with no tail is just a plain positive, so the model
never learns to fire when speech continues past the wake word. **If your run-on
recall is low, diff the durations before you touch a hyperparameter.** Fixing the
timestamp recovered run-on recall at every operating point, which confirms the cut
was the mechanism; it did not close the whole gap, and what remains — bf16 weights,
misaki phonemisation — is unexplained.

Two further engine facts that transfer. The same Kokoro-FastAPI version ran **3.7x**
faster in a host uv venv than in a container on this hardware, native arm64 both
times — the same torch/Accelerate story as every table above, and the reason
`src/scripts/start-kokoro-host.sh` exists. The standing warning attached to it:
`src/train/oww/train.py` records that server pinned at 101.8% CPU — exactly one
core — while the GPU sat at 21% and VRAM at 1.5 of 24 GB. The bottleneck was a
serialised stage, not compute. **"It initialised" and "it helped" are different
claims.** And voice count is not the lever: the same FastAPI corpus at 22 voices
scored *better* than an earlier one at 36, because 13 of those voices were `v0`
legacy variants of speakers already present — now skipped by default for every
engine (`LEGACY_VOICE_MARKER` in `src/train/corpus/negatives.py`), which takes ~36%
off the Kokoro clips at no measured quality cost.

**TWO NUMBERS PER MLX CELL, BECAUSE THE SPREAD IS THE POINT.** Identical corpus,
identical engine, two runs a day apart: TTS 19m06s then 24m54s, training 256 then
140 it/s. Nothing changed but what else the machine was doing. An earlier "MLX is
only 9% faster" reading came from a run while the machine was paging 540,000 times
against 296 MB of free swap, and was simply wrong.

## Measuring this on your own machine

The tools are in the repo and they are the only honest way to pick a route:

- `src/scripts/bench_tts.py` — per-engine clips/s, container vs host, same server
  version both sides. Note it feeds these tables; do not bias it by warming caches
  it would not warm in a real run.
- `src/scripts/bench_piper_fleet.py --trials 3 --sample-cpu` — the scaling question,
  with the load average recorded per trial and exact cumulative-CPU timing rather
  than a `top` reading.
- `src/scripts/tf_probe.py` / `src/train/train-applesilicon/.venv/bin/python
  src/scripts/coreml_probe.py` — op-level probes at the real shapes, in the library
  the target actually trains with. A probe in the wrong library is how the "native
  is slower" prior survived as long as it did.
- `--clips` on the CoreML probe, so its extrapolation is about your corpus rather
  than this one.

Record the load average, run the same-day comparison, and prefer a stage probe to a
whole-run total. Every conclusion above that survived did so because it was repeated
in one evening; the ones that did not are in the paragraph that starts "An earlier
note here claimed".
