# lva-wakeword-trainer

A Docker-based wakeword training pipeline for [Linux Voice Assistant](https://github.com/OHF-Voice/linux-voice-assistant).

## The pipeline

```mermaid
flowchart LR
    REC["1 · Record<br/>real speech, live mic"]
    TRN["2 · Train<br/>openWakeWord + microWakeWord"]
    EVL["3 · Eval<br/>scorecard per model"]
    PRE["4 · Preflight<br/>live mic"]
    SHIP(["Deploy"])

    REC --> TRN --> EVL --> PRE --> SHIP
    EVL -->|"not better — change ONE thing"| TRN
```

Step 3 feeding back into step 2 is the pipeline. Most of your time is spent in that
loop, changing one thing per run. See [ARCHITECTURE.md](ARCHITECTURE.md) for what
happens inside each step.

## What you get after each run

- Two `.tflite` models, one `microwakeword` and one `openwakeword`
- A performance scorecard for each, measured under its LVA inference configuration
- Suggestions for improving the next training run (requires an AI agent pointed at this repo)



## Quick start

Open the repo in a coding agent and say what you want:

```
I want a wakeword model for "hey jarvis"
```

That is the whole starting instruction. [CLAUDE.md](CLAUDE.md) tells the agent the
order to work in, and the skills in `.claude/skills/` trigger on their own — you
should not need to name a script, a path, or a step.

It will then:

1. **Write the wordlists** for your phrase. These are the adversarial negatives built
   from your wake word's own sounds, and they are the one part that does not transfer
   between wake words.
2. **Stop and hand recording to you** — the recorder waits on ENTER and needs a person
   at a microphone. It gives you the commands, then checks what comes back: clip
   counts, levels, and where the speech sits in the detection window.
3. **Fetch the training corpora** and **run the training**, once you confirm which
   target you want. Hours, on a GPU box.
4. **Read the scorecard back to you** — per category, per speaker, at matched
   false-accept counts — and say whether it is shippable.
5. **Stop again for the live mic check**, which needs your room and your voice.

The two stops are real limits, not caution: both steps need a human speaking.

Useful things to say along the way:

```
Why does it fire on "hey serious"?
Detection is poor for my daughter. What should I do?
Compare this run against the last one.
Is this good enough to ship?
```

### Prefer to drive it yourself?

**[docs/MANUAL_RUN.md](docs/MANUAL_RUN.md)** has the four steps end to end, with the
commands for each of the three scripts in `scripts/`.


## Design choices

- **Docker by default, host-native on Apple Silicon.** The Docker route is the portable one and is what other machines run. On a Mac, both training targets have first-class host routes (`scripts/*-applesilicon.sh`), and for both of them the host is the measured-faster route there (see below and `apple-port.md`).
- **A microphone and speaker are required**, for the voice capture and preflight check steps.
- **macOS is first class**, Linux is second class, and Windows is not supported.
- **GPU acceleration is optional**, and only applies to the training step.
- **Agent first.** This repo is meant to be handed to an agent or coding harness. The in-repo skills are written so an agent can drive the whole pipeline and explain the performance of the deployment candidate to you. If you have deep knowledge of how oww and mww models work, you can also refer to the [manual run docs](docs/MANUAL_RUN.md) to execute the pipeline by hand and interpret the results yourself.



## How long it takes

Three environments, all measured with `time` on the same wake word:

- **CUDA box** — RTX 3090, 20 GB RAM, 4 cores, `docker-compose.cuda.yml`
- **Mac, Docker** — M1 Max, 64 GB, 10 cores, `docker-compose.cpu.yml`, no GPU
- **Mac, host** — same Mac, no container: `scripts/*-applesilicon.sh`

| Step | CUDA box | Mac, Docker | Mac, host |
|---|---|---|---|
| Fetch external corpora | download-bound, ~43 GB for both targets | same | same |
| Record | human time, 20–50 clips per speaker | same | same |
| Train — microWakeWord | **28m03s** | **26m06s** | **14m14s** |
| ├ TTS corpus (Piper + Kokoro) | included above | included above | **6m12s** (all-Piper) |
| ├ Features | included above | included above | 1m22s |
| └ Training, 10,000 steps + TFLite | included above | included above | **6m40s** |
| Train — openWakeWord | **28m59s** | ~2h 15m | **~37–48m** |
| ├ Kokoro corpus, `--kokoro-url mlx://` | included above | ~49 min | **~19–25m** |
| ├ Augmentation + features | included above | ~21 min | ~12m30s |
| └ Training, 50k steps | ~16 min | **~37 min** | **~4m35s–6m** |
| Eval | minutes | minutes | minutes |
| Preflight | needs a mic | needs a mic | needs a mic |

The Mac-host openWakeWord row is the two measured `mlx://` runs at 22 voices
(37m and 47m43s), not a sum of stages - the ±30% machine-load caveat in the
Kokoro section below applies. An earlier 36-voice host-server run took ~1h15m;
most of that difference is corpus size, not speed.

**On Apple Silicon, leaving the container is worth ~7x on the training stage.** 250
it/s on the host against 26 in the container — same corpus, same commit, same
torch 2.5.1, only the environment differs. The macOS wheels link Accelerate and the
linux/arm64 ones do not; the container also mmaps a 16 GB feature array across
Docker Desktop's filesystem boundary, which the host reads natively. Those two have
not been separated, so treat "7x" as the combined effect rather than a claim about
either one. `apple-port.md` phase 1b has the component measurements.

**And it was not a trade against quality.** Evaluated on the same holdout, the
host-trained model beat the container-trained one at every matched false-accept
point and for every speaker - jay_runon, the largest sample at n=57, went 35% ->
51%. Two runs is not proof, and this repo has measured 10 points of run-to-run
variance before, but the direction was consistent everywhere.

**microWakeWord is also faster on the host on a Mac, and the host is the default route there.** The old reason to leave it in Docker was measured in the wrong library: macOS torch ships without oneDNN and measured **12x slower on conv1d**, and mixednet is the convolutional model — a fair prior, but that probe was torch, and mWW trains with TF 2.21.0. The same probe in TF (`tools/tf_probe.py`) reads 36.21 ms/step in the container against 30.86 on the host, with threading making both worse: at these shapes each build runs one op per thread, and the GEMM-bound rest is exactly where Accelerate pays. The other half of a run, TTS, is where most of a run goes, and the host servers are faster for both engines in the corpus: Piper 2.4x (21.66 vs 9.13 clips/s, same versions; the macOS wheel links Accelerate, the linux/arm64 one does not), and the same Kokoro-FastAPI v0.8.1 that the openWakeWord host route already uses (the section below documents its host-vs-container wall time). The corpus itself is now Piper-majority with a 30% Kokoro mix on the host route (the mirror of openWakeWord's 70/30): the all-Piper mix measured 6m12s, the first mixed run 7m24s - the same full-run wall time, and the mix is what moved run-on detection from 34% to 65% at matched false accepts on that same-day pair. The full run follows: 14m14s on the host against 26m06s in the container on the same Mac — a day apart, so treat 1.8x as directional and trust the same-day stage measurements behind it. Doubling corpus depth and training steps did not produce a deployable model in either engine mix: at 100% Piper it kept the best held-out detection of any run but lost its FAPH operating point entirely (the manifest stage refused to write it), and with the 30% mix it collapsed on held-out — the Kokoro share's benefit is at 1x, not scalable (`apple-port.md` phase 3, `train/mww/corpus.py`). The Docker route stays correct for every other machine, and on a Mac it is still a working fallback; `apple-port.md` phase 3 carries the full table and the caveats.

Do not read the mww result as "the GPU is pointless" either. It is a claim about one
25,537-parameter model, too small to fill a 3090, in a run that is mostly not
training: Piper corpus generation is CPU-only on both machines by choice, since
`--use-cuda` measured 2.5x *slower*.

### Kokoro TTS: three ways to run it, and the speed/quality tradeoff

Corpus generation is the largest stage, so the TTS engine matters more than the
trainer. Three configurations, all on the M1 Max, all generating the SAME corpus
(22 voices, 13,183 positives, 6,600 negatives) so the comparison is engine-only:

| | corpus TTS | whole run | run-on recall @ 6/32 FA |
|---|---|---|---|
| Kokoro-FastAPI in Docker | ~2h (est.) | not run | not run |
| **Kokoro-FastAPI on the host** | **30m29s** | 47m56s | **82%** |
| kokoro-mlx, in-process (`--kokoro-url mlx://`) | **19m06s** / 24m54s | 37m / 47m43s | 51% -> 61% |

**MLX generates the corpus 1.2-1.6x faster and produces a worse corpus on
run-on; the timing table above assumes `mlx://`, and the host FastAPI server is
the alternative when the run-on gap matters.** The regression is specific to
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
See `BUGREPORT-kokoro-mlx.md` and `train/corpus/kokoro_mlx.py`.

**What this settled:** voice count is not the problem. The same FastAPI corpus at 22
voices scored BETTER than an earlier one at 36 (82/65 against 76/56 at 4/32), so the
13 voices MLX lacks cost nothing - they are `v0` legacy variants of speakers it
already has, not distinct ones. Those are now skipped by default for every engine,
which takes ~36% off the Kokoro clips: see `LEGACY_VOICE_MARKER` in
`train/corpus/negatives.py`.

MLX remains worth having for plain clips and negatives, which are 79% of the corpus
and show no regression. That is the split to build if the remaining run-on gap turns
out not to be closable.

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