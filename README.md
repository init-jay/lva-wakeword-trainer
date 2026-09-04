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

- **Docker only.** The pipeline is built around containers, so host-native and notebook-based training runs are explicitly unsupported.
- **A microphone and speaker are required**, for the voice capture and preflight check steps.
- **macOS is first class**, Linux is second class, and Windows is not supported.
- **GPU acceleration is optional**, and only applies to the training step.
- **Agent first.** This repo is meant to be handed to an agent or coding harness. The in-repo skills are written so an agent can drive the whole pipeline and explain the performance of the deployment candidate to you. If you have deep knowledge of how oww and mww models work, you can also refer to the [manual run docs](docs/MANUAL_RUN.md) to execute the pipeline by hand and interpret the results yourself.



## How long it takes

Two machines, both measured end to end with `time`:

- **CUDA box** — RTX 3090, 20 GB RAM, 4 cores, `docker-compose.cuda.yml`
- **Mac** — M1 Max, 64 GB, 10 cores, `docker-compose.cpu.yml`, no GPU at all

| Step | CUDA box | Mac, CPU only |
|---|---|---|
| Fetch external corpora | download-bound, ~43 GB for both targets | same |
| Record | human time, 20–50 clips per speaker | same |
| Train — microWakeWord | **28m03s** | **26m06s** |
| Train — openWakeWord | **28m59s** | not yet measured |
| Eval | minutes | minutes |
| Preflight | needs a mic | needs a mic |

**The Mac is not slower, and for microWakeWord it was faster.** That is not a quirk:
the mWW model is 25,537 parameters, too small to fill a 3090, and most of a run is
not training at all. Piper corpus generation is CPU-only on both machines by
choice — `--use-cuda` measured 2.5x *slower* — so the majority of the work ran on
4 cores on the VM and 10 on the Mac. The GPU's advantage applied to a small slice
while its weaker CPU applied to the rest.

Do not read that as "the GPU is pointless". It is a claim about one small model.
openWakeWord is a different shape — it mmaps a 17.28 GB feature array and trains a
much larger network — and has not been measured on CPU yet.

Both figures are full script runs at defaults, not the training stage alone:
`run-mww-training.sh` covers corpus, features, training and manifest;
`run-oww-training.sh` covers TTS generation, augmentation, training and the tflite
conversion. The two targets are independent, so neither needs the other first.

Three things move these numbers more than the hardware does. `SKIP_CORPUS=1` skips
corpus generation on a re-run, which is most of the openWakeWord figure. Docker
Desktop's memory limit decides whether that 17.28 GB array is mmap'd or thrashed,
and falling short page-faults rather than erroring. And on the CUDA box, holding the
card alone is the difference between finishing and not: a Kokoro server left up cost
a run a 16.09 GiB allocation with 15.34 GiB free.