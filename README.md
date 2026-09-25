# lva-wakeword-trainer

A wakeword training pipeline for [Linux Voice Assistant](https://github.com/OHF-Voice/linux-voice-assistant) - meant to run natively on an Apple Silicon Mac, or alteranatively via Docker on a Linux machine (with optional CUDA acceleration).

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

- A `microwakeword` model: `.tflite` plus the ESPHome `.json`
- An `openwakeword` model: `.onnx`, and a `.tflite` where the converter
  cooperates (on macOS the conversion aborts mid-run, a known recorded
  non-fatal — the `.onnx` is the model and the run prints the one-line
  late conversion)
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

1. **Write the wordlists** for your phrase.
2. **Stop and hand recording to you** — the recorder waits a real person
   at a microphone. It gives you the commands, then checks what comes back.
3. **Train the model**, once you confirm which
   target you want (oww/mww).
4. **Evaluate the model scorecard for you** — per category, per speaker, at matched
   false-accept counts — and say whether it is shippable.
5. **Stop and ask for live mic check**, - you can then say the wakeword and see model detection live on your machine.

Useful things to say along the way:

```
Why does it fire on "hey serious"?
Detection is poor for my daughter. What should I do?
Compare this run against the last one.
Is this good enough to ship?
```

### Prefer to drive it yourself?

**`make help`** is the entry point, and **[docs/MANUAL_RUN.md](docs/MANUAL_RUN.md)** has the
four steps end to end, with the commands for each of the three scripts in `src/scripts/`.
The change-one-thing loop itself is a tool: `src/scripts/sweep.py` runs a small
grid against a frozen corpus and appends every point to a per-wake-word
ledger (`output/<word>/runs.jsonl`); `python -m train.ledger --wake-word ...`
summarises it.


## Design choices

- **Apple Silicon native for training** - a host environment first, a
  multi-arch CPU/GPU container as the fallback.
- **A microphone and speaker are required**, for the voice capture and preflight check steps.
- **Agent first.** This repo is meant to be handed to an agent or coding harness. The in-repo skills are written so an agent can drive the whole pipeline and explain the performance of the deployment candidate to you. If you have deep knowledge of how oww and mww models work, you can also refer to the [manual run docs](docs/MANUAL_RUN.md) to execute the pipeline by hand and interpret the results yourself.

## How long it takes

The measurements behind the numbers below - and every comparison that led to a
route recommendation - live in [SPEED.md](SPEED.md). This section only tells
you what to expect.

| Step | CUDA box | Mac, host |
|---|---|---|
| Fetch external corpora | download-bound, ~43 GB for both targets | same |
| Record | human time, 20–50 clips per speaker | same |
| Train — microWakeWord (corpus, features, 10k steps, TFLite, manifest) | **~30m** | **~15m** |
| Train — openWakeWord (corpus, augmentation, features, 50k steps, TFLite) | **~30m** | **~35m** |
| Eval | minutes | minutes |
| Preflight | needs a mic | needs a mic |

- Corpus generation is the largest stage of either run; it speaks to a TTS
  engine (`src/tts-service/`: Kokoro on 8900, Piper via `src/scripts/start-tts-fleet.sh`),
  which is why it is a separate, server-needing stage the trainers can skip -
  `--skip-corpus` on the oww scripts, `SKIP_CORPUS=1` on mww - and the re-run
  flags save about 10-15 minutes off the numbers above.
