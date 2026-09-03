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
