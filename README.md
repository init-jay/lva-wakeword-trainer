# lva-wakeword-trainer

A Docker-based wakeword training pipeline for [Linux Voice Assistant](https://github.com/OHF-Voice/linux-voice-assistant).

## Design choices

- **Docker only.** The pipeline is built around containers, so host-native and notebook-based training runs are explicitly unsupported.
- **A microphone and speaker are required**, for the voice capture and preflight check steps.
- **macOS is first class**, Linux is second class, and Windows is not supported.
- **GPU acceleration is optional**, and only applies to the training step.
- **Agent first.** This repo is meant to be handed to an agent or coding harness. The in-repo skills are written so an agent can drive the whole pipeline, but you can also run it by hand by following the docs. Each of the .py file's top comment has examples for how to run the file.


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
