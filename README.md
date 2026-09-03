# lva-wakeword-trainer
Docker based wakeword training pipeline for [linux voice assistant](https://github.com/OHF-Voice/linux-voice-assistant).


## Designed choices

- Docker, this is not designed to run without docker so vanilla host based or notebook based training runs are explicitly not supported
- Machine with microphone and speaker for voice capture and preflight check steps
- MacOS as first class, linux second, windows not supported
- Optional GPU acceleration for training step
- Agent first approach, this repo is designed for you to point an agent or coding harness at. You can manually run it as well by following the docs, but in-repo skills are written for agents to be able to drive the training pipeline for you.

## What you get after each run

- 2 .tflite formatted `microwakeword` and `openwakeword` models
- performance score cards for each tested using their LVA inference configuration
- suggestions for how to improve next training run (AI agent required, point it at this repo)



