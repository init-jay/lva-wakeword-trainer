# Working in this repo

A wake-word training pipeline for [Linux Voice Assistant]. It trains two models from
one corpus: **openWakeWord** (`.onnx`/`.tflite`, server) and **microWakeWord**
(`.tflite` + ESPHome `.json`, ESP32).

Four skills carry the detail and trigger on their own — `write-wordlists`,
`record-samples`, `train-apple-silicon`, `eval-models`. This file is the
orchestration between them.

## "I want a wake word model for X"

That single sentence is enough to start. Run the steps in this order, and stop at
each **STOP** until the human has done their part.

1. **Wordlists.** `wordlists/<x>.yaml`. Do this first: the adversarial phrases are
   built from the wake word's own consonants and vowels, and nothing downstream is
   meaningful without them. Use `write-wordlists`. Copy `wordlists/hey_seeree.yaml`
   as the worked example.
2. **STOP — recording.** You cannot do this. The recorder blocks on `input()` and
   needs a person at a microphone. Use `record-samples` to hand over the commands,
   then verify what comes back: clip counts, levels, alignment.
3. **External data**, once per machine: `./scripts/download-external-data.sh [all|oww|mww]`.
   ~43 GB for both. Check `df -h` first.
4. **Train.** `./scripts/run-oww-training.sh "X"` and/or `./scripts/run-mww-training.sh "X"`.
   On an Apple Silicon Mac both targets run on the host instead - the
   `train-apple-silicon` skill carries both routes, including the oww
   tflite-conversion workaround and the host Piper for mww. Hours.
   Ask which target they want before running both.
5. **Eval.** Use `eval-models`. Report per speaker and at matched false accepts.
6. **STOP — preflight.** Also needs their microphone:
   `cd preflight && uv run test_model.py --model ../output/<x>/mww/<tag>.json`

Ask which wake word and which target before starting anything expensive. Do not
guess a wake word from the repo's existing `hey_seeree` files.

## What you cannot do

- **Record or preflight.** Both need a human speaking into a microphone.
- **Use a GPU on a Mac.** Training itself now works there — `docker-compose.cpu.yml`
  builds both trainers multi-arch, native on arm64 — but on CPU only, and unmeasured
  as of 2026-09-04. Docker Desktop passes no Metal device through, so there is no MPS
  image to select and `docker-compose.mps.yml` is empty on purpose. Read
  `apple-port.md` before relitigating; it has the measurements and the phased plan.
- **Pick `probability_cutoff` from a default.** It comes from the measured ROC.

## Where things run

| step | machine |
|---|---|
| record, preflight | the human's machine, host, `uv` |
| train | the CUDA box, Docker - or anywhere, on CPU, with the `cpu` overlay; on a Mac, host (below) |
| eval | either — the `eval` image builds native on Apple Silicon |

On the CUDA box: `export COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml`.
NVIDIA only - `driver: nvidia` does not match an AMD card under ROCm.

Anywhere else, including a Mac: `docker-compose.yml:docker-compose.cpu.yml`, which
swaps both trainers for multi-arch CPU images. Slower, unmeasured as of 2026-09-04,
and the only in-Docker option on Apple Silicon - Docker Desktop passes no Metal
device through, so there is no MPS image to select and `docker-compose.mps.yml`
stays empty. Give Docker Desktop enough RAM first: the 17.28 GB feature array is
mmap'd, and running short of memory page-faults rather than erroring.
`SKIP_BUILD=1` on either training script reuses the image; needed after a
`docker builder prune`, since the rebuild is then cold.

A Mac can also train both targets entirely outside Docker: `train-applesilicon/`
is a host uv env (torch, `scripts/setup-applesilicon-trainer.sh`) run with
`scripts/run-oww-training-applesilicon.sh`, and `train-mww-applesilicon/` (TF,
`scripts/setup-mww-applesilicon-trainer.sh`) with
`scripts/run-mww-training-applesilicon.sh` - the mww corpus needs the host Piper
(`scripts/start-piper-host.sh`) because it is 100% Piper. That script starts no
tts servers of its own - its corpus calls go to a host uv venv
(`scripts/start-kokoro-host.sh`, `scripts/start-piper-host.sh`) or a Docker
service on a reachable port. The mww host route is the measured-faster one on a
Mac (full run measured 14m14s there against 26m06s in the container, 1.8x). Status, measurements, and the phased plan:
`apple-port.md`.

## Invariants that are easy to break

- **`data/recordings/holdout/` is never trained on.** The guarantee is *positional* —
  it is a sibling of `samples/`, not a child, because the trainer globs the samples
  tree recursively. `eval/paths.py` enforces and explains it.
- **`data/` is inputs and generated corpus; `output/` is models.** The trainers
  `rmtree` their corpus every run, so the split is what keeps that away from models.
- **Never compare models at a fixed threshold.** Two runs of an identical config
  measured 77% and 67% at 0.5. Use matched false accepts.
- **Never pool per-speaker results.** An average hides the voice that fails: 24% for
  a child against 97% for an adult, in the run that motivated the augmentation.
- **Never pool negative categories.** `extend` and `hey_other` carry the signal.
- **A tag is code + data**, `<commit>-d<audio hash>` — see `train/provenance.py`.
  Expect the data half to move between runs even when nothing changed; the TTS is
  not bit-reproducible.

## Conventions

Comments here explain *why*, usually with the measurement that settled it. Match
that. If you change something a comment justifies, update the comment in the same
edit. If a measurement is claimed, cite where it came from — several were found the
expensive way, and a plausible-sounding replacement is worse than none.

Verify before asserting. Much of what looks obvious in this repo is not: `os.mkdir`
is not recursive, `str.strip` is not `removesuffix`, and PyPI metadata does not
determine what pip installed. Each of those cost a training run.

[Linux Voice Assistant]: https://github.com/OHF-Voice/linux-voice-assistant
