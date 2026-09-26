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

1. **Wordlists.** `src/wordlists/<x>.yaml`. Do this first: the adversarial phrases are
   built from the wake word's own consonants and vowels, and nothing downstream is
   meaningful without them. Use `write-wordlists`. Copy `src/wordlists/hey_seeree.yaml`
   as the worked example.
2. **STOP — recording.** You cannot do this. The recorder blocks on `input()` and
   needs a person at a microphone. Use `record-samples` to hand over the commands,
   then verify what comes back: clip counts, levels, alignment.
3. **External data**, once per machine: `./src/scripts/download-external-data.sh [all|oww|mww]`.
   ~43 GB for both. Check `df -h` first.
4. **Train.** `./src/scripts/run-oww-training.sh "X"` and/or `./src/scripts/run-mww-training.sh "X"`.
   On an Apple Silicon Mac both targets run on the host instead - the
   `train-apple-silicon` skill carries both routes, including the oww
   tflite-conversion workaround and the host Piper for mww. Hours.
   Ask which target they want before running both.
5. **Eval.** Use `eval-models`. Report per speaker and at matched false accepts.
6. **STOP — preflight.** Also needs their microphone:
   `cd src/preflight && uv run test_model.py --model ../../output/<x>/mww/<tag>.json`

Ask which wake word and which target before starting anything expensive. Do not
guess a wake word from the repo's existing `hey_seeree` files.

## What you cannot do

- **Record or preflight.** Both need a human speaking into a microphone.
- **Pick `probability_cutoff` from a default.** It comes from the measured ROC.
- **Treat `deploy/` as staging ground.** It stages ONE candidate with its measured
  status written down (`deploy/README.md`), out of `output/` because trainers rmtree
  that tree every run. Replacing the candidate requires a measured win at matched
  false accepts, per speaker. The adversarial negative set is not a constant: it
  has been widened when the resolution of the matched-FA comparison demanded it
  (with few clips, a single clip was worth several points of every row), and a
  scorecard measured on an older negative set is not comparable to one on the
  current set, so the staged
  candidate must be re-baselined before a ship call.
- **Retune in search of the weak voice.** The weakest-detection speaker in the
  campaign this pipeline came out of was a child: it stayed the weakest at every
  measured point and step count, and no hyperparameter moved it. More recordings
  from that speaker is the only lever the measurements have pointed at.

## Where things run

| step | machine |
|---|---|
| record, preflight | the human's machine, host, `uv` |
| train | the CUDA box, Docker - or anywhere, on CPU, with the `cpu` overlay; on a Mac, host (below) |
| eval | either — the `eval` image builds native on Apple Silicon |

`make help` is the entry point: the Makefile at the repo root carries the commands
a human actually types, each annotated with what it costs (minutes, GB, TTS).

On the CUDA box: `export COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml`.
NVIDIA only - `driver: nvidia` does not match an AMD card under ROCm.

Anywhere else, including a Mac: `docker-compose.yml:docker-compose.cpu.yml`, which
swaps both trainers for multi-arch CPU images. Slower, and the only in-Docker
option on Apple Silicon - Docker Desktop passes no Metal
device through, so there is no MPS image to select and `docker-compose.mps.yml`
stays empty. Give Docker Desktop enough RAM first: the feature array is
mmap'd at multi-GB scale, and running short of memory page-faults rather than
erroring.
`SKIP_BUILD=1` on either training script reuses the image; needed after a
`docker builder prune`, since the rebuild is then cold.

A Mac can also train both targets entirely outside Docker: `src/train/train-applesilicon/`
is a host uv env (torch, `src/scripts/setup-applesilicon-trainer.sh`) run with
`src/scripts/run-oww-training-applesilicon.sh`, and `src/train/train-mww-applesilicon/` (TF,
`src/scripts/setup-mww-applesilicon-trainer.sh`) with
`src/scripts/run-mww-training-applesilicon.sh`. Neither script starts TTS of its
own - the corpus clients speak the TTS protocol in `src/tts-service/` to the engine
uv projects there (kokoro-mlx on 8900, in-process piper-tts on 8898; see
`src/tts-service/README.md`), or a Docker service on a reachable port. The mww
host route is the measured-faster one on a Mac; its corpus is Piper-majority
with a 30% Kokoro mix by default (`KOKORO_FRACTION=0` for the all-Piper corpus). Do not
scale corpus depth in search of quality: doubling it (with double the training
steps) produced no deployable model in either engine mix - the runs that
measured that live on branch `train/hey_seeree`.
The route rationale and the measurements: SPEED.md on branch `train/hey_seeree`.

## Invariants that are easy to break

- **`data/recordings/holdout/` is never trained on.** The guarantee is *positional* —
  it is a sibling of `samples/`, not a child, because the trainer globs the samples
  tree recursively. `src/eval/src/paths.py` enforces and explains it.
- **`data/` is inputs and generated corpus; `output/` is models.** The trainers
  `rmtree` their corpus every run, so the split is what keeps that away from models.
- **Never compare models at a fixed threshold.** Two runs of an identical config
  measured very different detection at the same fixed threshold, and the same
  quality at matched false accepts. Use matched false accepts.
- **Never pool per-speaker results.** An average hides the voice that fails:
  the run that motivated the child-range augmentation looked healthy pooled
  while one speaker's detection was far below the rest.
- **Never pool negative categories.** `extend` and `hey_other` carry the signal.
- **A tag is code + data**, `<commit>-d<audio hash>` — see `src/train/provenance.py`.
  Expect the data half to move between runs even when nothing changed; the TTS is
  not bit-reproducible.
- **`--seed` is a guarantee, not a convenience.** Same seed gives a byte-identical
  `.onnx` (verified across repeated runs: same md5). It holds because the global
  RNG seed covers augmentation and the checkpoint merge is a NO-OP on this corpus
  (`cleared=0/N` in every real run — the conjunction gate never fires, which is also
  why the bar was achievable). The ledger prints a DETERMINISM REGRESSION line when
  a duplicate (config, seed) pair's evals disagree — the standing check on this bar.
- **The run ledger is append-only** (`output/<word>/runs.jsonl`): never rewrite it,
  and a duplicate (target, tag) is refused. Corrections are errata in the git
  history, not edits to the file.
- **Known open: `--seed` is not in the corpus-reuse request.** `matches_requested`
  special-cases it and the manifest records it, but neither trainer passes it — so if
  the corpus augmentation consumes `--seed`, a different-seed run silently reuses
  another seed's corpus (the same one-axis hole the voice-set fix closed, one axis
  over). Decide whether "corpus fixed at build seed; the seed varies only
  features/training" is the intended semantic: then say so in the help text; otherwise
  pass the seed in the request.

## Conventions

Comments here explain *why*, usually with the measurement that settled it. Match
that. If you change something a comment justifies, update the comment in the same
edit. If a measurement is claimed, cite where it came from — several were found the
expensive way, and a plausible-sounding replacement is worse than none.

The plan file (`improvement.md`) and the review file (`bug.md`) were removed
from the repo: the measurements they carried live in SPEED.md on branch
`train/hey_seeree`,
the incident history in the git log and the test docstrings. Code comments that
cite "improvement.md P.." or "bug.md B.." are provenance for the commit that
implemented or fixed the finding, not links to keep alive.

Verify before asserting. Much of what looks obvious in this repo is not: `os.mkdir`
is not recursive, `str.strip` is not `removesuffix`, and PyPI metadata does not
determine what pip installed. Each of those cost a training run.

[Linux Voice Assistant]: https://github.com/OHF-Voice/linux-voice-assistant
