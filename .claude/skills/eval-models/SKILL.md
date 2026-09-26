---
name: eval-models
description: Score trained wake-word models against held-out recordings and decide whether one is better than another (step 3 of the training pipeline). Use when the user wants to evaluate a model, compare two runs, check whether a model is good enough to ship, pick a threshold or a microWakeWord probability_cutoff, read a scorecard, or asks why a model that scored well is failing for someone.
---

# Evaluating trained wake-word models

Step 3. On a Mac it runs in the host uv env - `src/eval/.venv`, set up by
`./src/scripts/setup-eval-host.sh` - no Docker (improvement.md P2.4); everywhere else,
in the eval container built from `src/eval/` - its own compose project. Both carry
both inference stacks and neither trainer, because scoring a model and training
one have incompatible pins - and both pin the SAME deployment-runtime wheels
(`src/eval/pyproject.toml` and `src/eval/Dockerfile` must stay equal), so a number from
one checks against a number from the other.

## You can run this one yourself

Unlike `record-samples`, nothing here needs a human or a microphone. Run it, read it,
and report the numbers. The value you add is in the reading: almost every number this
harness prints has a way of being read wrongly, and the sections below are those ways.

## Preflight

The whole step is self-contained in `src/eval/` — the uv project and the Python
sources, or (container) a compose file and an image over the same sources.

Host, the Mac default, from the repo root:

```bash
./src/scripts/setup-eval-host.sh          # once per machine; idempotent
ls output/*/oww output/*/mww          # models to score
ls data/recordings/holdout/           # what to score them against
ls data/corpus/eval/negatives_tts/    # the adversarial corpus
```

The same checks, for the container:

```bash
cd src/eval
docker compose build            # first time, or after Dockerfile changes
cd ../..                        # back to the repo root for the data-relative paths below
ls output/*/oww output/*/mww       # models to score
ls data/recordings/holdout/           # what to score them against
ls data/corpus/eval/negatives_tts/    # the adversarial corpus
```

If the negatives are missing, `eval_model.py` and `compare_models.py` both exit before
scoring anything — 366 clips across 6 categories for this corpus, so it is not a long run.

Generation is self-contained — it speaks the TTS protocol to whichever Kokoro
engine you point it at. On a Mac that is the mlx engine (a uv project, not
Docker — see `src/tts-service/README.md`); on the training box it is the `kokoro`
compose service (CPU image, or the CUDA overlay).

Host, the mlx engine is a process on this same machine, so its loopback port is
the URL:

```bash
# the mlx engine, in another terminal on the Mac
uv run --project src/tts-service/engines/kokoro_mlx python -m kokoro_mlx_engine --port 8900

# from the repo root, plain-path form
src/eval/.venv/bin/python src/eval/src/generate_negatives.py --url tcp://127.0.0.1:8900
src/eval/.venv/bin/python src/eval/src/generate_positives.py \
    --url tcp://127.0.0.1:8900 --wake-word "hey seeree"
```

In the container, the eval project cannot reach the Mac's mlx engine by a name
it owns — the engine is a HOST process — so it goes through the host's
published port via `host.docker.internal` (`src/eval/docker-compose.yml` maps that
name to `host-gateway`, which Linux needs):

```bash
# from src/eval/
docker compose run --rm eval python -m eval.generate_negatives \
    --url tcp://host.docker.internal:8900
docker compose run --rm eval python -m eval.generate_positives \
    --url tcp://host.docker.internal:8900 --wake-word "hey seeree"
```

On the training box, `docker compose up -d kokoro` instead (with the GPU
overlay for the faster image — same command otherwise, and the two render the
same voices, so the corpora are interchangeable), and the URL is its compose
name:

```bash
# from the repo root
docker compose up -d kokoro
# from src/eval/
docker compose run --rm eval python -m eval.generate_negatives --url tcp://kokoro:8899
```

If the image pull hangs at "Pulling fs layer" with no bytes moving, it is the
multi-arch index — use the arch-suffixed tag instead, and see docker/Dockerfile.kokoro
for the diagnosis:

```bash
KOKORO_IMAGE=ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1-arm64 \
    docker compose up -d --build kokoro
```

On the training server, add the GPU overlay for the faster image — same command
otherwise, and the two render the same voices, so the corpora are interchangeable
(from the repo root; the `--url` above then points at that machine's port):

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d kokoro
```

`--dry-run` prints the wordlists without calling the server; use it to check the
phrases are right for the wake word first. The lists are tuned for a "hey siri"-like
phrase — `EXTEND`, `RUNNING` and `HEY_OTHER` carry nearly all the signal and need
retargeting for a different one.

## The commands

Two invocation forms, and which is which is load-bearing. Inside the image,
`src/eval/src/` is mounted AS the `eval` package, so the tools run as modules —
`python -m eval.X`, all from `src/eval/`, with model paths relative to the
container's `/app` workdir, which is the mounted repo root. On the host there
is no such package (`src/eval/` is a namespace directory, `src/eval/src/` its sources),
so the invocation is the plain-path form, from the repo root:

```bash
# HOST (the Mac default) - plain-path form, from the repo root

# the four gates, one model
src/eval/.venv/bin/python src/eval/src/eval_model.py \
    --model output/hey_seeree/oww/hey_seeree_705c23b.onnx

# is the new run better than the last one
src/eval/.venv/bin/python src/eval/src/compare_models.py \
    --models output/hey_seeree/oww/<new>.onnx output/hey_seeree/oww/<previous-best>.onnx

# openWakeWord candidate against the microWakeWord build
src/eval/.venv/bin/python src/eval/src/compare_models.py --models \
    output/hey_seeree/oww/hey_seeree_705c23b.onnx \
    output/hey_seeree/mww/hey_seeree_705c23b.json

# choosing a deployment operating point for one model
src/eval/.venv/bin/python src/eval/src/compare_models.py --models M --sweep

# CONTAINER - module form, from src/eval/
# the four gates, one model
docker compose run --rm eval python -m eval.eval_model \
    --model output/hey_seeree/oww/hey_seeree_705c23b.onnx

# the other three commands, with `docker compose run --rm eval python -m
# eval.compare_models` in place of `src/eval/.venv/bin/python src/eval/src/compare_models.py`
```

**Pass the microWakeWord `.json`, not its `.tflite`.** `MicroWakeWord.from_config()`
reads `probability_cutoff` and `sliding_window_size` from the manifest, so scoring the
JSON puts the manifest under test too. Score a bare `.tflite` and it is not.

No `--positives`/`--runon` needed: they default to every held-out speaker directory
under `data/recordings/holdout/`, with the `_runon` ones kept separate.

## Five ways to misread the output

**1 · Never compare two models at a fixed threshold.** What varies between training
runs is largely where the score distribution sits, not how well the model separates
the classes. Two runs of an *identical* configuration measured 77% and 67% at
threshold 0.5, and both reached 95% at matched false accepts. `compare_models.py`
prints the 0.5 row labelled "do NOT compare on this" — it is there as a reference
point, and it is the row that has produced the most wrong conclusions in this project.
Read the matched table.

**2 · Never pool the negatives.** The corpus is adversarial by construction — a fifth
of it is phrase-extending — so a pooled false-accept rate is meaningless. `extend` and
`hey_other` carry the signal; `general` is the background rate and should sit near
zero.

**3 · Never pool the speakers.** See the next section.

**4 · An openWakeWord score and a microWakeWord score share no scale.** One is a raw
probability, the other a sliding-window average of an int8 output. They are comparable
only through "detection at the operating point that admits N adversarial false
accepts", which is what the matched table asks of both. A mWW cutoff also means
nothing without its `sliding_window_size`, which is why it is printed with every
result.

**5 · A few points is not a result.** Two runs of one configuration have landed 10
points apart. If a change is worth keeping it should be visible across several matched
false-accept points, not at one.

## Per speaker is where the real failure hides

Every pooled number is an average over speakers, and an average is exactly what hides
the person the model does not work for. This is not hypothetical: the run that
motivated the child-range shifting in `src/train/corpus/augment.py` measured a 4-year-old
at **24% while the adult read 97%**, and the pooled figure looked healthy throughout.

So `eval_model.py` always prints a per-speaker table when there is more than one
speaker, and applies the positive-detection gate to the **weakest** speaker as well as
to the average. A model can read PASS pooled and FAIL for one voice:

```
  speaker                n    detected  median score  median latency
  speaker1              35       28/35         0.951           -34ms
  speaker2              10        9/10         0.956          -210ms
  speaker3               6         3/6         0.520           140ms

  [FAIL]  clean positive detection        40/51 (78%)     >= 98%
  [FAIL]  weakest speaker (speaker3)      3/6 (50%)       >= 98%
```

**A large spread across speakers is not fixed by retuning the threshold, and it is not
reliably fixed by augmentation.** It means the model generalises to some voices and not
others. The fix is real recordings from the speaker who fails — send the user back to
`record-samples`. Check the clip count first: a speaker with six holdout clips gives a
weak estimate, and "record more of them" applies to the holdout as much as the
training set.

`compare_models.py` shows the same breakdown at one matched false-accept point
(`--per-speaker-fa`, default 4), so a model that carries most speakers while dropping
one reads as a spread rather than as slightly-worse-overall.

## The gates

```
extend + hey_other false accepts at 0.5    < 2/32   (6%)
clean positive detection at 0.5            >= 55/56 (98%)
the same, for the weakest speaker          >= 55/56 (98%)
detection with a command immediately after >= 27/30 (90%)
median latency from end of speech          < 120 ms
```

Negative latency is normal here, not a bug. "End of speech" is the last sample above
2% of peak, and these recordings have a high noise floor, so the marker lands on room
tone *after* the phrase — the model fires on the phrase, correctly, before the marker.
That is why median and p90 are reported rather than the mean.

The command gate uses a splice, not a real recording: `eval_model.py` concatenates a
command onto each plain clip. If the 300 ms-pause number is much better than the
no-pause one, that is the signature of a model that learned the phrase is followed by
quiet. Real run-on recordings are scored separately, by `compare_models.py`.

## Choosing the microWakeWord probability_cutoff

**It comes from the measurement, not from a default.** Training writes
`tflite_streaming_roc_<commit>.txt` next to the model — false rejection rate and false
accepts per hour at every cutoff. Picking a number without reading it is the same
mistake as comparing models at 0.5. It resets the streaming model per clip, as
deployment does, only for runs trained with `src/train/patches/per-clip-stream-reset.py`
applied (from 2026-09-26); an older ROC streamed without resets and can hide a model
that fires on every cold start, so it is not comparable with a newer one.

The current `hey_seeree_705c23b.json` ships `probability_cutoff: 0.5`, and **0.5 does
not appear anywhere in that ROC** — the lowest row is 0.55. It was not derived from
this measurement. Worth fixing before it ships.

Two traps in that file, both visible in the one committed here:

```
Cutoff 1.00: frr=1.0000; faph=0.000     <- NOT an operating point
Cutoff 0.98: frr=0.1111; faph=0.187
Cutoff 0.91: frr=0.0672; faph=0.562
```

The `frr=1.0000` row is a **synthetic terminator** that microWakeWord appends to close
the curve when no measured cutoff reaches the faph floor. It is a model that rejects
everything, and it is the only row satisfying the default `--max-faph 0.0` — which is
how a manifest once shipped with `probability_cutoff: 1.0`, a model that could not
fire. `choose_cutoff` now discards `frr == 1.0` rows and errors instead. For this ROC
that means `--max-faph 0.0` is unachievable and you must state a real budget, e.g.
`--max-faph 0.2` → cutoff 0.98.

Second, **the ROC is a guide, not the measurement**: it is scored on ambient evaluation
sets, not on this repo's adversarial negatives, so `extend` false accepts are not in it
at all. Confirm any cutoff with `compare_models.py` against the holdout before
deploying it.

## Check the ledger before comparing two models

Before spending an eval — and before making any "this one is better" claim — look at
what has already been measured. Every completed run is appended to
`output/<wake_word>/runs.jsonl` (the sweep runner `src/scripts/sweep.py` files each grid
point there; a manual run files a line the same way), and
`src/train/train-mww-applesilicon/.venv/bin/python src/train/ledger.py --wake-word "X"` (any python
with PyYAML; the system `python3` works too) prints each `(target, setting)` group's
false-accept and detection rates as mean [min–max] across repeats, grouped by corpus
ID. If both models you are about to compare are already in the ledger, the answer is
in the file — read it instead of re-measuring. And remember what the min–max column
is: two runs of an *identical* config measured 77% and 67%, so the spread on a line is
the resolution of the measurement, not noise to average away. A candidate whose mean
lands inside the spread of an earlier config is inside the noise floor, and
"replacing the candidate in `deploy/` requires a measured win" holds across sessions,
not just within one.

The corpus-ID column is the independent variable, read before anything else: two runs
are comparable at all only if they share one (same corpus half of the tag). Different
corpus ID means different audio, and a "win" between the two can only have been caused
by the corpus, not the setting — exactly the uncontrolled variable the gates exist to
keep out of a comparison.

And never edit the file. It is append-only history: deleting or rewriting the
"not better" result is the same failure this whole section exists to stop, just with a
longer shelf life. A record with a null eval block means the eval did not run — it is
not a failed measurement and not a bad result, and it is deliberately distinguishable
from both.

## When a model is not better

The pipeline is the loop, and the rule is **change ONE thing**. Two runs of an
identical config differ by ~10 points, so a run that changed three things and moved
five points has told you nothing. Say which single variable moved, and re-run.

## Known gaps — check before promising output

- `src/train/mww/manifest.py` reads
  `output/<wake_word>/mww/<run-tag>/tflite_stream_state_internal_quant/`, one
  directory per run — microWakeWord refuses to train into an existing directory, so
  runs cannot be flattened. The commit-tagged files sitting directly in
  `output/<wake_word>/mww/` are a hand-made collection step; there is no script for
  it yet (`run-oww-training.sh` does the equivalent for the openWakeWord `.onnx` only).
- `check_model_alignment.py` on a `.tflite` needs `ai-edge-litert`, which neither
  the eval image nor the host env carries. Use the `.onnx`, or the trainer image.
- The comments refer to "the tuning log" and "tuning run N" — seventeen runs of this
  pipeline whose write-up is not published with the repo. The gate values above are
  the part that matters; treat a run number as provenance for a measurement, not as
  something you can go and read.
- `pymicro_wakeword/microwakeword.py:158` has an upstream `print(config)`, so every
  mWW run dumps the manifest dict to stdout. Not this repo's bug; ignore the line.

## The voice-holdout ranking set (improvement.md P1.2)

A third corpus at `data/corpus/eval/voice_holdout_tts/`, rendered by
`make render-voice-holdout` (Kokoro on 8900): positives from the voices
`src/wordlists/voice_holdout.yaml` **holds out of every corpus build** (oww and
mww trainers enforce the exclusion; the live TTS catalog is the source of
truth, so a stale list is an error, not a skip), at speeds inside the
0.7-1.3 training range. The held-out axis is therefore the voice alone:
it is a low-variance *ranking* signal for sweep points — a real n per
point — not a speaker-generalisation gate. A synthetic voice is not a
person, and the four gates above stay on the real held-out recordings in
`data/recordings/holdout/`, which the top two or three of the ranked
points go to. The directory is labelled by its own `set.json`; score it
with the separate block, never merged into the gates:

```bash
src/eval/.venv/bin/python src/eval/src/eval_model.py --model M --voice-holdout-set data/corpus/eval/voice_holdout_tts
```
