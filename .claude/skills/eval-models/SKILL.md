---
name: eval-models
description: Score trained wake-word models against held-out recordings and decide whether one is better than another (step 3 of the training pipeline). Use when the user wants to evaluate a model, compare two runs, check whether a model is good enough to ship, pick a threshold or a microWakeWord probability_cutoff, read a scorecard, or asks why a model that scored well is failing for someone.
---

# Evaluating trained wake-word models

Step 3. Runs in the `eval` container — it carries both inference stacks and neither
trainer, because scoring a model and training one have incompatible pins.

## You can run this one yourself

Unlike `record-samples`, nothing here needs a human or a microphone. Run it, read it,
and report the numbers. The value you add is in the reading: almost every number this
harness prints has a way of being read wrongly, and the sections below are those ways.

## Preflight

```bash
docker compose build eval            # first time, or after Dockerfile.eval changes
ls output/*/oww output/*/mww         # models to score
ls data/recordings/holdout/          # what to score them against
ls data/corpus/eval/negatives_tts/   # the adversarial corpus
```

If the negatives are missing, `eval_model.py` and `compare_models.py` both exit before
scoring anything — 100 utterances across 6 categories, so it is not a long run.

Generation is self-contained — the `kokoro` service defaults to the CPU image, which
publishes linux/arm64, so it runs natively on the same Mac as the eval image. Measured
on an M-series Mac: ready in ~5 s, ~1 s per clip, the whole 100-clip corpus in under
two minutes.

```bash
docker compose up -d kokoro
docker compose run --rm eval python -m eval.generate_negatives \
    --url http://kokoro:8880/v1/audio/speech
docker compose stop kokoro
```

If the image pull hangs at "Pulling fs layer" with no bytes moving, it is the
multi-arch index — use the arch-suffixed tag instead, and see docker/Dockerfile.kokoro
for the diagnosis:

```bash
KOKORO_IMAGE=ghcr.io/remsky/kokoro-fastapi-cpu:v0.8.1-arm64 \
    docker compose up -d --build kokoro
```

On the training server, add the GPU overlay for the faster image — same command
otherwise, and the two render the same voices, so the corpora are interchangeable:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d kokoro
```

`--dry-run` prints the wordlists without calling the server; use it to check the
phrases are right for the wake word first. The lists are tuned for a "hey siri"-like
phrase — `EXTEND`, `RUNNING` and `HEY_OTHER` carry nearly all the signal and need
retargeting for a different one.

## The commands

```bash
# the four gates, one model
docker compose run --rm eval python -m eval.eval_model \
    --model output/hey_seeree/oww/hey_seeree_705c23b.onnx

# is the new run better than the last one
docker compose run --rm eval python -m eval.compare_models \
    --models output/hey_seeree/oww/<new>.onnx output/hey_seeree/oww/<previous-best>.onnx

# openWakeWord candidate against the microWakeWord build
docker compose run --rm eval python -m eval.compare_models --models \
    output/hey_seeree/oww/hey_seeree_705c23b.onnx \
    output/hey_seeree/mww/hey_seeree_705c23b.json

# choosing a deployment operating point for one model
docker compose run --rm eval python -m eval.compare_models --models M --sweep
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
motivated the child-range shifting in `train/corpus/augment.py` measured a 4-year-old
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
mistake as comparing models at 0.5.

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

## When a model is not better

The pipeline is the loop, and the rule is **change ONE thing**. Two runs of an
identical config differ by ~10 points, so a run that changed three things and moved
five points has told you nothing. Say which single variable moved, and re-run.

## Known gaps — check before promising output

- `train/mww/manifest.py` reads
  `output/<wake_word>/mww/<run-tag>/tflite_stream_state_internal_quant/`, one
  directory per run — microWakeWord refuses to train into an existing directory, so
  runs cannot be flattened. The commit-tagged files sitting directly in
  `output/<wake_word>/mww/` are a hand-made collection step; there is no script for
  it yet (`run-oww-training.sh` does the equivalent for the openWakeWord `.onnx` only).
- `check_model_alignment.py` on a `.tflite` needs `ai-edge-litert`, which the eval
  image does not carry. Use the `.onnx`, or the trainer image.
- The comments refer to "the tuning log" and "tuning run N" — seventeen runs of this
  pipeline whose write-up is not published with the repo. The gate values above are
  the part that matters; treat a run number as provenance for a measurement, not as
  something you can go and read.
- `pymicro_wakeword/microwakeword.py:158` has an upstream `print(config)`, so every
  mWW run dumps the manifest dict to stdout. Not this repo's bug; ignore the line.
