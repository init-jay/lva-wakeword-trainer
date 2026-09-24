# deploy/

The current deployment candidate **for each satellite**, staged here out of
`output/` so it is a stable, named thing rather than a file inside a directory the
trainers `rmtree` and repopulate every run. There are two targets and they are not
substitutes for each other:

| directory | satellite | engine | files the runtime loads |
|---|---|---|---|
| `esp32-mww/` | ESP32 | microWakeWord | `.tflite` + ESPHome `.json` (the manifest names its sibling; **do not rename either**) |
| `rpi-oww/` | Raspberry Pi | openWakeWord | `.tflite` + LVA `.json` manifest — LVA discovers models by globbing `*.json` only (`wake_word.py` `find_available_wake_words`); the manifest's `openWakeWord.probability_cutoff` carries the threshold and the filename stem is the model ID |

A staged candidate is `<word>_<tag>.tflite` beside its manifests. Two JSON files sit
next to a Raspberry Pi candidate and they are not interchangeable:

- `*.manifest.json` — what the runtime loads. Its filename stem is the model ID, and
  `openWakeWord.probability_cutoff` is the threshold the satellite actually runs at.
- `*.config.json` — the trainer's resolved config, kept beside the model for
  reproduction. It is not a runtime file.

## Where the measurements live

**Not in this file.** Every measured number for a staged or candidate model is one row of
[`scorecards.jsonl`](scorecards.jsonl) — append-only, like `output/<word>/runs.jsonl`: never
rewrite a row, corrections are new rows or errata in the git history. Each row carries its own
provenance (`tag`, `artifact`, `md5_prefix`, `threshold`, `positives`, `negatives`, `budget`,
`status`, `curve`) so a reader can tell whether two rows are comparable at all, and
`"unverified": true` marks rows nobody reproduced in a single clean pass.

Read it, don't copy it into prose:

```bash
python3 -c "
import json
for r in map(json.loads, open('deploy/scorecards.jsonl')):
    print(r['target'], r['id'], r['md5_prefix'], r['threshold'], r['status'])"
```

This file keeps the rules for changing what is staged and the traps that produce wrong
rows. Numbers belong in the JSONL and in the curve CSVs under `logs/scorecurves/`.

## FA budget

The operating constraint is a **false-accept rate on the adversarial negatives**, not a
threshold: `extend` + `hey_other` only, never pooled with the other categories and never
pooled across speakers. `tools/score_margins.py` derives the fire budget from the size of
the adversarial set it loaded (`ADV_FA_CONSTRAINT`, the largest count strictly inside it),
so the budget moves with the set instead of being a number somebody typed.

Every row is read at that budget or tighter, on the held-out real recordings, in **one
scoring pass over one set of clips**. Curves land in `logs/scorecurves/*.csv`; that
directory is gitignored, so a reviewer cannot open them — regenerate any row with

```bash
eval/.venv/bin/python tools/score_margins.py --model <path> \
  --csv logs/scorecurves/<name>.csv
```

adding `--sliding-window-size` for a bare `.tflite`, since a cutoff means nothing without
the window that produced it. Cross-day totals are not comparable: a fire count means
nothing without the set it was counted over, and the set is what changes between sessions.

### Which bytes a microWakeWord row belongs to

Every ESP32 row is the *content* of a run dir's weights file, and the tag alone cannot vouch
for it. Two reasons, both structural.

**The tag does not carry the shipped cutoff.** The `h-` half of a tag is a sha over the resolved
hyperparameters plus the seed (`train/provenance.py`), and `probability_cutoff` is not among
them — it cannot be, because the cutoff is read off the ROC *after* training
(`train/mww/train.py` prints it and says "Pick `probability_cutoff` from THIS, not from a
default"). The training-side `stride` and `window_step_ms` *are* in `config.json` and so are
hashed; what is not hashed is the inference `sliding_window_size` and `probability_cutoff`
that ship in the manifest's `micro` block. Two runs differing only in shipped cutoff
therefore share one tag.

**There are two copies of the weights and only one is per-run.** A manifest's `model` key is a
bare filename resolving to its **sibling**. But the corpus directory keeps a copy too
(`data/corpus/<word>/mww/<corpus-id>/tflite_stream_state_internal_quant/`), and that copy is
**shared by every run built against that corpus** — last write wins. Scoring from it yields
per-arm bytes wearing per-seed names, which is why `tools/score_margins.py` refuses a path
inside `data/corpus/` outright instead of warning. `prepare_corpus` rmtree's the corpus tree
at the start of every run, so those bytes are unrecoverable afterwards: a row traced to a
corpus dir has no evidence left, only a suspicion. What survives is the rule: score the run
dir, and treat the md5 `score_margins.py` prints as the row's identifier.

```bash
eval/.venv/bin/python tools/score_margins.py \
  --model output/<word>/mww/<tag>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --sliding-window-size <n>
```

openWakeWord does not have this problem: its output is one tagged `<tag>.onnx`/`<tag>.tflite`
per run. Anything staged into `esp32-mww/` must be md5-matched against its run-dir bytes
before a row beside it is trusted.

### What the manifest stage does and does not tell you

`train.mww.manifest` and the deployment budget do not measure the same thing, and each passes
a model the other fails. The reason is structural: manifest `faph` is computed on the
validation split, not on the adversarial negatives. So do not treat "manifest written" as
"usable inside the FA budget", and do not treat "manifest refused" as "no measured operating
point". The manifest stage is non-fatal by design (`run-mww-training-applesilicon.sh`), so a
sweep files the row either way.

### How the Raspberry Pi runtime resolves a model

Verified against `OHF-Voice/linux-voice-assistant` — re-check the line numbers before
relying on them, they are not ours to keep stable.

- `wake_word.py` globs `*.json` only, so **any** stray `.json` in the model directory becomes
  a selectable model.
- For openWakeWord the `.tflite` is resolved relative to the manifest's own directory: the
  pair must stay together.
- `probability_cutoff` comes from the manifest's `openWakeWord` block and **defaults to a
  loose value when the block is absent** — looser than any budget in this document, which is
  why the manifest ships with the weights instead of leaving the cutoff to a config default.
- The model ID is `Path.stem`, which strips exactly one suffix. A manifest named
  `<id>.manifest.json` therefore gets the ID `<id>.manifest`; every model LVA bundles itself
  is `<id>.json` beside `<id>.tflite`. Renaming a deployed manifest moves `WAKE_MODEL` with
  it, so the two change in one step or not at all.
- The manifest's `wake_word` field is a *display label*: HA builds its wake-word dropdown as
  a dict keyed on that phrase, so **two models sharing a label collapse to one option in the
  UI**. A candidate staged beside an incumbent needs a distinct label.
- `WAKE_MODEL` is only the *default*; the active model comes from HA's preference and
  overrides it, and the sensitivity setting follows the **slot**, not the model — re-set it
  after switching.

## What is never cleared by a measurement here

1. **Preflight, both targets, on a live mic.** Held-out TTS numbers are not a substitute:
   `cd preflight && uv run test_model.py --model ../deploy/esp32-mww/<candidate>.json`.
2. **Latency on the target runtime.** The sweep's scorecard times the host runtime; a
   `.tflite` on the actual satellite is a different runtime and has to be timed there.
3. **A `-dirty` tag.** It does not make a model unreproducible — the dirty files may not
   touch the training path at all — but it does make the tag unresolvable without the
   author's `git stash list`. Commit before training anything you intend to stage.
4. **Ranking arms on two seeds.** Two draws that move the same way are a direction, not a
   ranking.

## Rules for this directory

- **One candidate per target.** Replacing a staged file requires a measured win at matched
  false accepts, **per speaker**, on the same adversarial set and in one scoring pass —
  never at a fixed threshold, never pooled, never across days, and never against a figure
  measured on a differently-sized adversarial set.
- **New measurement → new `scorecards.jsonl` row**, with `artifact`, `md5_prefix`,
  `threshold`, `positives`, `negatives` and `budget` filled in. A row without them is not
  checkable, and an unchecked row gets cited as fact.
- **The threshold in each row is measured**, from the curve in `logs/scorecurves/`, not
  defaulted — and it is the threshold the satellite runs at, not a nearby one.
- **Corrections are new rows or errata in the git history**, never an edit to a row already
  written.
