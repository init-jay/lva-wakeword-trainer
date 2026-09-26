# This branch is a training record, not a pipeline

`train/hey_seeree` holds what was measured while training **hey seeree**: the numbers,
the curves behind them, the sweep configs that produced them, and the candidate that was
staged. It is deliberately never merged into `main`.

`main` carries the reusable pipeline only — trainers, corpus builders, gates, eval
harness, tools, and one worked example (`src/wordlists/hey_seeree.yaml`) showing what a
word's data file looks like. A measurement of one wake word on one machine is not part of
that. Mixing the two is what made the original PR #2 unreviewable: 40 files where a
mechanism change and a campaign result were interleaved, and no way to tell which lines
a different wake word would need.

## What is here that `main` does not have

| path | what it records |
|---|---|
| `SPEED.md` | route timings (host vs container, per engine) **and the model results those routes produced** |
| `src/train/sweeps/*.yaml` | the seven sweep configs that were actually run, plus the `HISTORY` block in `oww-training-steps.yaml`: corpus digests, clip counts, per-point wall time, and the verdict on each lever |
| `logs/scorecurves/*.csv` | 22 score curves — the evidence each `deploy/scorecards.jsonl` row cites in its `curve` field |
| `deploy/hey_seeree.md`, `deploy/scorecards.jsonl` | the staged candidate and one row per measurement (16 rows) |
| `deploy/esp32-mww/*.json`, `deploy/rpi-oww/*.json` | the ESPHome and LVA manifests for the staged models |
| `CLAUDE.md` | the dated measurement paragraphs `main` dropped |

The blobs themselves (`*.tflite`, `*.onnx`) are not tracked here either — `.gitignore`
excludes them on every branch, and no model binary has ever been committed to this repo.

## Errata

**`extend` and `hey_other` false-accept rates recorded before 2026-09-26 are
optimistic.** Nine phrases sat in both the training confusables and the eval corpus, so
they were trained on and then measured: `hey season`, `hey sedan`, `hey seizure`,
`hey serene`, `hey severe` (5 of `extend`'s 148 clips) and `hey Cynthia`, `hey Serena`,
`hey Sienna`, `hey Simon` (4 of `hey_other`'s 150). Fixed on `main` in
`cae46b2` by dropping them from the *training* side — `train.confusable` in
`src/wordlists/hey_seeree.yaml`, 38 → 29 — leaving the 366-clip eval set unchanged,
because the eval corpus is the measurement instrument. Rows scored before that commit are
comparable with each other and **not** with rows scored after it; the next corpus build
also has a different digest, so corpus reuse will not match and one rebuild is owed.

## Working on this branch

- Records only. A mechanism fix belongs on `main` through a PR; if it lands there first,
  merge `main` into this branch (never the reverse).
- `logs/*` is un-ignored for `logs/scorecurves/` **on this branch only** — see
  `.gitignore`. On `main`, all of `logs/` is ignored, because `main` has no rows to
  support.
- Every number written here should cite the curve or the run tag that produced it. A
  scorecard row without its `curve` file is a claim nobody can check.
- Never force-push `main`, and never merge this branch into it.
