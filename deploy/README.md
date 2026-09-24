# deploy/

The current deployment candidate **for each satellite**, staged here out of
`output/` so it is a stable, named thing rather than a file inside a directory the
trainers `rmtree` and repopulate every run. There are two targets and they are not
substitutes for each other:

| directory | satellite | engine | files the runtime loads |
|---|---|---|---|
| `esp32-mww/` | ESP32 | microWakeWord | `.tflite` + ESPHome `.json` (the manifest names its sibling; **do not rename either**) |
| `rpi-oww/` | Raspberry Pi | openWakeWord | `.tflite` + LVA `.json` manifest — LVA discovers models by globbing `*.json` only (`wake_word.py` `find_available_wake_words`); the manifest's `openWakeWord.probability_cutoff` carries the threshold and the filename stem is the model ID |

`rpi-oww/…hf37e3df.config.json` is the trainer's resolved config, kept beside the
model for reproduction. It is not a runtime file.

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
    if r['target']=='mww' and r['status']!='unreachable':
        print(r['id'], r['md5_prefix'], r['fa'], r.get('adults'), r['status'])"
```

This file keeps what is staged, the rules for changing it, and the traps that produced wrong
rows. Numbers belong in the JSONL and in the curve CSVs under `logs/scorecurves/`.

## What is staged

```
esp32-mww/hey_seeree_ecbf160-dirty-da01854d.{tflite,json}   unchanged since 2026-09-08
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.tflite   staged 2026-09-24
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.config.json
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.manifest.json   LVA discovery manifest (added 2026-09-25)
```

Their measured operating points are `oww-hf37e3df-584` and `mww-da01854d-150` (with
`mww-da01854d-009` beside it: what the ESP32 cutoff *actually* ships at, which is above the
budget and worth knowing before anyone quotes the staged row).

`*.manifest.json` is the LVA discovery manifest (model ID = its filename stem,
`probability_cutoff` 0.60), not the trainer's resolved config (`*.config.json`);
stage both the `.tflite` and the `.manifest.json`.

**Open item, naming.** `wake_word.py:40` sets the model ID to `Path.stem`, which strips one
suffix: the staged file's ID is `hey_seeree_55e182a-dirty-c9897b91-hf37e3df.manifest`, where
every model LVA bundles is `<id>.json` beside `<id>.tflite` (`hey_jarvis.json`, `alexa.json`) and
`WAKE_MODEL` is matched against the stem. Renaming the manifest to drop `.manifest` would also
clean the ID and the HA label, but the satellite is already deployed against the current name and
`WAKE_MODEL` would have to move in the same step, so it is left alone and recorded here. Verified
against OHF-Voice/linux-voice-assistant@HEAD 2026-09-24: `:34` globs `*.json` only — a `.json`
left in the model dir by anything else becomes a selectable model; `:52-53` for openWakeWord the
`.tflite` is resolved relative to the manifest's own directory, so the pair must stay together;
`:70` reads `model_config["openWakeWord"]["probability_cutoff"]` and defaults to **0.7** when the
block is absent — looser than any budget in this document, which is why the manifest ships with
the weights instead of leaving the cutoff to a config default; `models.py:38` fixes the type
string as `"openWakeWord"`.
On the Pi they land in the dir `WAKE_WORD_DIR` points at — on the current install that is the
`lva_wakeword_custom` volume (`/app/wakewords/custom`), host path
`/var/lib/docker/volumes/lva_wakeword_custom/_data/` — and `WAKE_MODEL` is set in the compose `.env`
to the manifest's stem. The manifest's `wake_word` field is a *display label*: HA builds its wake-word
dropdown as a dict keyed on that phrase (`esphome/select.py`), so two models sharing a label collapse
to one option in the UI — the staged v0.2 manifest therefore reads `"Hey Seeree v0.2"` to stay
selectable beside the house `hey_seeree_v0.1`. The `WAKE_MODEL` env is only the *default*; the active
model comes from HA's preference and overrides it, and the Wake Word 1 sensitivity (0.700 persisted)
follows the slot, not the model — re-set it to 0.60 after switching.

**Run the Raspberry Pi model at threshold 0.58–0.60, not at 0.5.** That band is flat in
detection and it is the tight edge of the measured curve; the numbers are in
`oww-hf37e3df-584`. 0.5 is not this model's operating point and nothing about it is calibrated
there. The ESP32 manifest's `probability_cutoff: 0.09` is unchanged and remains *its* measured
point — see `mww-da01854d-009` for why that point is above budget.

## FA budget

The operating constraint is **FA < 2% of the 298-clip adversarial set = at most 5 fires**. Every
JSONL row is read at that budget or tighter, on the real 51-clip holdout (jay 35, jen 10, ryan 6),
in **one scoring pass over one set of clips** (`tools/score_margins.py`, peaks per clip; curves in
`logs/scorecurves/*.csv` — that directory is gitignored, so a reviewer cannot open them: regenerate
any row with

```bash
eval/.venv/bin/python tools/score_margins.py --model <path> --adv-fa-budget 5 \
  --csv logs/scorecurves/<name>.csv
```

adding `--sliding-window-size 5` for a bare `.tflite`, since a cutoff means nothing without the
window that produced it). All saved curves were checked to sit on the same 298 adversarial clips
(extend 148 + hey_other 150) and the same 51 positives, so a fixed fire count is comparable across
those files. Cross-day totals are not comparable and no row pools across speakers.

The Pi candidate is preferred over the model that was firing in the house not because it wins at
the loose end — at ≤ 5 fires it is a one-clip tie — but because its curve holds as the budget
tightens and the house model's does not. Both ends are rows: `oww-hf37e3df-584`, `oww-house-783`,
`oww-house-670`.

### Which bytes an ESP32 row belongs to

Every ESP32 row is the *content* of a run dir's weights file, and the tag alone cannot vouch for
it. Two reasons, both learned the hard way.

**The tag does not carry the shipped cutoff.** The `h-` half of a tag is a sha over the resolved
hyperparameters plus the seed (`train/provenance.py:37`), and `probability_cutoff` is not among
them — it cannot be, because the cutoff is read off the ROC *after* training
(`train/mww/train.py:510` prints it and says "Pick `probability_cutoff` from THIS, not from a
default"). The training-side `stride` and `window_step_ms` *are* in `config.json` and so are hashed;
what is not hashed is the inference `sliding_window_size` and `probability_cutoff` that ship in the
manifest's `micro` block. Two runs differing only in shipped cutoff therefore share one tag.

**There are two copies of the weights and only one is per-run.** A manifest's `model` key is a bare
filename resolving to its **sibling** — audited 2026-09-25: 23/23 run manifests resolve inside their
own directory and every md5 is distinct. But the corpus directory keeps a copy too
(`data/corpus/hey_seeree/mww/<corpus-id>/tflite_stream_state_internal_quant/`), and that copy is
**shared by every run built against that corpus** — last write wins. Scoring from it yields
per-arm bytes wearing per-seed names. `prepare_corpus` rmtree's the corpus tree at the start of
every run, so those bytes are unrecoverable afterwards: a row traced to a corpus dir has no
evidence left, only a suspicion. What survives is the rule: score the run dir, and treat the md5
`score_margins.py` prints as the row's identifier.

```bash
eval/.venv/bin/python tools/score_margins.py \
  --model output/hey_seeree/mww/<tag>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --sliding-window-size 5 --adv-fa-budget 5
```

`scripts/score-esp32-arms.sh` runs exactly that over the eight 10x ESP32 models in one pass and
prints each md5, which is how the `unverified` flag on those rows gets cleared.

openWakeWord does not have this problem: its output is one tagged `<tag>.onnx`/`<tag>.tflite` per
run. Anything staged into `deploy/esp32-mww/` must be md5-matched against its run-dir bytes before
a row beside it is trusted.

### What the manifest stage does and does not tell you

`train.mww.manifest` and the deployment budget do not measure the same thing, and each passes a
model the other fails. Of the eight 10x runs it refused two — including one that reads as usable
on the holdout — while three rows have no budget point at all, two of *those* having manifests.
The reason is structural: manifest `faph` is computed on the validation split, not on the
adversarial negatives. So do not treat "manifest written" as "usable at FA<2%", and do not treat
"manifest refused" as "no measured operating point". The manifest stage is non-fatal by design
(`run-mww-training-applesilicon.sh`), so the sweep files the row either way.

## Corrections to what is written above

- **2026-09-25 — retracts the 2026-09-24 entry below.** Re-scoring the ESP32 models from their own
  run dirs showed the flat arm's best-seed row had described a model that does not exist: those
  bytes fire on every adversarial clip at any threshold. The retraction ran both ways — balancing
  *is* a win on this engine (the direction the 2026-09-24 entry wrongly denied), and the flat arm
  *did* produce fire-on-everything models (which that entry had also wrongly retracted). Both
  statements were mine, made hours apart. The rows now carry `md5_prefix` and, where nobody
  reproduced them in one clean pass, `"unverified": true`.
- **2026-09-25 — measurements left this file.** They had accumulated in prose, where a table could
  be edited without anyone noticing the scoring pass behind it change. `scorecards.jsonl` makes
  each row's provenance part of the row.
- **2026-09-24, this session:** two claims made verbally before this directory was updated were
  wrong (see above). Every mww model inside a run dir is `stream_state_internal_quant.tflite` and
  every manifest is `hey_seeree.json`, so a mistyped path silently scores a different run.
  `tools/score_margins.py` now prints the resolved artifact plus its md5, refuses any artifact
  resolving inside `data/corpus/`, exits if a manifest names a `.tflite` that is not there, labels
  an unreachable-budget reading `BEST-AVAILABLE … NOT AT BUDGET` instead of `matched-FA`, and keys
  its CSV `model` column on the artifact.
- **`train/ledger.py` C2:** the duplicate-run key was `(config-hash, seed)`, which is blind to the
  corpus. The four flat/balanced mww pairs at seeds 950–953 collapsed into four "samples" instead
  of eight, and the seed-953 pair printed a DETERMINISM REGRESSION about a difference that was the
  variable under test. `corpus_id` is in the key now; the byte-pin was re-captured at 45 records.

## What is not cleared

1. **Preflight, both targets, on a live mic.** Held-out TTS numbers are not a substitute and
   neither staged file has ever heard a real room.
   `cd preflight && uv run test_model.py --model ../deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.json`
   (the Pi model needs its `.tflite` plus the 0.58–0.60 threshold).
2. **Detection is far below the gate.** The staged Pi row is nowhere near the 97% both gates were
   set to (`f2865bc` moved them 98% → 97%). Nothing here is ship-eligible for a product; it is the
   best measured candidate per target.
3. **`-dirty` in the tag — resolved, and what it never meant.** `55e182a-dirty` was built from HEAD
   `55e182a`, which already contained the balance lever and both trainer wirings; the dirty files
   were `tools/score_margins.py`, `train/ledger.py`, `scripts/sweep.py`, their tests and the
   sentinel guard in `train/corpus/real.py`. None touch the training path, so `-dirty` never made
   the model unreproducible — it made the tag unresolvable without `git stash list`. Those files
   are now committed (`927c21b`, `e72845a`, `f5a3beb`, `320f93e`), so the next run's tag is clean.
   The reproducibility claim itself is still unverified: nobody has re-run `--seed 8001` at this
   HEAD to confirm it reproduces corpus `c9897b91` and a byte-identical `.onnx`.
4. **The eight 10x ESP32 rows are `unverified`.** They came from passes whose file provenance was
   caught being wrong once already; `scripts/score-esp32-arms.sh` reproduces all eight in one pass
   and clears the flag. Until then no ESP32 ship call rests on them.
5. **n=2 seeds per oww arm.** Both balanced seeds moved the same direction and the staged file is
   the better of the two, but two draws do not rank arms.
6. **Latency on the Pi is unmeasured.** The sweep's scorecard for this model reads median 107 ms /
   p90 228 ms through the onnx path; the `.tflite` on a Pi 4 is a different runtime and nobody has
   timed it.
7. **jen's real holdout is 10 clips.** One clip is the difference between most of the jen moves
   quoted in rows; what carries the oww jen claim is that both balanced seeds and both runtimes
   agree.
8. **ryan moved the wrong way across the engine family** — see the per-speaker fields on the
   `mww-*` and `oww-*` rows. His row share also fell as a side effect of lifting the adults. Per
   CLAUDE.md, no hyperparameter has ever moved him; more ryan recordings is the only lever the
   measurements have pointed at.

## Rules for this directory

- **One candidate per target.** Replacing a staged file requires a measured win at matched false
  accepts, **per speaker**, on the same 298-clip adversarial set and in one scoring pass — never at
  a fixed threshold, never pooled, never across days.
- **New measurement → new `scorecards.jsonl` row**, with `artifact`, `md5_prefix`, `threshold`,
  `positives`, `negatives` and `budget` filled in. A row without them is not checkable, and this
  directory's history says an unchecked row gets cited as fact within a day.
- Never compare a pre-2026-09-21 `/230` or `/32` adversarial figure to a `/298` one. The
  incumbent's 2026-09-08 gate claims are quoted with the old denominator; only the 2026-09-22 and
  2026-09-24 re-scores are on the current set.
- The threshold in each row is measured, from the curve in `logs/scorecurves/`, not defaulted.
