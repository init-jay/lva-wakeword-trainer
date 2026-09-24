# deploy/ — `hey seeree`: what is staged, and what is not cleared

The per-word record for the staging directory. [`README.md`](README.md) keeps the rules
and the traps; the measured numbers themselves are rows of
[`scorecards.jsonl`](scorecards.jsonl), and this file is the narrative that does not fit
in a row: what is on the satellites, what was retracted, and what no measurement here
has cleared.

## What is staged

```
esp32-mww/hey_seeree_ecbf160-dirty-da01854d.{tflite,json}   unchanged since 2026-09-08
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.tflite   staged 2026-09-24
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.config.json
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.manifest.json   LVA discovery manifest (added 2026-09-25)
```

Their measured operating points are the rows `oww-hf37e3df-584` and `mww-da01854d-009` -
one `staged` row per target, at the cutoff each satellite actually ships. The ESP32 one is
above the FA budget, and `mww-da01854d-150` is the `reference` reading of the same artifact
squeezed to budget: quote that one for a comparison, but know that the satellite is not
running it.

`*.manifest.json` is the LVA discovery manifest (model ID = its filename stem,
`probability_cutoff` 0.60), not the trainer's resolved config (`*.config.json`); stage
both the `.tflite` and the `.manifest.json`.

**Run the Raspberry Pi model at threshold 0.58–0.60, not at 0.5.** That band is flat in
detection and it is the tight edge of the measured curve; the numbers are in row
`oww-hf37e3df-584`, whose own threshold field is 0.584. 0.5 is not this model's operating
point and nothing about it is calibrated there.

The shipped manifest says **0.60**, and no row measures 0.60 — the row carries
`shipped_cutoff: 0.6` and says so in its `note`. Per `README.md` that is an open item, not a
resolved state: closing it means measuring at 0.60, or moving the manifest to 0.584
deliberately and re-deploying. It is *not* closed by editing the manifest to match a
measurement already taken, and the satellite is already deployed against 0.60. The ESP32 manifest's `probability_cutoff: 0.09` is unchanged and
remains *its* measured point — see `mww-da01854d-009` for why that point is above budget.

The Pi candidate is preferred over the model that was firing in the house not because it
wins at the loose end — at ≤ 5 fires it is a one-clip tie — but because its curve holds as
the budget tightens and the house model's does not. Both ends are rows:
`oww-hf37e3df-584`, `oww-house-783` — and `oww-house-670`, which is `unverified` and quoted
here only as the loose end of the same curve (see the corrections below).

**Open item, naming.** `wake_word.py:40` sets the model ID to `Path.stem`, which strips one
suffix: the staged file's ID is `hey_seeree_55e182a-dirty-c9897b91-hf37e3df.manifest`, where
every model LVA bundles is `<id>.json` beside `<id>.tflite` (`hey_jarvis.json`, `alexa.json`)
and `WAKE_MODEL` is matched against the stem. Renaming the manifest to drop `.manifest` would
also clean the ID and the HA label, but the satellite is already deployed against the current
name and `WAKE_MODEL` would have to move in the same step, so it is left alone and recorded
here. Verified against OHF-Voice/linux-voice-assistant@HEAD 2026-09-24: `:34` globs `*.json`
only — a `.json` left in the model dir by anything else becomes a selectable model; `:52-53`
for openWakeWord the `.tflite` is resolved relative to the manifest's own directory, so the
pair must stay together; `:70` reads
`model_config["openWakeWord"]["probability_cutoff"]` and defaults to **0.7** when the block is
absent — looser than any budget in this directory, which is why the manifest ships with the
weights instead of leaving the cutoff to a config default; `models.py:38` fixes the type
string as `"openWakeWord"`.

On the Pi they land in the dir `WAKE_WORD_DIR` points at — on the current install that is the
`lva_wakeword_custom` volume (`/app/wakewords/custom`), host path
`/var/lib/docker/volumes/lva_wakeword_custom/_data/` — and `WAKE_MODEL` is set in the compose
`.env` to the manifest's stem. The manifest's `wake_word` field is a *display label*: HA
builds its wake-word dropdown as a dict keyed on that phrase (`esphome/select.py`), so two
models sharing a label collapse to one option in the UI — the staged v0.2 manifest therefore
reads `"Hey Seeree v0.2"` to stay selectable beside the house `hey_seeree_v0.1`. The
`WAKE_MODEL` env is only the *default*; the active model comes from HA's preference and
overrides it, and the Wake Word 1 sensitivity (0.700 persisted) follows the slot, not the
model — re-set it to 0.60 after switching.

## Reading the rows

Every row is read at the budget or tighter, on the real 51-clip holdout (jay 35, jen 10,
ryan 6), against the 298-clip adversarial set (extend 148 + hey_other 150), in one scoring
pass.

The staged artifacts' full digests, so a reviewer can confirm the bytes a row describes
without the gitignored `.tflite` being in the repo:

| artifact | md5 |
|---|---|
| `deploy/rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.tflite` | `5cadbc531b0fc272b7ca7859157584e5` |
| `deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.tflite` | `248a15d88e601ece87b33070ec3836cc` |

The manifests carry no digest of their own and are not edited after deployment, so the row's
`md5_prefix` (and this table) is the digest of record; agreement between a row and its bytes
is checked by a human with `md5 -q`, per `README.md`. Every row now names the curve it was
read from in its `curve` field — eleven of them did not, and were resolved by tag filename
and by matching the CSV's `#md5` model key to the row's `md5_prefix`. Six of those CSVs
(`mww-incumbent-da01854d`, `mww-42f8982-h5f6f353`, `oww-bal-s8000-onnx`, `oww-flat-tflite`,
`oww-room-sep7`) predate the digest-in-the-model-key convention, so for them the link is
filename-only. `logs/scorecurves/mww-{flat,bal}-s95{0,1,2,3}.csv` are clip-identical
duplicates of the tag-named curves and back no row of their own. All saved curves were checked to sit on those same clips, so a fixed fire count is
comparable across those files.

- **Never compare a pre-2026-09-21 `/230` or `/32` adversarial figure to a `/298` one.** The
  incumbent's 2026-09-08 gate claims are quoted with the old denominator; only the
  2026-09-22 and 2026-09-24 re-scores are on the current set.
- **Cross-day totals are not comparable**, and no row pools across speakers.
- The curve CSVs a row cites live in `logs/scorecurves/`, which is gitignored, so a reviewer
  cannot open them. Regenerate any row with:

  ```bash
  eval/.venv/bin/python tools/score_margins.py --model <path> --adv-fa-budget 5 \
    --csv logs/scorecurves/<name>.csv
  ```

  adding `--sliding-window-size 5` for a bare `.tflite`, since a cutoff means nothing without
  the window that produced it.

Reproducing the ESP32 rows means scoring each run dir's own weights, not the shared corpus
copy — see README's "Which bytes a microWakeWord row belongs to". The eight 10x rows carry
`"unverified": true` until one pass over all eight run dirs clears them.

## Corrections to what is written above

- **2026-09-26 — `mww-da01854d-009` recorded the wrong threshold's reading.** It claimed
  ryan 5/6 and pooled 38/51 at cutoff 0.09; the curve reads ryan **4/6** and pooled **37/51**
  at 0.09. Ryan's fifth clip peaks at 0.0875, which fires at 0.0875 and not at the shipped
  0.09 — the row had the 0.0875 reading under a 0.09 label. `SPEED.md`'s "Current models"
  section already had it right (37/51, 73%, ryan 4/6), which is what caught it. Corrected in
  place with the erratum in the row's `note`, because the file has not landed on `main` yet
  and a knowingly wrong row gets cited as fact; once it has, corrections are new rows.
- **2026-09-26 — `oww-house-670` is not reproducible and is marked `unverified`.** Its
  numbers (fires 2, jay 22/35, jen 5/10, pooled 28/51 at 0.67) are not what
  `logs/scorecurves/oww-room-sep7.csv` reads at 0.67 (fires 8, jay 27/35, jen 9/10, pooled
  38/51), nor at any other threshold in it, and no other curve on this machine produces them.
  The sibling row `oww-house-783` matches that same curve exactly, so the curve is sound and
  this row's provenance is not. Nothing here rests on it — the house model's standing in the
  comparison comes from `oww-house-783` — but it stays in the file, flagged, rather than being
  quietly deleted.
- **2026-09-26 — two `staged` rows for one ESP32 artifact.** `mww-da01854d-150` and
  `mww-da01854d-009` both read `staged`, so filtering on it did not answer "what is on the
  satellite". `-009` keeps `staged` (its threshold is the shipped `probability_cutoff`);
  `-150` becomes `reference`. `README.md` now states the rule.
- **2026-09-25 — retracts the 2026-09-24 entry below.** Re-scoring the ESP32 models from
  their own run dirs showed the flat arm's best-seed row had described a model that does not
  exist: those bytes fire on every adversarial clip at any threshold. The retraction ran both
  ways — balancing *is* a win on this engine (the direction the 2026-09-24 entry wrongly
  denied), and the flat arm *did* produce fire-on-everything models (which that entry had also
  wrongly retracted). Both statements were made hours apart. The rows now carry `md5_prefix`
  and, where nobody reproduced them in one clean pass, `"unverified": true`.
- **2026-09-25 — measurements left `README.md`.** They had accumulated in prose, where a table
  could be edited without anyone noticing the scoring pass behind it change. `scorecards.jsonl`
  makes each row's provenance part of the row, and this file keeps the per-word narrative.
- **2026-09-24, that session:** two claims made verbally before this directory was updated were
  wrong (see above). Every mww model inside a run dir is `stream_state_internal_quant.tflite`
  and every manifest is `hey_seeree.json`, so a mistyped path silently scores a different run.
  `tools/score_margins.py` now prints the resolved artifact plus its md5, refuses any artifact
  resolving inside `data/corpus/`, exits if a manifest names a `.tflite` that is not there,
  labels an unreachable-budget reading `BEST-AVAILABLE … NOT AT BUDGET` instead of `matched-FA`,
  and keys its CSV `model` column on the artifact.
- **`train/ledger.py`:** the duplicate-run key was `(config-hash, seed)`, which is blind to the
  corpus. The four flat/balanced mww pairs at seeds 950–953 collapsed into four "samples"
  instead of eight, and the seed-953 pair printed a DETERMINISM REGRESSION about a difference
  that was the variable under test. `corpus_id` is in the key now.

## What is not cleared

1. **Preflight, both targets, on a live mic.** Held-out TTS numbers are not a substitute and
   neither staged file has ever heard a real room.
   `cd preflight && uv run test_model.py --model ../deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.json`
   (the Pi model needs its `.tflite` plus the 0.58–0.60 threshold).
2. **The Pi's shipped cutoff has no measured row.** The manifest says 0.60; the staged row
   measures 0.584 and calls 0.58–0.60 flat. Either measure 0.60 or move the manifest to a
   measured point — see above. Until then the satellite runs at a cutoff this directory has
   not read.
3. **Detection is far below the gate.** The staged Pi row is nowhere near the 97% both gates
   were set to. Nothing here is ship-eligible for a product; it is the best measured candidate
   per target.
4. **`-dirty` in the tag — resolved, and what it never meant.** `55e182a-dirty` was built from
   HEAD `55e182a`, which already contained the balance lever and both trainer wirings; the
   dirty files were `tools/score_margins.py`, `train/ledger.py`, `scripts/sweep.py`, their tests
   and the sentinel guard in `train/corpus/real.py`. None touch the training path, so `-dirty`
   never made the model unreproducible — it made the tag unresolvable without `git stash list`.
   Those files are now committed, so the next run's tag is clean. The reproducibility claim
   itself is still unverified: nobody has re-run `--seed 8001` at this HEAD to confirm it
   reproduces corpus `c9897b91` and a byte-identical `.onnx`.
5. **The eight 10x ESP32 rows are `unverified`.** They came from passes whose file provenance
   was caught being wrong once already. Until one clean pass reproduces all eight, no ESP32
   ship call rests on them.
6. **n=2 seeds per oww arm.** Both balanced seeds moved the same direction and the staged file
   is the better of the two, but two draws do not rank arms.
7. **Latency on the Pi is unmeasured.** The sweep's scorecard for this model reads median
   107 ms / p90 228 ms through the onnx path; the `.tflite` on a Pi 4 is a different runtime
   and nobody has timed it.
8. **jen's real holdout is 10 clips.** One clip is the difference between most of the jen moves
   quoted in rows; what carries the oww jen claim is that both balanced seeds and both runtimes
   agree.
9. **ryan moved the wrong way across the engine family** — see the per-speaker fields on the
   `mww-*` and `oww-*` rows. His row share also fell as a side effect of lifting the adults.
   Per CLAUDE.md, no hyperparameter has ever moved him; more ryan recordings is the only lever
   the measurements have pointed at.
