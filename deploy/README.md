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

## What is staged

```
esp32-mww/hey_seeree_ecbf160-dirty-da01854d.{tflite,json}   unchanged since 2026-09-08
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.tflite   staged 2026-09-24
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.config.json
rpi-oww/hey_seeree_55e182a-dirty-c9897b91-hf37e3df.manifest.json   LVA discovery manifest (added 2026-09-25)
```

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

**Run the Raspberry Pi model at threshold 0.58–0.60, not at 0.5.** At 0.60 it measures
**2 adversarial false accepts of 298 (0.67%)** with jay 26/35, jen 9/10, ryan 2/6; 0.5
is not this model's operating point and nothing about it is calibrated there. The
ESP32 manifest's `probability_cutoff: 0.09` is unchanged and remains *its* measured
point (8/298 = 2.7% FA — above the 2% budget, see below).

Independently re-scored after staging (same md5 `5cadbc53`) with the same result, and the
curve is **flat from 0.35 to 0.60** — pooled 73%, jay 26/35, jen 9/10 at every grid point in
that band — so tightening from the 1.7% budget to 0.67% costs no detection at all. The
endpoints are what pay: 0.65 → 35/51, 0.70 → 34/51 at FA 1, 0.75 → 32/51. The threshold was
chosen at the flat region's tight edge for that reason, not because 0.58 is special.

## FA budget

The operating constraint is **FA < 2% of the 298-clip adversarial set = at most 5
fires**. Every number below is read at that budget or tighter, on the real 51-clip
holdout (jay 35, jen 10, ryan 6), in **one scoring pass over one set of clips**
(`tools/score_margins.py`, peaks per clip; curves in `logs/scorecurves/*.csv` — that directory
is gitignored, so a reviewer cannot open them: regenerate any row with
`eval/.venv/bin/python tools/score_margins.py --model <path> --adv-fa-budget 5 --csv
logs/scorecurves/<name>.csv`, adding `--sliding-window-size 5` for a bare `.tflite`, since a
cutoff means nothing without the window that produced it). All 13 saved curves were checked
to sit on the same 298 adversarial clips (extend 148 + hey_other 150) and the same 51
positives, so a fixed fire count is comparable across those files.
Cross-day totals are not comparable and none of these rows are pooled across speakers.

### Raspberry Pi / openWakeWord

| model | threshold | adv FA | jay | jen | ryan | adults | pooled |
|---|---|---|---|---|---|---|---|
| `c9897b91-hf37e3df` **staged** | 0.584 | 2/298 (0.67%) | 26/35 | 9/10 | 2/6 | **35/45 (77.8%)** | **37/51** |
| `c9897b91-hc34dc7e` (same arm, other seed, onnx) | 0.487 | 4/298 | 25/35 | 9/10 | 1/6 | 34/45 | 35/51 |
| flat 10x `c306c00f-h509d9ea` (tflite, md5 `6c99f78a`) | 0.35 | 4/298 (1.3%) | 23/35 | 8/10 | 1/6 | 31/45 | 32/51 |
| **what was firing in the house** (untagged Sep-7 `hey_seeree.tflite`, md5 `5014ac67f5b1ede7d518c13fb46a3cc8`, 5k steps) | 0.783 | 5/298 | 26/35 | 8/10 | 2/6 | 34/45 | 36/51 |

At the loose end of the budget the staged candidate is a **one-clip tie** with the
house model (35 vs 34 adults). The reason to prefer it is the tight end, where the
house model's curve falls apart:

| FA budget | staged candidate adults / pooled | house model adults / pooled |
|---|---|---|
| ≤ 5 fires (1.7%) | 35 / 37 | 34 / 36 |
| ≤ 3 fires (1.0%) | 35 / 37 | 32 / 34 |
| ≤ 2 fires (0.67%) | 35 / 37 | **27 / 28** |

That is the argument for staging it: under 1% FA it is worth +8 adult points and
+9 pooled detections. Squeeze it to a single fire — 0.34%, half the budget — and it
still reads 33/45 adults (jay 24, jen 9, ryan 1); the house model is at 27/45 by two
fires.

### ESP32 / microWakeWord — the balance lever did help; three of eight models cannot be calibrated

| md5 | model (corpus = arm, seed) | threshold | adv FA | jay | jen | ryan | adults |
|---|---|---|---|---|---|---|---|
| `248a15d8` | `ecbf160-da01854d` **staged** | 0.15 | 4/298 (1.3%) | 30/35 | 1/10 | 4/6 | 31/45 |
| `ee79c1fd` | `42f8982-ce1500f5-h5f6f353` (2026-09-22) | 0.65 | 5/298 (1.7%) | 34/35 | 4/10 | 0/6 | 38/45 |
| `54570f4e` | flat 10x seed 950 `c574e978-hd6b366e` | 0.60 | 5/298 | 19/35 | **0/10** | 0/6 | 19/45 |
| `cbf0ec81` | flat 10x seed 951 `c574e978-hbd0e5de` | 0.95 | 4/298 | 27/35 | **0/10** | 0/6 | 27/45 |
| `a293a2ce` | flat 10x seed 952 `c574e978-h3be78bf` | — | 298/298 at 0.05 | — | — | — | no budget point |
| `16296dbb` | flat 10x seed 953 `c574e978-hd5948fa` | — | 298/298 at 0.05 | — | — | — | no budget point |
| `1072e809` | balanced seed 950 `c5a47eeb-hd6b366e` | 0.90 | 5/298 | 29/35 | 6/10 | 0/6 | 35/45 |
| `e74e471c` | balanced seed 951 `c5a47eeb-hbd0e5de` | 0.85 | 5/298 | **35/35** | 6/10 | 0/6 | **41/45** |
| `56e7dca0` | balanced seed 952 `c5a47eeb-h3be78bf` | 0.75 | 2/298 (0.7%) | 19/35 | 2/10 | 0/6 | 21/45 |
| `a7eb5614` | balanced seed 953 `c5a47eeb-hd5948fa` | — | 298/298 at 0.05 | — | — | — | no budget point |

Every row is scored from that run dir's **own** `stream_state_internal_quant.tflite` — the md5 in
the first column is the row's identity, see *Which bytes an ESP32 row belongs to* below. Re-derived
2026-09-25; the flat rows that stood here before came from corpus-dir copies and are gone.

The incumbent appears twice with different per-speaker numbers and that is not an
error: at **its own manifest cutoff 0.09** it sits at FA 8/298 (2.7% — above the 2%
budget) reading jay 31, jen 2, ryan 5, which is the "weakest speaker jen 2/10 = 20%"
figure in the older notes; squeezed to the budget (0.148, 4 fires) jen is 1/10 and ryan
4/6. Recomputed from `logs/scorecurves/mww-incumbent-da01854d.csv` rather than from
memory, so both rows are reproducible from the same curve file.

### Which bytes an ESP32 row belongs to

Every ESP32 row above is the *content* of a run dir's weights file, and the tag alone cannot vouch
for it. Two reasons, both learned the hard way this session.

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
per-arm bytes wearing per-seed names. That is the only mechanism that fits the flat rows that stood
here on 2026-09-24: none of them reproduce from a run dir, seed 953's own bytes fire on 298/298
adversarial clips at every threshold, and the file behind its "41/45" row (`d9cfb496`) has no
run-dir owner. The exact provenance of each stale row cannot be reconstructed — `prepare_corpus`
rmtree's the corpus tree at the start of every run, and the next sweep had already replaced those
copies by the time this was checked. What survives is the rule: score the run dir.

So: score the run dir, and treat the md5 `score_margins.py` prints as the row's identifier.

```bash
eval/.venv/bin/python tools/score_margins.py \
  --model output/hey_seeree/mww/<tag>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --sliding-window-size 5 --adv-fa-budget 5
```

openWakeWord does not have this problem: its output is one tagged `<tag>.onnx`/`<tag>.tflite` per
run. Anything staged into `deploy/esp32-mww/` must be md5-matched against its run-dir bytes before
a row beside it is trusted.

Adults per usable model at FA ≤ 5, all four seeds of each arm:

- flat: **19, 27**, and two seeds with no budget point → mean 23, best 27
- balanced (jen lifted to 18x): **35, 41, 21**, one seed with no budget point → mean 32.3, best 41

**Balancing helped this engine, and the opposite claim written here on 2026-09-24 was an
attribution error.** Per seed the arms pair up 950: 19→35, 951: 27→41, 952: no-budget-point→21,
953: no-budget-point→no-budget-point. jen is the stable part: **0/10 in both usable flat seeds**,
6/10 in the two best balanced seeds, 2/10 in the third — on 10 real clips that is the largest
per-speaker move any lever made this session. ryan is 0/6 in every 10x-family model, usable or not,
balanced or flat.

Three of the eight cannot be calibrated at all: `train.mww.manifest` refuses to write a manifest
(no cutoff meets `--max-faph 0.2` while detecting anything) and the tightest grid threshold still
fires on all 298 adversarial clips. **Two of those three are flat seeds** — the 2026-09-24 note
named only the balanced one. The manifest stage is non-fatal by design
(`run-mww-training-applesilicon.sh`), so the sweep still files the row.

The balanced best (`e74e471c`, 41/45 adults, jay 35/35, jen 6/10) beats the incumbent on both
adults and jen but loses ryan outright (0/6 vs 4/6). That trade is a judgement call, not a
measurement, and it is the user's to make. Re-baselining and staging it is a
`tools/score_margins.py` pass plus a preflight, not a code change.

Neither challenger replaces the incumbent on the stated bar: both beat it on the
adults (41 vs 31) and lose ryan outright (0–1/6 vs 4/6). That trade is a judgement
call, not a measurement, and it is the user's to make. Re-baselining and staging one
of them is a `tools/score_margins.py` pass plus a preflight, not a code change.

## Corrections to what is written above

- **2026-09-25 — this one retracts the 2026-09-24 entry below it, in both
  directions.** Re-scoring every ESP32 model from its own run dir (md5s in the ESP32 table)
  shows the flat arm's "best seed 41/45" was not a reading of a model that exists: seed 953's own
  bytes fire on 298/298 adversarial clips at every threshold. So (a) "balancing is not a win on
  mww" was wrong — it is a win, 19/27 usable adults flat vs 35/41/21 balanced, and jen goes
  0/10 → 6/10; and (b) "the flat arm produced models that fire on all 298 clips" was right,
  which the 2026-09-24 entry wrongly retracted. Both statements were mine, made hours apart,
  and the correction is the table above plus *Which bytes an ESP32 row belongs to*.
  `tools/score_margins.py` now prints the resolved artifact plus its md5, exits if a
  manifest names a `.tflite` that is not there, labels an unreachable-budget reading
  `BEST-AVAILABLE … NOT AT BUDGET` instead of `matched-FA`, and keys its CSV `model`
  column on the artifact. Every mww row written before that landed is suspect until re-scored
  from the run dir — several of the ones in this file's history were, including two I
  "corrected" and got wrong again in the conservative direction.
- **2026-09-24, this session:** two claims made verbally before this directory was
  updated were wrong and are superseded by the tables here. (a) "balancing is a big
  win on mww" — superseded by 2026-09-25: it is a win. (b) "the flat arm produced models that
  fire on all 298 clips" — also superseded: true, for seeds 952 and 953. Every mww model
  inside a run dir is `stream_state_internal_quant.tflite` and every manifest is
  `hey_seeree.json`, so a mistyped path silently scores a different run.
  `tools/score_margins.py` now prints the resolved artifact plus its md5, exits if a
  manifest names a `.tflite` that is not there, labels an unreachable-budget reading
  `BEST-AVAILABLE … NOT AT BUDGET` instead of `matched-FA`, and keys its CSV `model`
  column on the artifact. Any mww "degenerate model" claim made before this change is
  suspect until re-scored.
- **`train/ledger.py` C2:** the duplicate-run key was `(config-hash, seed)`, which is
  blind to the corpus. The four flat/balanced mww pairs at seeds 950–953 collapsed
  into four "samples" instead of eight, and the seed-953 pair printed a
  DETERMINISM REGRESSION about a difference that was the variable under test.
  `corpus_id` is in the key now; the byte-pin was re-captured at 45 records.

## What is not cleared

1. **Preflight, both targets, on a live mic.** Held-out TTS numbers are not a
   substitute and neither staged file has ever heard a real room.
   `cd preflight && uv run test_model.py --model ../deploy/esp32-mww/hey_seeree_ecbf160-dirty-da01854d.json`
   (the Pi model needs its `.tflite` plus the 0.58–0.60 threshold).
2. **Detection is far below the gate.** 77.8% adults / 73% pooled vs the 97% `f2865bc`
   set both gates to (it moved them 98% -> 97%). Nothing here is ship-eligible for a
   product; it is the best measured
   candidate per target.
3. **`-dirty` in the tag — resolved, and what it never meant.** `55e182a-dirty` was built
   from HEAD `55e182a`, which already contained the balance lever and both trainer
   wirings; the dirty files were `tools/score_margins.py`, `train/ledger.py`,
   `scripts/sweep.py`, their tests and the sentinel guard in `train/corpus/real.py`.
   None touch the training path, so `-dirty` never made the model unreproducible — it
   made the tag unresolvable without `git stash list`. Those files are now committed
   (`927c21b`, `e72845a`, `f5a3beb`, `320f93e`), so the next run's tag is clean. The
   reproducibility claim itself is still unverified: nobody has re-run `--seed 8001` at
   this HEAD to confirm it reproduces corpus `c9897b91` and a byte-identical `.onnx`.
4. **n=2 seeds per oww arm.** Both seeds of the balanced arm moved the same direction
   and the staged file is the better of the two, but two draws do not rank arms —
   the mww table above is what a four-draw comparison of a similar-sized effect looks
   like.
5. **Latency on the Pi is unmeasured.** The sweep's scorecard for this model reads
   median 107 ms / p90 228 ms through the onnx path; the `.tflite` on a Pi 4 is a
   different runtime and nobody has timed it.
6. **jen's real holdout is 10 clips.** 8 → 9 is one clip; what carries the jen claim is
   that both balanced oww seeds and both runtimes read 9/10.
7. **ryan moved the wrong way across the engine family** (4/6 → 0–1/6 on mww under 10x
   copies, in both arms; 2/6 on oww, same as the house model). His row share also fell
   24% → 19.2% as a side effect of lifting the adults. Per CLAUDE.md, no
   hyperparameter has ever moved him; more ryan recordings is the only lever the
   measurements have pointed at.

## Rules for this directory

- **One candidate per target.** Replacing a staged file requires a measured win at
  matched false accepts, **per speaker**, on the same 298-clip adversarial set and in
  one scoring pass — never at a fixed threshold, never pooled, never across days.
- Never compare a pre-2026-09-21 `/230` or `/32` adversarial figure to a `/298` one.
  The incumbent's 2026-09-08 gate claims are quoted with the old denominator; only the
  2026-09-22 and 2026-09-24 re-scores are on the current set.
- The `probability_cutoff` / threshold above each file is measured, from the curve in
  `logs/scorecurves/`, not defaulted.
