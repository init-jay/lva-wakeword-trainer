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

`*.manifest.json` is the LVA discovery manifest (model ID = its filename stem, `probability_cutoff` 0.60),
not the trainer's resolved config (`*.config.json`); stage both the `.tflite` and the `.manifest.json`.

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

## FA budget

The operating constraint is **FA < 2% of the 298-clip adversarial set = at most 5
fires**. Every number below is read at that budget or tighter, on the real 51-clip
holdout (jay 35, jen 10, ryan 6), in **one scoring pass over one set of clips**
(`tools/score_margins.py`, peaks per clip; curves in `logs/scorecurves/*.csv`).
Cross-day totals are not comparable and none of these rows are pooled across speakers.

### Raspberry Pi / openWakeWord

| model | threshold | adv FA | jay | jen | ryan | adults | pooled |
|---|---|---|---|---|---|---|---|
| `c9897b91-hf37e3df` **staged** | 0.584 | 2/298 (0.67%) | 26/35 | 9/10 | 2/6 | **35/45 (77.8%)** | **37/51** |
| `c9897b91-hc34dc7e` (same arm, other seed, onnx) | 0.487 | 4/298 | 25/35 | 9/10 | 1/6 | 34/45 | 35/51 |
| flat 10x `c306c00f-h509d9ea` | 0.339 | 4/298 | 24/35 | 8/10 | 1/6 | 32/45 | 33/51 |
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

### ESP32 / microWakeWord — incumbent stands, and the balance lever did not help here

| model | threshold | adv FA | jay | jen | ryan | adults |
|---|---|---|---|---|---|---|
| `ecbf160-da01854d` **staged** | 0.148 | 4/298 | 30/35 | 1/10 | 4/6 | 31/45 |
| `42f8982-h5f6f353` (2026-09-22) | 0.748 | 3/298 | 34/35 | 4/10 | 0/6 | 38/45 |
| flat 10x, best seed `c574e978-hd5948fa` | 0.970 | 3/298 | 34/35 | 7/10 | 1/6 | **41/45** |
| balanced `jay,jen`, best seed `c5a47eeb-hbd0e5de` | 0.870 | 5/298 | 35/35 | 6/10 | 0/6 | **41/45** |

Adults per speaker at FA ≤ 5, all four seeds of each arm:

- flat: 19, 33, 32, 41 → mean 31.3
- balanced (jen lifted to 18x): 36, 41, 23, **unusable** → mean 25

**No evidence that balancing helps this engine.** The seed-to-seed spread inside each
arm (22 points) is larger than the difference between the arms, the best single model
came from the *flat* arm, and the balanced arm produced one model that cannot be
calibrated at all: `train.mww.manifest` refused to write a manifest for
`c5a47eeb-hd5948fa` because no cutoff meets `--max-faph 0.2` while detecting anything,
and its per-clip peaks sit in 0.65–0.996 across positives *and* negatives — no
separation, not a dead model. The manifest stage is non-fatal by design
(`run-mww-training-applesilicon.sh`), so the sweep still filed the row.

Neither challenger replaces the incumbent on the stated bar: both beat it on the
adults (41 vs 31) and lose ryan outright (0–1/6 vs 4/6). That trade is a judgement
call, not a measurement, and it is the user's to make — say the word and I will
re-baseline and stage one of them.

## Corrections to what is written above

- **2026-09-24, this session:** two claims made verbally before this directory was
  updated were wrong and are superseded by the tables here. (a) "balancing is a big
  win on mww" — it is not; the flat arm's best seed matched the balanced arm's best
  and the flat arm's mean is higher. (b) "the flat arm produced models that fire on
  all 298 clips" — that reading came from scoring the wrong artifact. Every mww model
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
2. **Detection is far below the gate.** 77.8% adults / 73% pooled vs the 98% the eval
   gates want. Nothing here is ship-eligible for a product; it is the best measured
   candidate per target.
3. **`-dirty` in the tag, and what it does and does not mean.** `55e182a-dirty` was built
   from HEAD `55e182a` — which already contains the balance lever and both trainer
   wirings — with a dirty tree of `tools/score_margins.py`, `train/ledger.py`,
   `scripts/sweep.py`, their tests and a 5-line sentinel guard in `train/corpus/real.py`.
   None of those touch the training path, so `--seed 8001` at this HEAD should still
   reproduce corpus `c9897b91` and a byte-identical `.onnx` (CLAUDE.md's determinism
   bar). Nobody has re-run it to confirm that; the dirty files should be committed so
   the next tag is clean.
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
