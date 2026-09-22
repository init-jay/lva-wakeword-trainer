# improvement.md

Written 2026-09-20 against `separate-tts-service` @ `b38715c`, for the two stated aims:

1. **Apple Silicon as the first-class training route**, not the measured-and-documented
   alternative to Docker that it currently is.
2. **Sub-hour passes for both targets**, so hyperparameter tuning becomes a loop
   somebody actually runs rather than a day's work per question.

Everything below was checked against the working tree, not inferred from the docs.
Where a claim is a measurement someone still has to take, it says so.

---

## The diagnosis in one paragraph

**Wall time is not the binding constraint any more; measurement noise is.** A Mac host
run is already 14m14s for mww and ~35m for oww (SPEED.md), and — verified below — an
oww re-run with `--skip-corpus` reuses the cached feature arrays too, so a
training-only oww iteration is already ~6 minutes. Both targets are already inside the
hour. What stops the tuning loop is on the other side: **nothing is seeded**, the run
tag **collides across hyperparameter configs** so two sweep points cannot even be
filed, the knobs worth sweeping (`lr`, `batch_n_per_class`, `negative_class_weight`,
`learning_rates`, the per-set sampling weights) are **not reachable from any CLI**, the
oww trainer **silently multiplies `max_negative_weight` by up to 4x behind your back**,
and the scorecard those sweeps would be judged on has **n=6 for one speaker and 32
adversarial negatives in total**. This repo has already measured 10 points of
run-to-run variance at an identical config (SPEED.md, CLAUDE.md). A sweep run on top
of that measures the seed, not the hyperparameter.

So the work splits: make a pass cheap and *repeatable* (P0), make the number it
produces trustworthy (P1), then take the remaining wall time (P2).

---

## Verified facts the plan rests on

| # | Fact | Evidence |
|---|---|---|
| 1 | `--skip-corpus` already skips augmentation *and* feature computation — the help text says otherwise | `openwakeword/openwakeword/train.py:778` gates the whole augment block on `positive_features_train.npy` not existing; `setup_training_dirs` only rmtree's when not skipping (`train/oww/train.py:349`). The four `.npy` are sitting in `data/corpus/hey_seeree/oww/` now. Help text at `train/oww/train.py:568` claims "still re-runs augmentation" |
| 2 | `patches/honour-augmentation-rounds.py` is **not idempotent** and has been applied 3x | `openwakeword/openwakeword/train.py:810` reads `*config["augmentation_rounds"]*config["augmentation_rounds"]*config["augmentation_rounds"]`. The regex `n_total=len\(os\.listdir\((\w+)\)\)` re-matches its own output. `scripts/setup-applesilicon-trainer.sh` advertises idempotence |
| 3 | Not data corruption — just 9x the disk and a pointless trim pass | `positive_features_train.npy` is 205,701,248 B = 33,480 rows; `positive_train/` holds 11,162 wavs × 3 rounds = 33,486. `trim_mmap` (`data.py:856`) cut the over-allocation correctly |
| 4 | Nothing in either training path is seeded | No `manual_seed`/`np.random.seed` anywhere in `train/`, `openwakeword/train.py`, or `microwakeword/`. `augment_clips` *takes* a `seed` (`openwakeword/data.py:313,369-372`) and this repo never passes one |
| 5 | The oww run tag cannot distinguish two hyperparameter configs | `train/provenance.py:104-110` hashes commit + `data/recordings/samples` + `data/corpus/<wake>`. Two sweep points at the same dirty commit and same corpus get the **same tag**; `train/mww/train.py` then refuses to train into the existing directory |
| 6 | That hash also reads the *other* trainer's corpus and any stray backup | `components()` hashes `data/corpus/<safe>` wholesale — which here includes `mww/` (2.1 GB) and `oww.backup-preruonsplit/` (1.5 GB). Measured: **13.9 s per tag computation** |
| 7 | oww's learning rate is hardcoded and `max_negative_weight` silently self-doubles | `openwakeword/train.py:273` `lr = 0.0001`; lines 285-291 and 309-314 double `max_negative_weight` at sequence 2 and again at 3 whenever `best_val_fp > target_fp_per_hour` — and this repo sets that target to **0.1** (`train/oww/train.py`, `create_config`), half of upstream's 0.2. A run asking for 2000 can train at 8000 |
| 8 | The exported oww model is a weight-average of whichever checkpoints cleared a percentile | `openwakeword/train.py:200-224` + the 90th/10th-percentile merge after sequence 3. A large, unlogged variance source |
| 9 | mww's real knobs are unreachable | `train/mww/train.py:159-163` never passes `negative_class_weight` even though `mww_config.build` accepts it; `learning_rates`, `positive_class_weight`, per-set `sampling_weight`/`penalty_weight` and `eval_step_interval` are constants in `train/mww/config.py` |
| 10 | Piper corpus generation is strictly sequential on a 10-core machine | `train/corpus/piper.py:303-378` — one client, one URL, a plain `for` over jobs. Its own docstring prescribes the fix: "Parallelism has to come from separate instances … sharding a multi-voice corpus BY VOICE". The server confirms the constraint: one engine, one global lock (`tts-service/tts_protocol/tts_protocol/server.py`) |
| 11 | `trim_directory` is likewise a serial loop over ~14k clips | `train/corpus/augment.py:200-219` |
| 12 | The holdout the whole loop is judged on is tiny | `jay` 35, `jay_runon` 57, `jen` **10**, `ryan` **6**, `ryan_runon` 14. Adversarial negatives: `extend` 20 + `hey_other` 12 = **32 total** — the "/32" every matched-FA comparison in SPEED.md is keyed on. One ryan clip is 16.7 points |
| 13 | `compare_models.py` emits no machine-readable output and no confidence interval | `eval/src/compare_models.py:101-122` — no `--json`, no CI |
| 14 | A host eval venv is possible; Docker is not required on a Mac for any step | `pymicro_wakeword 2.5.0` and `pyopen_wakeword 1.1.0` both publish `macosx_15_0_universal2` wheels (checked against PyPI). This machine is Darwin 25.5 |

---

## P0 — make a pass cheap and repeatable

### P0.1 Seed every stage, and make the seed part of the run's identity

Nothing is seeded (fact 4). Until this lands, every sweep result is contaminated by the
variance this repo has already measured at 10 points on an identical config.

- Add `--seed` (default 0 = "random, and say so in the log") to `train/oww/train.py`,
  `train/mww/corpus.py`, `train/mww/features.py`, `train/mww/train.py`.
- At the top of each stage's `main()`: `random.seed`, `np.random.seed`,
  `torch.manual_seed` / `tf.keras.utils.set_random_seed`, and `PYTHONHASHSEED`.
- Pass it into `augment_clips(..., seed=...)` — the parameter already exists and is
  already documented (`openwakeword/data.py:369-372`). This needs a one-line patch in
  `patches/` since the call site is upstream's.
- Print the seed in the run header and record it in the tag (P0.2).

**Caveat to write into the comment, because it will otherwise look like a bug:** the
TTS engines are *not* seedable. Piper's VITS samples noise per call
(`tts-service/engines/piper/piper_engine/__init__.py` docstring says so), so a seeded
corpus stage still renders different audio. That is precisely why P0.3 freezes the
corpus instead of trying to reproduce it.

**Verify:** two runs at the same seed, same frozen corpus, same config must produce
byte-identical `.onnx`/`.tflite`. If they don't, the remaining nondeterminism is worth
finding before any sweep starts — it *is* the 10-point variance.

### P0.2 Split the run tag into corpus × config, so two sweep points can coexist

Today `<commit>-d<audio>` (fact 5) is the same string for every point in a sweep, and
mww then refuses to train. Proposed:

```
<commit>[-dirty]-c<corpus>-h<config>
                  ^^^^^^^^ ^^^^^^^^
                  the audio the run consumed   sha over the resolved
                  (this trainer's corpus only) hyperparameters + seed
```

- `train/provenance.py::components()` must scope to the target:
  `data/corpus/<safe>/<oww|mww>`, not `data/corpus/<safe>` (fact 6). That drops a
  cross-trainer false dependency, stops a stray `*.backup-*` directory from renaming
  runs, and cuts the 13.9 s.
- Add `config_tag(dict)` — sha256 over the *resolved* config (every CLI default
  materialised, plus the seed), short-7. Write the full resolved config next to the
  model as `<tag>.config.json`. That file is what the ledger (P0.5) reads.
- Keep the corpus half cheap: hash the corpus **manifest** (P0.3) when one exists and
  fall back to the tree digest when it doesn't.

### P0.3 Make the corpus an explicit, named, frozen artefact

**This is the methodological point, not just a speed trick: during a sweep the corpus
is a held-fixed independent variable.** Regenerating it per run means every comparison
carries a fresh draw of TTS noise — which is half of what the repo's own 77%/67%
observation is made of.

- After a corpus stage completes, write `data/corpus/<wake>/<target>/corpus.json`:
  wake word, engine names + versions + URLs as reported by the `voices` probe, the
  voice list actually used, per-bucket clip counts, every corpus-shaping flag
  (`samples_per_voice`, `runon_fraction`, `child_fraction`, `piper_fraction` /
  `kokoro_fraction`, `real_copies`, exclusions), the wordlist file hash, the seed, the
  wall time, and a content digest.
- Run scripts learn `--corpus reuse|rebuild` (default `reuse` when a manifest exists
  and matches the requested shaping flags; `rebuild` otherwise). This replaces
  `SKIP_CORPUS=1`/`--skip-corpus` as the ergonomic front door while keeping the env
  vars working.
- **Refuse silently-stale reuse.** If the manifest's shaping flags differ from what was
  asked for, fail with a diff rather than reusing. Today `--skip-corpus` with a changed
  `--samples-per-voice` just ignores the flag (`train/oww/train.py:568` says so) and
  with a changed `--augmentation-rounds` it silently reuses stale features (fact 1).

### P0.4 Fix the two caching defects that make "fast iteration" unsafe today

- **The `--skip-corpus` help text is wrong** (fact 1). It claims augmentation re-runs.
  It does not. Correct the text, and make the behaviour explicit rather than inherited
  from an upstream `os.path.exists` check: add `--skip-augmentation` /
  `--rebuild-features`, key the cached `.npy` on
  `(corpus digest, augmentation_rounds, total_length, seed)` via a sidecar
  `features.json`, and rebuild when it doesn't match. Right now, changing
  `--augmentation-rounds` on a `--skip-corpus` run is a no-op that looks like a
  measurement.
- **Make `patches/honour-augmentation-rounds.py` idempotent** (facts 2, 3). Guard the
  substitution with a sentinel check (`if '*config["augmentation_rounds"]' in content:
  already applied`), the way the other patches check their anchor. Then reset the
  clone and re-apply once. Cost of leaving it: the positive-train array is allocated at
  27N rows instead of 3N and `trim_mmap` copies the whole thing — pure disk and time.
  While there, audit the other five patches for the same property; the setup script
  advertises idempotence for all of them.

### P0.5 A sweep runner and a results ledger

The loop the README draws (`EVL -->|not better| TRN`) has no machine-readable edge.
Results live in 40+ `training-*.log` files at the repo root and in scrollback.

- `eval/src/eval_model.py --json <path>` and `eval/src/compare_models.py --json <path>`
  (fact 13). Same numbers, emitted as data.
- `output/<wake>/runs.jsonl`, one line per run: tag, target, corpus id, seed, resolved
  config, wall time per stage, and the eval block — per-speaker detection at matched
  false accepts, per-category false accepts, median latency, the mww cutoff and sliding
  window. Append-only; never overwritten.
- `scripts/sweep.py`: takes a YAML of `{base: {...}, grid: {...}, repeats: N}`, freezes
  the corpus once, then for each point trains → evals → appends. It must run
  `repeats >= 2` by default and report the **spread across repeats beside the effect
  size**, because a difference smaller than the seed-to-seed spread is not a result.
  Resumable: skip any point whose tag is already in the ledger.
- Teach the `eval-models` skill to read `runs.jsonl` so the agent compares against
  history instead of two models at a time.

---

## P1 — make the number trustworthy enough to tune against

Doing P0 without P1 produces a fast machine for generating noise.

### P1.1 Say out loud how small the holdout is, and widen the cheap half

With `ryan` at n=6 and `jen` at n=10 (fact 12), a single clip is 16.7 and 10 points.
The per-speaker rows are the ones CLAUDE.md says decide a ship — and they are the
noisiest numbers in the report.

- Print `n=` beside every per-speaker rate in `eval_model.py` and `compare_models.py`,
  and a **Wilson interval** on each. A row reading `jen 40% (n=10, 95% CI 17-69)` stops
  a sweep from chasing a 10-point "win".
- Add bootstrap CIs on the matched-false-accept comparison in `compare_models.py`, and
  have it refuse the word "better" when the intervals overlap. This is the mechanical
  form of CLAUDE.md's "never compare at a fixed threshold" rule.
- **Grow the adversarial negative set.** 32 clips (`extend` 20 + `hey_other` 12) is the
  denominator of every matched-FA comparison in SPEED.md; one clip is 3.1 points.
  `eval/src/generate_negatives.py` can render more from the same wordlist at a cost of
  minutes. Take `extend` and `hey_other` to ~150 each. Regenerating changes the
  scorecard baseline, so do it **once**, before the first sweep, and re-baseline the
  deploy candidate against it. (`train/provenance.py`'s own note records that
  regenerating the eval corpus moved a category 0/12 → 1/12 — which is the variance
  argument, again.)

### P1.2 A held-out-**voice** synthetic set, for high-n ranking

`eval/src/generate_positives.py` states plainly that its corpus is inside the training
distribution, so it cannot measure speaker generalisation. `wordlists/__init__.py`
already enforces train/eval *phrase* disjointness. Nothing enforces **voice**
disjointness — and that is the axis a hyperparameter sweep needs, because it is the
only one where you can cheaply get n in the thousands.

- Reserve a voice holdout: N Kokoro voices and M Piper (voice, speaker) pairs excluded
  from every corpus build, in a tracked list beside the wordlists so it cannot drift.
- Render the eval positives from those, and report them as a **separate, clearly
  labelled ranking signal** — never as a substitute for the real-speaker gates. A
  synthetic voice is not a person; its value is that it has low variance and can rank
  20 sweep points, after which the top 2-3 go to the real holdout.
- Sanity check to run once: does the synthetic-holdout ranking correlate with the real
  per-speaker ranking across the seven mww runs already in `output/`? If it doesn't,
  the proxy is worthless and should be dropped rather than trusted.

**Done (2026-09-05).** `wordlists/voice_holdout.yaml` is the tracked list (7 Kokoro
voices from the live catalog, 2 single-speaker Piper pairs; audited, and disjoint
from both the trainable set and the in-corpus eval voices), with `load_voice_holdout()`
and `exclude_voice_holdout()` in `wordlists/__init__.py`: both corpus builders
(`train/oww/train.py`, `train/mww/corpus.py`) exclude the entries, the live catalog
is the source of truth (a stale entry exits loudly, naming the file to update; a
missing file is a no-op), and the manifest records the list. `generate_positives.py
--voice-holdout` (make render-voice-holdout) renders the set into its own
`data/corpus/eval/voice_holdout_tts/` at speeds 0.7-1.3 - the training range, since
the held-out axis is the voice - and writes a `set.json` labelling it a ranking set,
not a gate. `eval_model.py --voice-holdout-set` scores it as a separate, labelled
block (its own JSON key, never merged into the gates), which stay on the real
`data/recordings/holdout/`; the top two or three of the ranked points go there.
Seven unit tests in `tests/test_voice_holdout.py` (suite 45 → 52).

**Scope of the guarantee (bug.md B4):** the voice reservation binds corpus
BUILDS after commit 3499919. The frozen cf9c065b corpus predates it, so
models trained on cf9c065b (the four filed sweep rows, the bar-test model)
saw the held-out voices in training; voice-holdout numbers against them
measure nothing the holdout set was built to measure. A clean measurement
needs a post-reservation corpus (new corpus id in the tag's data half).

**Sanity check, run 2026-09-22 on the first four oww sweep points**
(sweeps/oww-training-steps.yaml, ledger output/hey_seeree/runs.jsonl;
all four are 50k-step runs per the B1 erratum above - the check compares
models within one config, so the erratum does not change it): the
synthetic set did NOT move with the real holdout over these four points -
real spread 11.8 pts against a 5.7 pt voice-holdout spread, pearson 0.25
at n=4 (below the 10-point noise floor, so this is a direction check, not a
measurement). The set still does one useful thing the real holdout cannot:
name the failing voice type (all misses at the weakest point were the single
`am_santa` voice, at every speed). Verdict: keep it as a cheap screen for
voice-type failures across many sweep points - not as a proxy for the real
per-speaker gates, which remain the only thing that ranks a deploy decision.


### P1.3 Un-confound the oww trainer's hidden weight doubling

Fact 7: with `target_false_positives_per_hour` at 0.1, `auto_train` can double
`max_negative_weight` at sequence 2 and again at 3. The run-8 measurement recorded in
`train/oww/train.py` (2000 vs 4000) was comparing *requested* weights whose *effective*
values are unknown.

**Logging: implemented (`patches/log-weight-and-merge.py`, applied by the setup script
and both oww Dockerfiles).** Each sequence now prints
`# WEIGHT_AUDIT sequence=<n> requested=<x> doubled=<bool> effective=<y>` at the moment
the decision happens, and the wrapper files the per-sequence list in
`<tag>.config.json` as `effective_max_negative_weight` (appended after the tag is
computed, so the outcome cannot move the tag). `--target-fp-per-hour` already existed;
its help now names the doubling.

The first verified run made the condition itself visible: `best_val_fp` is initialised
to 1000 in `Model.__init__` and is never updated, so the doubling is not conditional at
all - it ALWAYS fires, and a run requesting 2000 trains sequence 1 at 2000, sequence 2
at 4000, sequence 3 at 8000. The run-8 comparison was therefore 8000 vs 16000 on
sequences 2-3, not 2000 vs 4000.

- Re-measure: resolved by inspection plus live audit, 2026-09-22 - no new run
  needed. The doubling condition (`best_val_fp > target_fp_per_hour`, with
  `best_val_fp` fixed at 1000) depends only on `--target-fp-per-hour`, which
  was identical for both run-8 points; `--max-negative-weight` does not
  enter it. The audit lines prove the pattern live (requested 2000 ->
  effective 2000/4000/8000 per sequence, filed in config.json as
  `effective_max_negative_weight`). So run 8 was a fair comparison of
  effective 8000 vs 16000 at sequence 3, and its verdict (4000 not better,
  leave the default) stands as the effective-weight measurement - the note
  in train/oww/train.py is no longer provisional (updated in the same edit,
  per CLAUDE.md). What the audit DOES enable, as a new lever position not
  yet measured: a LOWER setting (e.g. 1000 -> effective peak 4000), if a
  future run's FP rate invites it.

### P1.4 Log the checkpoint merge

Fact 8: the exported oww model is an average of whichever checkpoints cleared the 90th
percentile. That is a plausible mechanism for a large share of the observed
run-to-run variance and it is currently invisible. Log the number of merged
checkpoints and their steps. If the count swings between runs at a fixed seed and
fixed features, that is the variance, found.

**Logging: implemented (same patch).** The merge prints one `# MERGE_AUDIT merged
step=<n> seq=<s>` line per checkpoint that cleared the gate, plus a summary
`# MERGE_AUDIT cleared=<k>/<total> percentile=90 steps=<...>`, and the wrapper files
the steps list in `<tag>.config.json` as `merged_checkpoints`.

**Result, read 2026-09-22 (bug.md B2): the merge is a no-op on this corpus.** Every
real run prints `cleared=0` (grep MERGE_AUDIT logs/ - the 55/55 lines are smoke
runs, where the percentiles degenerate on a 55-entry history). The gate
(openwakeword/openwakeword/train.py) requires val_accuracy >= p90 AND
val_recall >= p90 AND val_fp_per_hr <= p10 SIMULTANEOUSLY, and no checkpoint
clears all three - so `len(models) == 0`, `average_models` is never called, and
the exported model is the final training state. Fact 8 is disproven for this
corpus: the checkpoint merge cannot be the run-to-run variance mechanism, and
this is also why the P0.1 byte-identical reproducibility was achievable - there
is no percentile-dependent averaging step to perturb. Two follow-ups, noted
not acted: `merged_checkpoints: []` (audit present, nothing cleared) is the
EXPECTED value, distinct from null (audit absent) - the wrapper's comment says
so; and a merge gate that never fires is arguably upstream's conjunction being
too strict here - a new lever for a future sweep, not a defect in this pass.

---

## P1.5 — expose the hyperparameters, or there is nothing to sweep

The stated goal is hyperparameter tuning. The surface currently reachable from a
command line is a small and not especially interesting subset.

**openWakeWord** — reachable today: `samples_per_voice`, `training_steps`,
`layer_size`, `max_negative_weight`, `augmentation_rounds`, and the corpus-shaping
fractions. Not reachable, and worth more:

| knob | where it lives now | why it matters |
|---|---|---|
| `lr` | hardcoded `0.0001` at `openwakeword/train.py:273`, then `/10`, `/10` | the first thing anyone tunes |
| `batch_n_per_class` | `custom_model.yml`, never overridden by `create_config` | every step draws 1024 ACAV100M against **50** positives and **50** adversarial negatives. The class balance per batch is a major lever and nobody in this repo has touched it |
| `target_false_positives_per_hour` | `create_config`, hardcoded 0.1 | drives both checkpoint selection and the weight doubling (P1.3) |
| `target_accuracy` / `target_recall` | `create_config`, 0.7 / 0.5 | the checkpoint-merge gate |
| `n_samples_val`, `augmentation_batch_size` | derived / default | secondary |

`create_config` already builds the YAML from a template, so this is plumbing: add the
flags, write them through, record them in `<tag>.config.json`. `lr` needs a small
patch in `patches/` to read it from the config — same shape as
`configurable-corpus-dir.py`.

**microWakeWord** — reachable today: `training_steps`, `batch_size`, and `--model-flags`
as an all-or-nothing passthrough. Not reachable:

- `negative_class_weight` — `mww_config.build` accepts it and `train/mww/train.py:159`
  doesn't pass it (fact 9). One-line fix, and the default is `[20]`, a big lever.
- `learning_rates`, `positive_class_weight`, `eval_step_interval`.
- The per-feature-set `sampling_weight` / `penalty_weight` — hardcoded 2.0/1.0 in
  `build()`. `train/mww/config.py` itself calls this "the per-set lever openWakeWord
  did not have". It is not wired to anything.
- `mixednet` geometry: `pointwise_filters`, `repeat_in_block`, `mixconv_kernel_sizes`,
  `first_conv_filters`, `stride`. `--model-flags` is REMAINDER-style all-or-nothing;
  a sweep wants to move one. Add `--model-flag k=v` (repeatable) that merges over
  `MODEL_FLAGS`, and keep `check_quantization_constraint` running before every launch —
  it already exists and already saves a full run.

Suggested shape for both: `--set key=value` (repeatable) merged into the generated
config, with the resolved result written to `<tag>.config.json`. That way the sweep
runner needs no per-knob plumbing and the ledger records exactly what ran.

---

## P2 — take the remaining wall time (Apple Silicon)

Order matters: with P0.3 in place the corpus stage is *out* of the iteration loop, so
these mostly buy faster **cold** runs and faster corpus re-rolls, not faster sweep
points. Do them after P0/P1.

### P2.1 Shard Piper across processes — the biggest single win left — DONE 2026-09-21

Implemented: `PiperFleet` in train/corpus/piper.py (probe requires every instance to serve an identical
voices catalog; `shard()` pins each model - all its speakers - to one instance,
least-loaded by job count, so no instance reloads a model mid-run; per-stage
`piper/jobs.json` records which instance rendered what). Comma-separated
`--piper-url` on both trainers; `scripts/start-tts-fleet.sh N` (idempotent,
pidfile-managed, comma-joined URL on stdout for `$(...)` capture); `PIPER_URLS`
in both Apple-Silicon run scripts. Verified with a live 2-instance fleet:
every model pinned to exactly one instance, 40/40 clips, single-URL backward
compat intact. N-way throughput measured 2026-09-22 (below): **no win from N>1 on a 10-core box** - one Piper instance already saturates all ten cores, so there is no throughput left to shard out.

Fact 10. `generate_piper_samples` is a serial loop; the engine serialises by design
(one model resident, one lock); the docstring already specifies the fix.

- Accept comma-separated `--piper-url` (as `--kokoro-url` already does) and **shard by
  voice, not round-robin** — round-robin would make every instance reload a model per
  request, which the docstring warns about explicitly.
- `scripts/start-tts-fleet.sh N`: launch N `piper_engine` processes on consecutive
  ports, wait for each `voices` round-trip, print the comma-joined URL, and trap
  cleanup.
- Expected: the mww corpus stage is 4,920 + 984 clips in 6m12s ≈ 16 clips/s aggregate
  against `tools/bench_tts.py`'s 21.66 clips/s single-process. On 10 cores, 6 instances
  should land near 1-1.5 min — roughly **5 minutes off a 14-minute mww run**.
  `tools/bench_tts.py` already exists to confirm it; measure before claiming it.
  REFUTED by the 2026-09-22 measurement below: 6 instances gave 17.26 clips/s, i.e. the
  same as one - the ~5 min is not there, because one instance already owns all ten cores.
- Same mechanism applies to Kokoro-MLX, but with much less confidence: MLX contends on
  one GPU, so expect 2 instances to help and more to not. Measure, don't assume - and
  note SPEED.md's ±30% machine-load rule while doing it.

Measured 2026-09-22, `tools/bench_piper_fleet.py`: the real `select_piper_voices` voice
list + `PiperFleet.shard`, workers == N, one clip per request (Piper is serial - no
batch, `engine.py`), synthesize-and-discard so it measures the fleet not the disk. The
same 1,440-clip oww workload at every N (25 models, 90 (voice, speaker) pairs x 16
clips), built deterministically so a change between sizes is the fleet, not the
workload; each model's one 0.6 s load is paid, not warmed away. One clean run per N, two
for N=2:

  N    wall      clips/s    vs N=1    box while rendering
  1    86.4 s    16.66      1.00x     ~100% busy; 1 instance, ~11-17 onnxruntime threads
  2    158.6 s    9.09      0.55x     100% busy; 2 instances, ~18-19 threads each
  4    84.1 s    17.11      1.03x     100% busy
  6    83.4 s    17.26      1.04x     100% busy
  8    79.3 s    18.16      1.09x     100% busy

N=2 is a clean loss (9.08 and 9.09 on two back-to-back runs, 0.55x); N=4/6/8 all land
inside the box's noise band around N=1 (16.66-18.16, and N=1 re-measured at 17.5-17.8 on
the longer 24-clip/pair workload). **There is no scaling.**

Why the "N x 21.66" never happens: Piper is not single-threaded the way Kokoro-CPU is.
onnxruntime runs intra-op parallelism across cores, and one instance already used ~980%
CPU in `bench_tts.py` - ten of this box's ten cores. The ~17-18 clips/s ceiling is the
*machine's*, not the instance's, and no value of N can create more than ten cores. N=2
is the worst case: two ~19-thread pools oversubscribe ten cores ~2x and thrash (0.55x);
four to eight smaller pools share the cores and recover to the ceiling, so they merely
tie N=1. This confirms and extends `bench_tts.py`'s "a second instance was 0.88x, slower
than one."

So the sharding is correct engineering (voice-pinning works, per-instance assignment and
single-URL backward compat verified) but it is NOT "the biggest single win left": on a
10-core Mac one Piper instance is as fast as any fleet, and N=2 is slower. The mww corpus
stage (Piper-majority) is already at the box's Piper ceiling and adding instances does not
cut it. It *would* help where one instance cannot already use every core - a wider box,
or if each instance's onnxruntime intra-op threads were capped to a fair 10/N share -
but that is a different change than the fleet as shipped.

Caveat: the box carried background load this session (Apple ML churn, 1-min average
2.6-17 across the runs), so the absolute ceiling reads low against `bench_tts.py`'s
21.66 (which also used the shorter "hey seeree" phrase, not the longer oww wordlist
phrases). That pushes every number down a little; it does not change the scaling
verdict, and the ~100% box utilization in every run shows the piper work - not the
background - bound each measurement.

### P2.2 Parallelise the two serial CPU passes over the corpus - CLOSED 2026-09-22, negative result: neither pass is worth it

Measured 2026-09-22 before deciding, per the item's own instruction. Box was under a
background training sweep the whole time (load average ~11 on 10 cores), so all
absolute numbers are inflated; serial runs were taken back-to-back on FRESH copies of
the same 2,000-clip subset (500 per oww subdirectory, first 500 in sorted order,
padded with 300 ms of trailing silence to recreate the pre-trim state — the frozen
corpus is already trimmed in place, so re-trimming it would skip nearly all writes
and flatter the number). Numbers, `train-applesilicon/.venv/bin/python`:

- `trim_directory` serial: 0.33 s and 0.35 s wall for 2,000 clips, every one trimmed
  and rewritten — **0.17-0.18 ms/clip** (clips are short: 0.4-2.5 s, 16 kHz mono,
  so the RMS pass is trivial). Extrapolated over the full 22,144-clip oww corpus:
  **~4 s** (0.06 min). That settles what the 2026-09-09 log left open: its 17m45s
  "corpus+trim" was corpus generation, not trim. Caveat: measured on the host Mac;
  a slow Docker/CPU box could be several times worse, but the per-clip work is
  read + tiny RMS + write on short files, and even 10x is under a minute.
- `add_child_range_copies` serial: 40 x 0.5 s runon clips, fraction=1.0 (every clip
  through read + resample_poly + time_stretch + write): 0.09 s wall, **2.2 ms/clip**;
  one 2.0 s synthetic clip (Piper-scale length) averaged **6.3 ms/clip** over 10 runs
  (min 5.4, max 12.7 — the spread is the sweep's load). The frozen corpus's synthetic
  positives are 2,904 runon clips, so at the default 0.5 fraction VTLP is ~1,452
  clips, **~3 s** (pathological all-22,144-at-2 s would be ~70 s, still under the
  ~2 min bar, and that composition does not exist).

Both serial costs at full size are single-digit seconds, far under the "worth
attacking (>~2 min at full size)" bar, and the per-clip work is dominated by
opening and rewriting a ~40 KB file, which more threads buy little of. **No code
change** — `trim_directory` and `add_child_range_copies` stay serial. A
ProcessPoolExecutor version would add spawn/pickling machinery to the hottest
shared module to save ~4 s of a 14-35 minute run; the byte-identity verification
that would be required for it therefore has no implementation to verify.

`trim_directory` (fact 11) and `add_child_range_copies` are per-clip, independent, and
run over ~14k files on a 10-core machine. A `ProcessPoolExecutor` is mechanical. Worth
a stopwatch first — the 2026-09-09 log shows corpus+trim at 17m45s without separating
them, so nobody knows what trim actually costs. Measure, then decide.

**Decision (2026-09-22): the stopwatch says no.** Trim is ~4 s at full size and VTLP
~3 s (numbers above); the 17m45s figure was corpus generation. Closed, no change.

### P2.3 Measure CoreML for the oww feature stage

The augmentation + feature stage is the largest non-TTS stage on the host (~6 min in
the 2026-09-09 run, 11-12m30s in SPEED.md's range), and it is onnxruntime on CPU.
`train-applesilicon/pyproject.toml` notes CoreMLExecutionProvider is present in the
wheel and that "the CoreML route was abandoned" — but SPEED.md's abandoned-Metal
section is about **training** (MPS/CoreML for the torch model, and tensorflow-metal for
mww), not about the melspectrogram/embedding ONNX models.

So this is an open question, not a closed one. `patches/feature-device-selection.py`
already keys the device off onnxruntime's real providers, which is the right place to
add a CoreML branch. Probe it the way `tools/tf_probe.py` probes TF: run the actual
embedding model on the actual batch shape, CPU vs CoreML, on this machine. If it wins,
it is several minutes off every cold oww run. If it doesn't, write that into SPEED.md
so the question is closed properly rather than by assumption.

**Do not** chase MPS for the oww training loop. Post-P0 it is ~5 minutes of a ~6-minute
sweep point, and SPEED.md already explains that oww is pure-GEMM and Accelerate already
wins that half.

**Done (2026-09-22), negative by measurement.** `tools/coreml_probe.py` ran the real
models at the real batch shape (16 x 19,200 samples, ncpu 5): CoreML is 0.28x on
melspectrogram and 0.40x on embedding (0.39x per batch - ~31 min where CPU measures
~12), and the embedding output drifts up to 6.1e-02 (above float noise, and the
features are baked into the model). The graphs only partially convert (11/18 and
44/65 nodes). Result recorded in SPEED.md, "CoreML for the oww feature stage:
measured, loses"; no CoreML branch is being added.

### P2.4 Finish the Mac story: a host eval environment — DONE 2026-09-22

Implemented: `eval/pyproject.toml` (+ uv.lock, .dockerignore) - the fifth host uv env
(`eval/.venv`, Python 3.11, the image's base) with the Dockerfile's exact pins: it
resolved to pymicro-wakeword 2.5.0, pyopen-wakeword 1.1.0, numpy 2.4.6, scipy 1.17.1,
PyYAML 6.0.3, onnxruntime 1.30.0 + tqdm/requests/scikit-learn (the .onnx path's
declared-but-undeclared imports). `scripts/setup-eval-host.sh` - idempotent
(verified: second run changes no package list and no .pth) - reuses the
openwakeword/ clone `uv pip --no-deps -e` like the trainers, writes a path .pth and
asserts `openwakeword.__file__` against the PEP 660 shadowing (verified live against
train-applesilicon/.venv, which has exactly that bug: `__file__` is None). The host
invocation is the PLAIN-PATH form from the repo root (`eval/.venv/bin/python
eval/src/eval_model.py`) - `python -m eval.src.X` fails, and so does `python -m eval.X`
on a host checkout, because the `eval` package only exists inside the image's mount;
eval_model.py/compare_models.py gained the try/except import guard the other four
scripts already carry, and both forms were proven (module form via a simulated mount).
Verified end-to-end with no Docker: hey_seeree.onnx (md5
fdc78d06c42028d7100c596074f3ffaf) 84% detection, weakest speaker ryan 33%; newest mww
run hey_seeree_ecbf160-dirty-da01854d 47%, weakest jen 10%; compare_models on two mww
runs (matched-FA table, per-speaker, 1000-resample bootstrap CI, "not a result"
verdict); --json writes. Makefile `eval` is the host path, `eval-docker` keeps the
container route; SKILL.md's invocation section is host-first. Left: the mww run
script's closing "Evaluate (Docker)" pointer (run scripts were out of scope for this
change).

Eval is the last step that still requires Docker on a Mac
(`run-mww-training-applesilicon.sh` ends by telling you to `cd eval && docker compose
run`). That is a real cost in the tuning loop — it is the step you run most often.

Fact 14 says it is unnecessary: both deployment-runtime wheels publish
`macosx_15_0_universal2`. Add `eval/pyproject.toml` + `scripts/setup-eval-host.sh` as
the fifth host uv env, reusing the `openwakeword/` clone that
`setup-applesilicon-trainer.sh` already makes. Keep the image — it is the right answer
on the CUDA box and on any eval-only machine — but make the host path the documented
one on a Mac, and say in the file's header comment that the two must stay at the same
pinned wheel versions or the numbers stop being comparable.

### P2.5 One entry point

Four setup scripts, four run scripts, two engine `uv run` invocations to remember, and
the ordering lives in CLAUDE.md. Add a `Makefile` (or `scripts/mac`) with
`setup`, `tts-up`, `tts-down`, `corpus`, `train-oww`, `train-mww`, `eval`, `sweep`,
`status`. Thin wrappers only — the scripts keep their logic and their comments. This is
ergonomics, but it is the difference between "first-class" and "documented".

---

## P3 — guardrails, so the loop stays cheap

### P3.1 A smoke mode

Every expensive lesson in CLAUDE.md — `os.mkdir` not recursive, `str.strip` not
`removesuffix`, the quantization constraint, the mislabelled uppercase positives — cost
a full training run. Add `--smoke`: 2 voices, 20 clips each, 50 training steps, real
augmentation, real conversion, real manifest. Target under 2 minutes end to end for
both targets. Run it in the sweep runner before the first real point, and after any
change to `train/`.

### P3.2 Tests for the pure functions

There are none. These have no I/O and have each already caused a failure:
`mww_config.check_quantization_constraint` / `mixednet_slices_dropped`,
`phrase_end_sample`, `trim_silence`, `provenance.digest_tree` + the new `config_tag`,
`manifest`'s cutoff selection, `wordlists`' disjointness checks, and the matched-FA
threshold search in `compare_models`. A `pytest` run of seconds guards decisions that
currently cost tens of minutes each.

### P3.3 Repo hygiene

- 40+ `training-*.log` (and two multi-MB ones) in the repo root, plus
  `training_config.yaml`. Write logs to `logs/` and gitignore it. The root is the first
  thing an agent reads.
- `data/corpus/hey_seeree/oww.backup-preruonsplit/` — 1.5 GB that currently perturbs
  the run tag (fact 6). Either delete it or move it outside `data/corpus/`.
- Stale `__pycache__` for cpython-311/312/314 across `train/` and `eval/` suggests three
  interpreters have run this code. Worth a line in the docs about which is canonical.

---

## Where this lands

| | today | after P0 | after P0+P2 |
|---|---|---|---|
| mww, cold (corpus + features + train + manifest) | 14m14s | ~14m | ~9m |
| oww, cold (corpus + augment + features + train) | ~35m | ~35m | ~25-30m |
| **mww, one sweep point** (frozen corpus + features) | ~7m, and the tag collides | **~7m, filed and reproducible** | ~7m |
| **oww, one sweep point** (frozen corpus + features) | ~6m, undocumented and unsafe | **~6m, explicit and keyed** | ~6m |
| sweep points per hour, both targets | effectively 0 | **~8** | ~8 |

The headline: P0 does not make a pass much faster — **it makes the pass you already had
usable**, which is the actual blocker. P2 is what buys the cold-run time back.

---

## Deliberately not proposed

Stated so nobody spends a run re-deriving what this repo already measured:

- **MPS/Metal for either trainer.** Closed in SPEED.md: Docker Desktop passes no Metal
  device, tensorflow-metal does not pair with TF 2.21.0, and the oww model is pure GEMM
  where Accelerate already wins. The one *open* Metal-adjacent question is CoreML for
  the ONNX **feature** models (P2.3), which is a different question.
- **More corpus depth.** Measured twice, both ways, both failures — `train/mww/corpus.py`
  point 4 and SPEED.md. Depth is not the lever.
- **A TTS clip cache keyed on (voice, text, speed).** Both engines are stochastic per
  call and the repo counts that as free diversity. Freezing the corpus (P0.3) gets the
  same speed without flattening the distribution.
- **Unpinning torch 2.5.1 / TF 2.21.0 / piper-tts 1.7.0.** Every pin in
  `train-applesilicon/` and `train-mww-applesilicon/` carries a comment naming what
  broke. Chasing a newer wheel invalidates the host-vs-container comparisons those
  environments exist to make.
- **Rewriting either upstream trainer.** The `patches/` approach is right; it keeps the
  host and container paths comparable. Every change above that touches upstream should
  be a new patch file with the same explain-the-why header the existing six have.

## One latent trap, noted while reading

`openwakeword/train.py` built `val_steps` as `np.int16` in sequences 2 and 3.
Sequences 2 and 3 run at `steps/10`, so at the default 50,000 they were fine - but
`--training-steps` above ~327,000 overflowed to negative validation steps with no
error. **Now fixed in `patches/log-weight-and-merge.py` (int64, like sequence 1's
array - identical below the bound).** (SPEED.md's run-11 note already
found 100k steps to be worse at matched false accepts, so this is a trap rather than a
recommendation.)

---

## Status — 2026-09-20, committed as `5b09148`

**P0: implemented.** What exists and how it hangs together:
- **Seed (P0.1):** `--seed` on `train/oww/train.py`, `train/mww/{corpus,features,train}.py`
  (default 0 = unseeded, said in the run header). oww subprocess seeded via
  `patches/seed-augment.py` (global RNGs only - `augment_clips` in upstream data.py
  takes NO seed parameter and draws from the global random/np/torch, so one global
  seed before the first call covers it; the earlier kwarg-passing version of the
  patch crashed the first real run in the augmentation stage and was fixed
  2026-09-21, with a self-heal pass); mww TF subprocess via `TF_SET_RANDOM_SEED`;
  `PYTHONHASHSEED` pinned in the oww subprocesses. Seed is in the
  tag (config half). TTS engines remain unseeded — the corpus is frozen, not reproduced.
- **Tag (P0.2):** `provenance.run_tag(wake_word, target, config, fallback)` →
  `<code>[-dirty]-c<corpus7>[-h<config7>]`; legacy `-d` format kept for target=None.
  Corpus half is manifest-aware (hash of `corpus.json` bytes, else scoped tree digest;
  13.9 s → 1.6 s). mww tags come from `train.mww.train --print-tag` (Apple Silicon
  script); the container script uses `provenance --target mww` (no config half — manual
  collisions still fail loudly). oww files `.last_run_tag` + `<tag>.config.json` beside
  the model; run-oww-training.sh reads the filed tag.
- **Frozen corpus (P0.3):** `train/corpus/manifest.py` writes `corpus.json` after each
  corpus stage (engines, voices, per-bucket counts, every shaping flag, seed, wall
  time, content digest). oww: `--corpus auto|reuse|rebuild` (auto reuses on manifest
  match); mww: `--skip`. Both run scripts' SKIP_CORPUS paths verify the manifest and
  exit with a diff on mismatch.
- **Caching (P0.4):** oww `features.json` sidecar keys the .npy cache on
  (corpus digest, rounds, seed); mismatch → recompute loudly; `--rebuild-features`
  forces it; `--skip-corpus` help corrected. `patches/honour-augmentation-rounds.py`
  made idempotent (collapses N factors → 1) and the working clone's 3× state healed;
  the other six patches sentinel-audited.
- **Sweep + ledger (P0.5):** `scripts/sweep.py` (YAML grid × repeats, deterministic
  seeds `base+1000·point+repeat`, frozen corpus, resumable, `--dry-run`), `train/ledger.py`
  (append-only `output/<wake>/runs.jsonl`, duplicate-tag refusal, mean[min–max]
  summarise, `python -m train.ledger`). `eval/src/{eval_model,compare_models}.py --json`.
  eval-models skill reads the ledger first.

**P1 (partial):** per-speaker n= + Wilson CIs and matched-FA bootstrap CIs with a
NOT DISTINGUISHABLE gate (verified on two real models). All sweep knobs reachable:
oww `--lr` (`patches/configurable-lr.py`), `--batch-n-per-class`,
`--target-fp-per-hour/-accuracy/-recall`, `--n-samples-val`, `--augmentation-batch-size`,
repeatable `--set k=v`; mww `--learning-rates`, `--positive/negative-class-weight`,
`--eval-step-interval`, six per-set sampling/penalty weights, repeatable
`--model-flag k=v` (merged before the quantization pre-check). All hash into the tag's
config half.

**P3:** `tests/` — 45 tests over the pure functions (no pytest in any venv; run via
`train-applesilicon/.venv/bin/python tests/test_*.py`); logs → `logs/`; pycache cleared.

**Progress 2026-09-21/22 (post bar test):**
- First real sweep, 2026-09-22: sweeps/oww-training-steps.yaml (25k vs 50k
  steps x 2 repeats, frozen corpus cf9c065b, host route). 4/4 filed in
  output/hey_seeree/runs.jsonl, 0 failed, ~17 min/point end-to-end
  (train + eval + ledger) - the loop (frozen corpus, sidecar features,
  tags, sweep, ledger, host eval) is proven end to end on this Mac.
  **ERRATUM (bug.md B1, found 2026-09-22): the grid was never applied to
  the trainer commands** - every point ran the base config (50k steps, the
  parser default) and was filed under a grid label the run did not use. The
  mislabeled "25k vs 50k NOT DISTINGUISHABLE" headline did not say that; it
  said 50k vs 50k across four seeds. The rows stay in the ledger as they are
  (append-only, and each row's config.steps already records the true
  config, so the measurement inside them is real): four 50k-step runs,
  seeds {42, 43, 1042, 1043} - a direct measurement of this repo's seed
  noise floor at one config: detection 86.3% mean, 78.4-90.2 spread
  (11.8 points, against the 10 points quoted from memory); adversarial FA
  3-4%, under the gate; ryan at 50% at every point. The grid fix landed
  (scripts/sweep.py threads the grid through trainer_cmd; --dry-run prints
  the resolved command; tests/test_sweep.py asserts a grid value reaches the
  rendered command and would have caught this), and the **training-steps
  lever is now measured** on the fixed runner - re-run 2026-09-22, full
  result in the B1-re item below. Short version: no win from 50k.
- **B1-re (2026-09-22): the training-steps re-run on the fixed runner - the
  lever is now measured, and 50k is not a win.** Re-ran sweeps/oww-training-steps.yaml
  (25k vs 50k steps, 2 repeats each) on the frozen cf9c065b corpus; tag
  c6542f5-dirty, 4 points (seeds 42/43 at 25k, 1042/1043 at 50k). Grid fix
  verified in the data: every record's config.steps now equals its grid label
  (grid==cfg for all four), unlike the original four rows where the two "25k"
  rows actually trained at 50k. Matched-false-accept comparison - the harness
  takes one --threshold, so the matched-FA curve was built by hand (I drove
  eval_model.py across decision thresholds 0.25-0.85 per model, 40 evals,
  total-negative FA over the 366-clip negatives set): the two settings occupy
  near-disjoint FA bands (25k reaches 0.55-2.19% total-negative FA, 50k only
  2.46-3.01%), and in the comparable region (~2.2-2.5% FA) detection is the same
  - 25k ~78-82%, 50k ~77-81%, overlapping inside the 2-6 point repeat spread and
  far under the 10-point noise floor: **NOT distinguishable at matched false
  accepts.** 50k only reaches its higher detection ceiling (up to 90% for seed
  1042) at 3.0-3.3% FA, a higher-false-accept operating point the 25k models
  cannot reach - so 50k buys its extra detection with more false triggers, not
  for free. Per speaker (the axis that matters, never pooled): ryan is the
  lowest-detection voice at every point and step count (17-50% across the curve;
  17-33% in the comparable ~2.5% FA band) while jay (77-94%) and jen (80-100%)
  are solid for both - doubling steps does not lift the weak voice. **Verdict:
  doubling training steps 25k->50k is not a win** - no detection gain in the
  achievable band, a false-accept cost to reach the higher ceiling, no
  per-speaker improvement. Consistent with the corpus-depth result (SPEED.md:
  more of the training signal did not produce a deployable model). Caveats: 366
  negatives quantise FA to 0.27%/clip, so the FA bands nearly touch without
  overlapping and the "match" reads each curve at its nearest common point; one
  25k repeat (seed 43) aborted on the first pass in an onnxruntime thread race
  (recursive_mutex lock failed, SIGABRT) during feature computation at machine
  load ~20 and was re-run cleanly at load ~8-10 (an environment flake, not a
  model property - the same abort is the known non-fatal tflite-conversion
  warning); the corpus cf9c065b predates the P1.2 voice reservation (3499919) so
  the voice-holdout axis is contaminated for these models too (the per-speaker
  numbers here are the standard in-distribution set); per-speaker n is small
  (ryan 6 clips).
- P1.1 negative growth: DONE - wordlists/hey_seeree.yaml extend 20->148, hey_other
  12->150; 298 clips rendered (kokoro-mlx, 18 voices) into
  data/corpus/eval/negatives_tts, 12 stale pre-widening hey_other clips removed.
  Scorecard baseline moves - the deploy candidate must be re-baselined before a
  ship call (commit 73bb78d).
- P2.1 Piper sharding: DONE (commit 7036f74) - PiperFleet with voice-pinned
  sharding, start-tts-fleet.sh, PIPER_URLS in both run scripts. N-way throughput
  measured 2026-09-22: no win from N>1 - one instance saturates 10 cores, N=2 is a 0.55x
  loss, N=4-8 tie N=1 (negative result, full note at P2.1).
- P1.3/P1.4: DONE (commit 456e443) - WEIGHT_AUDIT/MERGE_AUDIT into config.json;
  finding: the doubling is unconditional (best_val_fp never updated).
- P3.1/P2.5: DONE (commit bca9f53) - --smoke on both trainers + Makefile
  entry point (make help / smoke-oww / smoke-mww / test / eval / corpus-*).
  Verified end-to-end 2026-09-22: oww smoke 13.5 min (exit 0, corpus reused,
  features recomputed, canonical md5 + tag untouched), mww smoke ~1 min
  (exit 0, tflite + ROC written). The first live runs caught and fixed
  three environment bugs: a fused newline in the mww run script, uv's PEP 660
  editable install shadowed by a namespace package from the repo root (now
  a path .pth + __file__ asserts in setup/run), and get_default taking the
  dest, not the option string, on recent 3.12.
- P2.4 host eval env: DONE 2026-09-22 - eval/pyproject.toml + uv.lock +
  .dockerignore, scripts/setup-eval-host.sh (idempotent, reuses the
  openwakeword/ clone), plain-path import guards in eval_model.py / compare_models.py,
  Makefile eval -> host (eval-docker keeps the image), SKILL.md host-first.
  Proven on this Mac with no Docker: oww + newest mww scorecards, matched-FAR
  comparison, --json, both invocation forms (full note at P2.4).
- P2.2 parallel trim: CLOSED 2026-09-22, negative result - measured first, per the
  item: trim_directory 0.17-0.18 ms/clip (2,000-clip subset, load ~11 on 10 cores),
  ~4 s at full 22,144; VTLP 2.2 ms/0.5 s clip, 6.3 ms/2 s clip, ~3 s over the actual
  synthetic set. Both far under the ~2 min bar - no code change (full note at P2.2).

**Still open:**
- The B1 sweep re-run: DONE 2026-09-22 (tag c6542f5-dirty, 4/4 points, grid fix
  verified in the data) - the training-steps lever is now measured and 50k is
  not a win at matched false accepts (full note in the B1-re item above). The
  original four rows stay as a 50k seed-noise measurement; the four c6542f5
  rows supersede the mislabeled 25k-vs-50k comparison.
- P1.2's rendered set is COMPLETE 2026-09-22: the Piper half (10 clips,
  en_GB-alan-medium + en_US-lessac-medium x the 5-speed grid) rendered with
  the tts-service Piper engine on 8898 into the same labelled directory
  (45 clips total; the eval block reports n=45).
- P2.1 Piper-fleet throughput at N>1: MEASURED 2026-09-22 (tools/bench_piper_fleet.py,
  N = 1/2/4/6/8, same 1,440-clip oww workload) - negative result: 16.66 / 9.09 / 17.11 /
  17.26 / 18.16 clips/s, i.e. no scaling (one instance saturates 10 cores; N=2 loses 45%,
  N>=4 only ties N=1). The ~5-min wall-time claim in the plan is refuted and must NOT go
  into SPEED.md or the README.
- The CUDA images on the training box still need their next build to pick up
  the new patches (the CPU images were rebuilt 2026-09-22).
- **RESOLVED 2026-09-22: the corpus/holdout contamination, stated here where
  the positional guarantees live.** Caught live, not on paper: a
  post-reservation run was launched expecting a rebuild, and the reuse check
  REUSED the pre-reservation cf9c065b corpus - `matches_requested` compared
  the shaping flags and the seed but never the voice set, so the 7 held-out
  Kokoro voices (and the 2 Piper pairs) trained the model after all. The fix
  adds the voice set to the reuse diff (top-level `voices`, compared per
  engine; a manifest missing the field refuses, as with a missing shaping
  key), and the oww voice-set resolution - catalog probe,
  mispronunciation/`--exclude-voices`/holdout exclusions, Piper selection -
  now runs BEFORE the corpus-mode decision so the check and the build
  consume one computation (the old "only when generating" deferral was what
  made the check voice-blind); the mww `--skip` check got the same axis. Six
  new tests, one shaped exactly like this incident. The corpus was then
  rebuilt with `--corpus rebuild`: **c348af7b** (17,241 files, 15 Kokoro
  voices, the holdout list recorded in the manifest), and a baseline point
  trained on it: `81490a6-dirty-c348af7b-hc250775` (seed 55, 25k steps).
  **First clean voice-holdout reference: 28/45 (62%) at threshold 0.5,
  median latency 94 ms** - the baseline for ranking future points, which
  will all be built on c348af7b or later. The contaminated-era 4/45 is NOT
  comparable to it: different models at a fixed threshold, which this repo's
  invariants forbid. The real gates are unchanged and the standing caveat
  holds: on the clean corpus ryan is still 1/6 (17%) - the rebuild did not
  move the one thing the corpus cannot fix.
- **Follow-up flagged by that fix: `seed` is not in either trainer's reuse
  request**, even though `matches_requested` special-cases it and the
  manifest records it. If the corpus augmentation consumes `--seed`, a
  different-seed run would silently reuse another seed's corpus - the same
  one-axis hole, one axis over. (The sweep's frozen-corpus design may make
  "corpus built at seed 0; the seed varies only features/training" the
  INTENDED semantic - in which case say so in the help text; otherwise pass
  the seed in the request.)
**Docker image rebuild: done 2026-09-22** - both CPU images rebuilt with the
new patches and code (verified in-image: 8 PATCHED markers in the oww clone,
smoke/ledger code present in the mww image). The container routes now match
the host route; the CUDA images on the training box still need their next
build to pick the new patches up.
- P0.1 bar test (two same-seed runs → byte-identical .onnx) — the gate before any sweep.
  **MET at 2026-09-21 19:20**: run A (full corpus rebuild seed 1234 + augment + features
  + train) and run B (`--skip-corpus`, features reused via the sidecar) produced
  byte-identical .onnx (md5 fdc78d06c42028d7100c596074f3ffaf, tag
  5b09148-dirty-cf9c065b-h8b4b5fa); augmentation determinism separately probed
  (same-seed augment_clips batches byte-identical). Run C (independent
  `--rebuild-features`) **also passed ~19:45**: rebuilt .npy identical and .onnx
  byte-identical - all three links (augment, features, train) are deterministic at a
  fixed seed. Two real bugs were
  found and fixed the expensive way, both recorded at the site:
  (a) the first seed-augment version passed `seed=` to `augment_clips`, which takes
  no such parameter - fixed + self-heal pass in the patch; (b) `convert_to_tflite`
  imported TF in-process after the torch run and SIGABRT'd the whole trainer on
  macOS arm64 - now a subprocess (the abort dies with the child, run exits 0).
- A real (non-dry-run) `sweep.py` pass end to end.
- Existing corpora have NO manifest (not backfilled on purpose — the mww corpus was
  built with non-default flags). First rebuild writes one.
- P2 (Piper fleet sharding, parallel trim, CoreML feature probe, host eval env,
  Makefile), P1.1 (negatives 32 → ~300, needs TTS), P1.3/P1.4 (weight-doubling and
  checkpoint-merge logging), P3.1 (`--smoke`).
- The np.int16 `val_steps` overflow trap (>~327k steps) is unpatched; a sweep reaching
  for long runs must patch it first.
- Docker images are NOT rebuilt with the new patches (seed-augment, configurable-lr) —
  host route only until the next container build.
