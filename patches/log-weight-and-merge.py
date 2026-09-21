"""Patch openwakeword's train.py so auto_train's two hidden decisions print
machine-readable audit lines, and val_steps can no longer overflow.

P1.3 (improvement.md): a sequence that ends with best_val_fp above
target_false_positives_per_hour DOUBLES max_negative_weight, and it can fire
twice - a run requesting 2000 can train at 8000. This repo targets 0.1 (half
of upstream's 0.2), and the measurement that set the default (tuning run 8:
2000 vs 4000) compared REQUESTED weights whose effective values were never
recorded. (Worse than "can fire": best_val_fp is initialised to 1000 in
Model.__init__ and never updated, so the condition is simply always true and
a run requesting N trains sequence 2 at 2N and sequence 3 at 4N - the audit
below makes that visible for the first time.) This patch prints, per
sequence, at the moment the decision happens:

    # WEIGHT_AUDIT sequence=<n> requested=<x> doubled=<true|false> effective=<y>

P1.4: the exported model is a weight-average of whichever checkpoints
cleared the 90th-percentile gate. Which checkpoints - and how many - was
previously invisible, and the gate is thresholded on the run's own validation
metrics, so the count is a plausible source of the 10 points of run-to-run
variance this repo has already measured at an identical config (SPEED.md,
CLAUDE.md). This patch prints one line per merged checkpoint plus a summary:

    # MERGE_AUDIT merged step=<training_step_ndx> seq=<n>
    # MERGE_AUDIT cleared=<k>/<total> percentile=90 steps=<comma-list>

The wrapper (train/oww/train.py) parses these from the subprocess stdout and
files them in <tag>.config.json as effective_max_negative_weight (the
per-sequence list) and merged_checkpoints (the steps list) - appended AFTER
the run tag is computed, because they are recorded outcomes of the run, not
inputs to it.

ALSO FIXED HERE, the "one latent trap" in improvement.md: sequences 2 and 3
build val_steps as np.int16 (upstream lines 299 and 319), so a
--training-steps above ~327,000 overflows to NEGATIVE validation steps with
no error. They are int64 now, like sequence 1's array; below that bound the
array is identical modulo dtype. (SPEED.md's run-11 note already found 100k
steps to be worse, so this un-traps rather than recommends.)

The audit is PRINTS ONLY: no RNG draw (a seeded run stays byte-reproducible -
the P0.1 bar), no control-flow change, no new config key. The
self._audit_seq_bounds bookkeeping records len(self.best_models) before each
sequence so the merge audit can say which sequence each merged checkpoint
came from; it is read-only bookkeeping that train_model neither sees nor
needs.

Idempotent: the sentinel is the first audit comment, and all five edits land
in one atomic pass (no partial state - if any anchor is missing, nothing is
written).
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

SENTINEL = "# PATCHED: audit the negative-weight schedule (improvement.md P1.3)"
if SENTINEL in content:
    print(f"Already patched: {path}")
    sys.exit(0)

# Five anchored edits, all in auto_train. Each anchor must occur exactly once
# or the whole patch is refused: partial application would plant the sentinel
# and make later runs report "Already patched" over a broken file.
edits = [
    # 1. Top of auto_train: remember where sequence 1's checkpoints begin.
    ("        # Get false positive validation data duration\n"
     "        val_set_hrs = 11.3\n",
     "        # Get false positive validation data duration\n"
     "        val_set_hrs = 11.3\n"
     "        # PATCHED: remember where each sequence's checkpoints begin in\n"
     "        # best_models (appended before sequences 2 and 3 below), so the\n"
     "        # merge audit at the end of this method can say which sequence\n"
     "        # each merged checkpoint came from. Read-only bookkeeping.\n"
     "        # See patches/log-weight-and-merge.py.\n"
     "        self._audit_seq_bounds = [len(self.best_models)]\n"),

    # 2. Sequence 1: uses the requested weight as-is; announce it. (The
    #    sequence-2/3 boundaries get appended where edits 3 and 4 land, which
    #    is AFTER the previous sequence's train_model call has run.)
    ("        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        val_steps = np.linspace(steps-int(steps*0.25), steps, 20).astype(np.int64)\n",
     "        # PATCHED: audit the negative-weight schedule (improvement.md P1.3):\n"
     "        # sequence 1 uses the requested weight as-is; sequences 2 and 3 may\n"
     "        # double it (below). Machine-readable so train/oww/train.py can file\n"
     "        # it in <tag>.config.json. Print only - no RNG draw, no flow change.\n"
     '        print(f"# WEIGHT_AUDIT sequence=1 requested={max_negative_weight} "\n'
     '              f"doubled=false effective={max_negative_weight}", flush=True)\n'
     "        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        val_steps = np.linspace(steps-int(steps*0.25), steps, 20).astype(np.int64)\n"),

    # 3. Sequence 2: the first doubling, plus the np.int16 trap.
    ("        # Adjust weights as needed based on false positive per hour performance from first sequence\n"
     "        if self.best_val_fp > target_fp_per_hour:\n"
     "            max_negative_weight = max_negative_weight*2\n"
     '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
     "\n"
     "        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        val_steps = np.linspace(1, steps, 20).astype(np.int16)\n",
     "        # Adjust weights as needed based on false positive per hour performance from first sequence\n"
     "        _seq_requested = max_negative_weight\n"
     "        if self.best_val_fp > target_fp_per_hour:\n"
     "            max_negative_weight = max_negative_weight*2\n"
     '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
     "        _doubled = \"true\" if max_negative_weight != _seq_requested else \"false\"\n"
     '        print(f"# WEIGHT_AUDIT sequence=2 requested={_seq_requested} "\n'
     '              f"doubled={_doubled} effective={max_negative_weight}", flush=True)\n'
     "\n"
     "        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        # PATCHED: int64, like sequence 1 - np.int16 overflowed to NEGATIVE\n"
     '        # validation steps at --training-steps > ~327k with no error\n'
     '        # (improvement.md\'s "one latent trap"); identical array below it.\n'
     "        val_steps = np.linspace(1, steps, 20).astype(np.int64)\n"
     "        self._audit_seq_bounds.append(len(self.best_models))\n"),

    # 4. Sequence 3: the second (final) doubling, plus the same trap.
    ("        # Adjust weights as needed based on false positive per hour performance from second sequence\n"
     "        if self.best_val_fp > target_fp_per_hour:\n"
     "            max_negative_weight = max_negative_weight*2\n"
     '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
     "\n"
     "        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        val_steps = np.linspace(1, steps, 20).astype(np.int16)\n",
     "        # Adjust weights as needed based on false positive per hour performance from second sequence\n"
     "        _seq_requested = max_negative_weight\n"
     "        if self.best_val_fp > target_fp_per_hour:\n"
     "            max_negative_weight = max_negative_weight*2\n"
     '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
     "        _doubled = \"true\" if max_negative_weight != _seq_requested else \"false\"\n"
     '        print(f"# WEIGHT_AUDIT sequence=3 requested={_seq_requested} "\n'
     '              f"doubled={_doubled} effective={max_negative_weight}", flush=True)\n'
     "\n"
     "        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()\n"
     "        # PATCHED: int64, same np.int16 trap as sequence 2.\n"
     "        val_steps = np.linspace(1, steps, 20).astype(np.int64)\n"
     "        self._audit_seq_bounds.append(len(self.best_models))\n"),

    # 5. The checkpoint merge: which checkpoints cleared the gate, and which
    #    sequence each came from.
    ("        # Get models above the 90th percentile\n"
     "        models = []\n"
     "        for model, score in zip(self.best_models, self.best_model_scores):\n"
     '            if score["val_accuracy"] >= accuracy_percentile and \\\n'
     '                    score["val_recall"] >= recall_percentile and \\\n'
     '                    score["val_fp_per_hr"] <= fp_percentile:\n'
     "                models.append(model)\n",
     "        # Get models above the 90th percentile\n"
     "        models = []\n"
     "        _cleared = []\n"
     "        for _i, (model, score) in enumerate(zip(self.best_models, self.best_model_scores)):\n"
     '            if score["val_accuracy"] >= accuracy_percentile and \\\n'
     '                    score["val_recall"] >= recall_percentile and \\\n'
     '                    score["val_fp_per_hr"] <= fp_percentile:\n'
     "                models.append(model)\n"
     "                _cleared.append(_i)\n"
     "        # PATCHED: audit the checkpoint merge (improvement.md P1.4): the\n"
     "        # exported model is a weight-average of whatever cleared this gate,\n"
     "        # and which checkpoints that was previously invisible - a\n"
     "        # plausible source of the measured run-to-run variance. Print only.\n"
     "        # See patches/log-weight-and-merge.py.\n"
     '        _bounds = getattr(self, "_audit_seq_bounds", [0] * len(self.best_models))\n'
     "        for _i in _cleared:\n"
     "            _seq = 1 + sum(1 for _b in _bounds[1:] if _i >= _b)\n"
     '            _step = self.best_model_scores[_i]["training_step_ndx"]\n'
     '            print(f"# MERGE_AUDIT merged step={_step} seq={_seq}", flush=True)\n'
     '        _steps_csv = ",".join(str(self.best_model_scores[_i]["training_step_ndx"])'
     ' for _i in _cleared)\n'
     '        print(f"# MERGE_AUDIT cleared={len(models)}/{len(self.best_models)} "\n'
     '              f"percentile=90 steps={_steps_csv}", flush=True)\n'),
]

for anchor, _ in edits:
    n = content.count(anchor)
    if n != 1:
        first = anchor.strip().splitlines()[0]
        print(f"WARNING: patch target not found (expected exactly 1 of\n"
              f"{first!r}, found {n}) in {path} - patch not applied")
        sys.exit(0)

for anchor, replacement in edits:
    content = content.replace(anchor, replacement, 1)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched: {path} (weight doubling + checkpoint merge audited; "
      f"val_steps int16 -> int64)")
