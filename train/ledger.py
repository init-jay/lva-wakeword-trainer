#!/usr/bin/env python3
"""The run ledger: output/<wake_word_safe>/runs.jsonl, one JSON object per line.

    python -m train.ledger --wake-word "hey seeree" [--show] [--grid-keys K [K...]]

WHAT A RECORD IS. One line per COMPLETED training run: the target (oww/mww), the
full run tag (train/provenance.py: code + corpus + config, the same string the
model directory is named after), the corpus id (the manifest's short-7 identity,
so a record says WHICH frozen corpus it trained on), the seed, the resolved
config the trainer filed as <tag>.config.json, wall time per stage, and the
eval block. The grid values a sweep varied sit in "grid", so summarise() can
reconstruct the comparison the sweep was meant to answer.

APPEND-ONLY, NEVER REWRITTEN. A ledger you can rewrite is a ledger you can lie
in: the moment a "not better" result can be edited away, the record of which
settings were tried and what they did stops being usable as the answer to
"why do we still ship the one we ship". record() therefore refuses to append a
second record with the same (target, tag) and exits - the ledger is HISTORY,
not a cache. A run that should be re-measured is a NEW run with a new tag
(different seed or config); the old record stays, which is the whole point.
Nothing in this module opens the file for writing except one append per
record, and existing lines are never touched.

THE EVAL BLOCK IS VERBATIM, NOT RECOMPUTED. It is the JSON that
eval/src/eval_model.py --json wrote for exactly this model (same numbers,
same run of the scoring code). Recomputing any of it here - "for convenience"
- would make the ledger a second scorer that can drift from the first, and a
drifted scorer is how a 10-point difference (measured twice, at an IDENTICAL
config: 77% and 67%) stops being readable as noise. If the eval JSON moves,
the ledger says so by not having it, not by showing a different number.

SPREAD IS THE POINT. summarise() prints MIN/MAX across repeats beside the
mean, because a difference smaller than the repeat-to-repeat spread is not a
result - the 10-point measurement above is the noise floor this repo has
directly observed, at an unchanged configuration.
"""

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reserved record fields; record()'s **extra may not shadow them, or two
# writers could disagree about what a field means and the file would parse
# but no longer be honest.
RESERVED = ("wake_word", "target", "tag", "corpus_id", "seed", "config",
            "wall_time", "eval_block")


def safe_name(wake_word):
    """Same <wake_word> -> directory-name mapping every other module uses."""
    return wake_word.replace(" ", "_").lower()


def ledger_path(wake_word):
    """output/<wake_word_safe>/runs.jsonl - beside output/<safe>/, never in data/.

    data/ is inputs and generated corpus; models and their history live in
    output/. The trainers rmtree their own corpus every run but never this
    file, so a sweep interrupted mid-point still leaves what was filed intact.
    """
    return REPO_ROOT / "output" / safe_name(wake_word) / "runs.jsonl"


def load(wake_word):
    """Every record as a list of dicts, in file order (= the order runs were
    filed). An absent file is an empty ledger, not an error: the first record
    creates it."""
    path = ledger_path(wake_word)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def by_tag(wake_word):
    """{target: {tag: record}} - the resumability check the sweep runner uses.

    Keyed by target as well as tag: the oww and mww trees are separate, and a
    tag computed for one could in principle collide with the other's (the code
    and corpus halves do not encode the target)."""
    out = {}
    for rec in load(wake_word):
        out.setdefault(rec.get("target"), {})[rec.get("tag")] = rec
    return out


def record(wake_word, *, target, tag, corpus_id, seed, config,
           wall_time=None, eval_block=None, **extra):
    """Append one record and return it.

    Refuses a duplicate (target, tag) and exits 1: the ledger is history, not
    a cache (module docstring). The write is a single append; nothing existing
    is rewritten.
    """
    for key in extra:
        if key in RESERVED:
            sys.exit(f"record() extra key {key!r} shadows a reserved field - "
                     f"a field with two meanings is how a ledger stops being "
                     f"trustworthy")
    path = ledger_path(wake_word)
    for rec in load(wake_word):
        if rec.get("target") == target and rec.get("tag") == tag:
            sys.exit(f"ledger already has {target} / {tag} - refusing to append a "
                     f"second record for the same run. The ledger is history, not a "
                     f"cache: re-measure as a NEW run (new seed or config, new tag) "
                     f"and file that beside this one.")
    rec = {
        "wake_word": wake_word,
        "target": target,
        "tag": tag,
        "corpus_id": corpus_id,
        "seed": seed,
        "config": config,
        "wall_time": wall_time,          # {stage: seconds}; the sweep's timing
        "eval_block": eval_block,        # eval_model.py --json, verbatim
    }
    rec.update(extra)
    rec["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _fmt(value):
    """A grid value as a stable table cell: missing reads as '-', not 'None'."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _unwrap(value):
    """A single-element list as its element.

    mww files its multi-value knobs as lists (train/mww/config.py build():
    training_steps: [50000]) even when the sweep's grid passed a scalar, so
    the label 50000 and the config [50000] must compare equal or the drift
    warning (below) fires on every mww record. Multi-element lists keep
    their shape: that is a genuinely different configuration.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


# Grid key (the trainer's CLI spelling, what the sweep's YAML says) -> the
# key the RESOLVED config files it under (train/oww/train.py create_config /
# train/mww/config.py build). bug.md C1 (2026-09-22): the 094e414 sweep ran
# before the grid was threaded into the command, so four rows carried 25k
# labels around runs that were filed at 50k, and grouping on the label
# printed an 80.9% "25k" mean that was really a 25k/50k mix - contradicting
# the hand-built 73.5% verdict in improvement.md. The map is keyed by target
# because the two trainers file the same CLI option under different names:
# oww --training-steps lands in config["steps"] (train/oww/train.py:438),
# mww's in config["training_steps"] as a list (train/mww/config.py:222) -
# a map that is not target-aware is the same bug wearing a different coat.
# Keys that spell the same in both (e.g. lr) need no entry: the
# hyphen-to-underscore fallback in _resolved_value covers them, and a grid
# key that maps to no config key falls back to the label, as before.
GRID_TO_CONFIG_KEY = {
    "oww": {
        "training-steps": "steps",
        "layer-size": "layer_size",
        "max-negative-weight": "max_negative_weight",
        "augmentation-rounds": "augmentation_rounds",
        "augmentation-batch-size": "augmentation_batch_size",
        "batch-n-per-class": "batch_n_per_class",
        "target-fp-per-hour": "target_false_positives_per_hour",
        "target-accuracy": "target_accuracy",
        "target-recall": "target_recall",
        "n-samples-val": "n_samples_val",
    },
    "mww": {
        "training-steps": "training_steps",
        "batch-size": "batch_size",
        "learning-rates": "learning_rates",
        "positive-class-weight": "positive_class_weight",
        "negative-class-weight": "negative_class_weight",
        "eval-step-interval": "eval_step_interval",
        "positive-sampling-weight": "positive_sampling_weight",
        "positive-penalty-weight": "positive_penalty_weight",
        "negative-sampling-weight": "negative_sampling_weight",
        "negative-penalty-weight": "negative_penalty_weight",
        "ambient-sampling-weight": "ambient_sampling_weight",
        "ambient-penalty-weight": "ambient_penalty_weight",
        # ("model" needs no entry: identical in both; `lr` the same.)
        # model-flag is deliberately ABSENT: the trainer merges its K=V
        # into the model_flags flag LIST, so the grid label is not the
        # resolved value and comparing them would warn on every such
        # record. Unmapped, it falls back to the label (as before).
    },
}


def _resolved_value(rec, key):
    """What the run ACTUALLY used for grid key `key`: (value, has_config).

    The record's grid label is what the sweep SAID it would run; the
    resolved config it filed is what it DID run, so a value that exists in
    the config wins (C1: this makes the grouping self-correcting for any
    future label/config drift, not just the 094e414 rows). A record that
    filed no resolved config (config: null - predates the config half) has
    no better evidence than its label, and keeps it.

    One shape exception: oww's --batch-n-per-class is the ACAV100M draw, but
    create_config files the whole per-class DICT around it, so comparing
    the label 1024 against that dict would warn on every record - compare
    against the ACAV100M entry instead.
    """
    cfg = rec.get("config") or {}
    if rec.get("target") == "oww" and key == "batch-n-per-class":
        value = cfg.get("batch_n_per_class")
        if isinstance(value, dict):
            return value.get("ACAV100M_sample"), True
        if value is not None:
            return value, True
    cfg_key = (GRID_TO_CONFIG_KEY.get(rec.get("target") or {}, {})
               .get(key, key.replace("-", "_")))
    if cfg_key in cfg:
        return cfg[cfg_key], True
    return (rec.get("grid") or {}).get(key), False


def _config_hash(rec):
    """The tag's h-half: the resolved-config hash (train/provenance.py).

    bug.md C2 (2026-09-22): the 50k group printed n=4 when its four records
    were two (config-hash, seed) pairs, each run at two commits - h94736bd/
    1042 and h642a48d/1043, agreeing to the last digit across a week of
    code change, which is P0.1's determinism holding, not four draws from
    a distribution. Two records sharing the h-half AND the seed are the
    same run measured twice. Tags without an h-half (legacy d-format, smoke
    runs) have no config half to compare, so the whole tag is the identity
    and nothing can collapse.
    """
    parts = (rec.get("tag") or "").split("-")
    if parts and len(parts[-1]) > 1 and parts[-1].startswith("h"):
        return parts[-1]
    return rec.get("tag") or "?"


def _eval_signature(rec):
    """The headline eval numbers, to compare two records of one
    (config-hash, seed) pair. A pair whose signatures differ is a
    determinism regression (C2): the same run, computed twice, disagreeing
    - the byte-identity bar has moved, and the table says so instead of
    averaging over it."""
    eb = rec.get("eval_block") or {}
    return ((eb.get("adversarial") or {}).get("rate"),
            (eb.get("positives") or {}).get("rate"),
            eb.get("threshold"))


def _stat(rates):
    """mean [min-max] across repeats, in percent - the spread beside the point.

    One repeat is printed bare: a [x-x] column would advertise a spread the
    data does not contain."""
    if not rates:
        return ""
    lo, hi, mean = min(rates), max(rates), sum(rates) / len(rates)
    if len(rates) == 1:
        return f"{mean:.1f}%"
    return f"{mean:.1f}% [{lo:.1f}-{hi:.1f}]"


def summarise(wake_word, grid_keys=None):
    """A compact human table over the whole ledger, one row per configuration.

    For each distinct combination of the given config keys (default: the keys
    any record's "grid" carries), the eval headline - adversarial false-accept
    rate and pooled detection rate from each record's eval block - with
    MIN/MAX across repeats beside the mean: a difference smaller than the
    repeat-to-repeat spread is not a result (module docstring). Records that
    carry no eval block show their config and tag only.

    Three honesty rules, each the shape of an incident (bug.md, 2026-09-22):
    GROUP ON WHAT RAN (C1) - a key present in the record's resolved config is
    grouped on that value, not on the grid label, and a label that disagrees
    with it warns on stderr by name; the 094e414 rows were the 25k/50k mix
    that made this table contradict improvement.md. HONEST N (C2) - n counts
    distinct (config-hash, seed) pairs, because two records sharing both are
    the same run at two commits, not two draws; exact duplicates collapse,
    and duplicates whose evals differ are printed as a determinism regression
    rather than averaged over. THRESHOLD LABELS (C3 step 1) - the columns
    carry the recorded eval threshold (@0.5, or `mixed`), because the rates
    are one-threshold readings and a detection difference beside a different
    FA rate is the comparison CLAUDE.md forbids; a matched-FA comparison is
    step 2, which this table does not attempt.
    """
    path = ledger_path(wake_word)
    records = load(wake_word)
    if not records:
        return f"  no ledger at {path} - nothing has been filed yet"
    if grid_keys is None:
        grid_keys = sorted({k for r in records for k in (r.get("grid") or {})})

    groups = {}
    no_eval = []
    for rec in records:
        grid = rec.get("grid") or {}
        values = []
        for k in grid_keys:
            resolved, has_cfg = _resolved_value(rec, k)
            value = _unwrap(resolved)
            # C1: the run used a different value than its label claims. Name
            # it - a warning without a tag is a rumour.
            if k in grid and has_cfg and _fmt(_unwrap(grid[k])) != _fmt(value):
                print(f"  WARNING: {rec.get('target')} {rec.get('tag')}: grid label "
                      f"{k}={_fmt(_unwrap(grid[k]))} but the run's resolved config has "
                      f"{k}={_fmt(value)} - grouped under the config value (bug.md C1, "
                      f"2026-09-22: the 094e414 sweep ran before the grid was threaded "
                      f"into the command)", file=sys.stderr)
            values.append(value)
        if not rec.get("eval_block"):
            no_eval.append((rec, values))
            continue
        combo = tuple(_fmt(v) for v in values)
        groups.setdefault((rec.get("target"), combo), []).append(rec)

    lines = [f"ledger: {path}  ({len(records)} record(s))", ""]
    for (target, combo), recs in sorted(groups.items(),
                                        key=lambda kv: (kv[0][0] or "", kv[0][1])):
        label = "  ".join(f"{k}={v}" for k, v in zip(grid_keys, combo)) or "-"
        # C2: distinct (config-hash, seed) pairs are the samples; records
        # that share both are the same run re-computed at another commit.
        # Agreements collapse to one statistic; disagreements do not.
        pairs = {}
        for r in recs:
            pairs.setdefault((_config_hash(r), r.get("seed")), []).append(r)
        n = len(pairs)
        n_runs = len(recs)
        adv, det = [], []
        for (chash, seed), dups in pairs.items():
            sigs = [_eval_signature(r) for r in dups]
            collapsed = len(sigs) <= 1 or len(set(sigs)) == 1
            if not collapsed:
                names = ("adversarial.rate", "positives.rate", "threshold")
                differing = [f"{name} ({' vs '.join(repr(s[i]) for s in sigs)})"
                             for name, i in zip(names, range(3))
                             if len({s[i] for s in sigs}) > 1]
                for a, b in itertools.combinations(dups, 2):
                    print(f"  DETERMINISM REGRESSION: {a.get('tag')} and "
                          f"{b.get('tag')} are the same (config, seed) run but their "
                          f"evals differ - {', '.join(differing)}. P0.1's byte-identity "
                          f"bar is not holding; the pair does not collapse, so both "
                          f"values stay in the row (bug.md C2, 2026-09-22)",
                          file=sys.stderr)
            for r in (dups[:1] if collapsed else dups):
                if (r["eval_block"].get("adversarial") or {}).get("rate") is not None:
                    adv.append(r["eval_block"]["adversarial"]["rate"] * 100)
                if (r["eval_block"].get("positives") or {}).get("rate") is not None:
                    det.append(r["eval_block"]["positives"]["rate"] * 100)
        # C3 step 1: name the threshold the rates were read at, or say mixed
        # when the group's records do not agree on one.
        thresholds = {_fmt((r.get("eval_block") or {}).get("threshold")) for r in recs}
        th = thresholds.pop() if len(thresholds) == 1 else "mixed"
        n_str = f"n={n}" if n_runs == n else f"n={n} ({n_runs} runs)"
        line = f"  {target or '?':<5} {label:<38} {n_str}"
        line += f"  adv FA@{th}  {_stat(adv)}".rstrip()
        line += f"  detection@{th}  {_stat(det)}".rstrip()
        lines.append(line)
    # Records that carry no eval block show their config and tag only: there is
    # no number to put in the table, and pretending there was would be a lie.
    for rec, values in no_eval:
        label = "  ".join(f"{k}={_fmt(v)}" for k, v in zip(grid_keys, values)) or "-"
        lines.append(f"  {rec.get('target') or '?':<5} {label:<38} "
                     f"(no eval)  {rec.get('tag', '?')}")
    if any(len(recs) > 1 for recs in groups.values()):
        lines.append("")
        lines.append("  [min-max] is the repeat-to-repeat spread. The noise floor in")
        lines.append("  this repo is 10 points, measured at an identical config (77% and")
        lines.append("  67% on the same holdout) - a difference inside that band is not")
        lines.append("  a result, no matter which side of it the mean lands on.")
    if groups:
        lines.append("")
        lines.append("  The @ value is the threshold the column was READ at. These are")
        lines.append("  one-threshold readings, not a matched-FA comparison: a detection")
        lines.append("  difference between rows is not a verdict - 'Never compare models")
        lines.append("  at a fixed threshold' (CLAUDE.md); 77% vs 67% at 0.5 was the")
        lines.append("  SAME config (bug.md C3, 2026-09-22).")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Show the run ledger: one record per completed training run")
    p.add_argument("--wake-word", required=True)
    p.add_argument("--show", action="store_true",
                   help="print the summary table (also the default action)")
    p.add_argument("--grid-keys", nargs="*", default=None, metavar="KEY",
                   help="config keys to group by (default: the keys any record's "
                        "\"grid\" carries)")
    args = p.parse_args()
    print(summarise(args.wake_word, args.grid_keys))


if __name__ == "__main__":
    main()
