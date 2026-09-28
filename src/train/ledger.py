#!/usr/bin/env python3
"""The run ledger: output/<wake_word_safe>/runs.jsonl, one JSON object per line.

    python -m train.ledger --wake-word "<wake word>" [--show] [--grid-keys K [K...]]

One line per COMPLETED training run: target, the full run tag (code + corpus +
config; the name of the model directory), the corpus id (the manifest's short-7
identity - which frozen corpus this trained on), seed, the resolved config filed
as <tag>.config.json, wall time per stage, and the eval block, which is the
src/eval/src/eval_model.py --json output for this model VERBATIM - never
recomputed here, or the ledger becomes a second scorer that can drift. The grid
values a sweep varied sit in "grid".

APPEND-ONLY, NEVER REWRITTEN. record() refuses a second record with the same
(target, tag) and exits; the ledger is history, not a cache, so a
not-better result cannot be edited away. Re-measure as a NEW run (new seed or
config, new tag); the old record stays. Nothing here opens the file except one
append per record.

summarise() prints MIN/MAX across repeats beside the mean: a difference
smaller than the repeat-to-repeat spread is not a result.

Eval blocks also carry threshold_sweep: a fixed-grid re-threshold of that run's
own per-clip peaks (the model ran once; the threshold is a filter). summarise()
reads those curves to report detection at a COMMON matched-FA budget across
groups - the 40-eval manual 0.25-0.85 re-thresholding job is what this stops
happening.
"""

import argparse
import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The GIT root: data/ and output/ live there, not under src/.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Reserved record fields; record()'s **extra may not shadow them.
RESERVED = ("wake_word", "target", "tag", "corpus_id", "seed", "config",
            "wall_time", "eval_block")


def safe_name(wake_word):
    """Same <wake_word> -> directory-name mapping every other module uses."""
    return wake_word.replace(" ", "_").lower()


def ledger_path(wake_word):
    """output/<wake_word_safe>/runs.jsonl - beside output/<safe>/, never in data/."""
    return REPO_ROOT / "output" / safe_name(wake_word) / "runs.jsonl"


def load(wake_word):
    """Every record, in file order (= the order runs were filed). An absent file
    is an empty ledger, not an error: the first record creates it."""
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
    Keyed by target as well as tag: a tag's code and corpus halves do not
    encode the target, so the two trees' tags could collide."""
    out = {}
    for rec in load(wake_word):
        out.setdefault(rec.get("target"), {})[rec.get("tag")] = rec
    return out


def record(wake_word, *, target, tag, corpus_id, seed, config,
           wall_time=None, eval_block=None, **extra):
    """Append one record and return it. Refuses a duplicate (target, tag) and
    exits 1: the ledger is history, not a cache. Single append; nothing
    existing is rewritten."""
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
    """A single-element list as its element: mww files multi-value knobs as
    lists even when the grid passed a scalar, so 50000 and [50000] must
    compare equal or the drift warning fires on every mww record.
    Multi-element lists keep their shape: a genuinely different configuration.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


# Grid key (the trainer's CLI spelling, what the sweep's YAML says) -> the
# key the RESOLVED config files it under (oww create_config / mww build).
# Keyed by target: the two trainers file the same CLI option under different
# names (oww --training-steps -> config["steps"], mww -> config["training_steps"]
# as a list). Keys that map to no config key fall back to the label.
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
    The grid label is what the sweep SAID it would run; the filed resolved
    config is what it DID run, so a key present in the config wins - grouping
    stays correct through any label/config drift. A record with no resolved
    config (config: null) keeps its label.

    Shape exception: oww's --batch-n-per-class is the ACAV100M draw, but
    create_config files the whole per-class DICT - compare against the
    ACAV100M entry, or the label drifts on every record.
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
    """The tag's h-half: the resolved-config hash (src/train/provenance.py).
    Two records sharing the h-half AND the seed are the same run measured
    twice. Tags without an h-half (legacy d-format, smoke runs) have no
    config half to compare, so the whole tag is the identity and nothing
    can collapse.

    The h-half and seed are NOT the whole identity: the duplicate key below
    is (config-hash, seed, corpus_id), because a corpus-axes sweep holds the
    trainer fixed and redraws the TTS per arm - one (config, seed)
    legitimately spans two corpora whose evals differ by design.
    """
    parts = (rec.get("tag") or "").split("-")
    if parts and len(parts[-1]) > 1 and parts[-1].startswith("h"):
        return parts[-1]
    return rec.get("tag") or "?"


def _eval_signature(rec):
    """The headline eval numbers (adversarial rate, positives rate, threshold)
    to compare two records of one (config-hash, seed) pair. A pair whose
    signatures differ is a determinism regression: the same run, computed
    twice, disagreeing - the byte-identity bar has moved, and the table says
    so instead of averaging over it."""
    eb = rec.get("eval_block") or {}
    return ((eb.get("adversarial") or {}).get("rate"),
            (eb.get("positives") or {}).get("rate"),
            eb.get("threshold"))


def _sweep_curve(rec):
    """The record's eval threshold sweep as [(fa%, det%, threshold)], or None.
    The curve is a step function in the FA axis; every reading of it is a
    POINT PICK, never an interpolation. The FA axis is the extend+hey_other
    subset, never pooled with the other categories (the 'never pool'
    invariant applies to the curve as to the gate).

    No sweep (pre-sweep records) or a structurally inconsistent block
    (mismatched lengths, missing rates) returns None: the ledger does not
    repair or recompute the eval block - it says what it has and falls back
    to the one-threshold reading.
    """
    ts = (rec.get("eval_block") or {}).get("threshold_sweep") or {}
    thr, adv, det = (ts.get("thresholds"), ts.get("adv_rate"),
                     ts.get("positives_rate"))
    if not (thr and adv and det) or not (len(thr) == len(adv) == len(det)):
        return None
    if any(a is None or d is None for a, d in zip(adv, det)):
        return None
    return [(float(a) * 100, float(d) * 100, float(t))
            for t, a, d in zip(thr, adv, det)]


def _at_most_budget(curve, budget_pct):
    """(det%, fallback, fa%) - a curve's reading at a common FA budget B.

    'Detection at AT-MOST-B false accepts' = the max detection over the
    curve's points with FA <= B. A step function: a pick among measured
    points, never an interpolation - an interpolated point is a number no run
    ever produced. Ties in detection go to the point with FEWER false
    accepts - the higher threshold, the safer operating point of the two.
    No point meeting the budget (B below the curve's best FA) returns the
    best-reachable point with fallback=True: printed, marked, as a floor at
    its own FA rather than a reading at B.
    """
    eligible = [pt for pt in curve if pt[0] <= budget_pct]
    if not eligible:
        best_fa = min(pt[0] for pt in curve)
        det = max(pt[1] for pt in curve if pt[0] == best_fa)
        return det, True, best_fa
    fa, det, _t = max(eligible, key=lambda pt: (pt[1], -pt[0]))
    return det, False, fa


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
    any record's "grid" carries), the eval headline - adversarial
    false-accept rate and pooled detection - with MIN/MAX across repeats
    beside the mean: a difference smaller than the repeat-to-repeat spread is
    not a result. Records with no eval block show config and tag only.

    GROUP ON WHAT RAN: a key present in the record's resolved config is
    grouped on that value, not the grid label, and a label that disagrees
    warns on stderr by name. HONEST N: n counts distinct (config-hash, seed,
    corpus) triples - records sharing all three are the same run re-filed at
    another commit, not separate draws; agreeing duplicates collapse, and
    duplicates whose evals differ print as a determinism regression.
    THRESHOLD LABELS: columns carry the recorded eval threshold (@0.5, or
    `mixed`) - the rates are one-threshold readings, and a detection
    difference beside a different FA rate is the comparison CLAUDE.md
    forbids. MATCHED FA: when any record carries a threshold sweep the table
    gains a det@FA<=B column. B is ONE common budget for every swept group -
    the median across swept groups of each group's own median
    recorded-threshold FA, printed above the table with that derivation - and
    every group is read AT B: the best detection on its OWN curve with FA <=
    B, a step-function point pick, never an interpolation. A group whose
    curve cannot reach B prints its best-reachable point, marked '*'. When NO
    record carries a sweep the output is byte-identical to the pre-sweep
    table - the ledger holds both vintages, so the fallback path must not
    move a single character of the old reading.
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
            # The run used a different value than its label claims. Name it -
            # a warning without a tag is a rumour.
            if k in grid and has_cfg and _fmt(_unwrap(grid[k])) != _fmt(value):
                print(f"  WARNING: {rec.get('target')} {rec.get('tag')}: grid label "
                      f"{k}={_fmt(_unwrap(grid[k]))} but the run's resolved config has "
                      f"{k}={_fmt(value)} - grouped under the config value; the "
                      f"grid label is stale (the run used the config's value) - "
                      f"check the sweep that filed this record", file=sys.stderr)
            values.append(value)
        if not rec.get("eval_block"):
            no_eval.append((rec, values))
            continue
        combo = tuple(_fmt(v) for v in values)
        groups.setdefault((rec.get("target"), combo), []).append(rec)

    # A group whose records were trained on DIFFERENT frozen corpora
    # (more than one corpus_id) averages DATA, not seeds: part of its
    # [min-max] spread is TTS redraw, not repeat-to-repeat noise.
    for (target, combo), recs in sorted(groups.items(),
                                        key=lambda kv: (kv[0][0] or "", kv[0][1])):
        corpora = sorted({str(r.get("corpus_id")) for r in recs
                          if r.get("corpus_id") is not None})
        if len(corpora) > 1:
            label = "  ".join(f"{k}={v}" for k, v in zip(grid_keys, combo)) or "-"
            print(f"  WARNING: group {target or '?'} {label} spans "
                  f"{len(corpora)} corpus_id(s) {', '.join(corpora)} "
                  f"({len(recs)} record(s)) - cross-corpus rows must never be "
                  "silently averaged: part of the [min-max] spread is TTS "
                  "redraw, not seed noise (src/scripts/compare_arms.py is the "
                  "per-corpus view)",
                  file=sys.stderr)

    lines = [f"ledger: {path}  ({len(records)} record(s))", ""]

    # Per-group sweep curves from the records that carry one, with each
    # record's own recorded-threshold FA (its eval block's adversarial rate)
    # for the budget derivation. Groups without a sweep are absent here.
    group_sweep = {}
    for (target, combo), recs in groups.items():
        entries = []
        for r in recs:
            curve = _sweep_curve(r)
            if curve is None:
                continue
            fa0 = (r["eval_block"].get("adversarial") or {}).get("rate")
            entries.append((fa0 * 100 if fa0 is not None else None, curve))
        if entries:
            group_sweep[(target, combo)] = entries

    # The common FA budget, or None when no record carries a sweep. The
    # median of medians sits at the heart of the swept groups' operating
    # region: tight enough that the reading discriminates, derived from each
    # group's own measured point so no group is forced to a corner of its
    # curve.
    matched = None
    n_swept = 0
    if group_sweep:
        per_group = []
        for entries in group_sweep.values():
            fa0s = [f for f, _ in entries if f is not None]
            if fa0s:
                per_group.append(statistics.median(fa0s))
        n_swept = len(per_group)
        if per_group:
            matched = statistics.median(per_group)
    if matched is not None:
        lines += [
            f"  matched-FA budget: adv FA <= {matched:.1f}% - the median, across the",
            f"  {n_swept} swept group(s), of each group's own median adv FA at its recorded",
            "  threshold. Every swept group is read AT that budget: the best detection on its",
            "  own curve with FA <= B - a step-function point pick, never interpolated",
            "  (CLAUDE.md: never compare models at a fixed threshold).",
            "",
        ]
    fallback_notes = []
    for (target, combo), recs in sorted(groups.items(),
                                        key=lambda kv: (kv[0][0] or "", kv[0][1])):
        label = "  ".join(f"{k}={v}" for k, v in zip(grid_keys, combo)) or "-"
        # Distinct (config-hash, seed, corpus) triples are the samples; records
        # that share all three are the same run re-filed at another commit.
        # The corpus belongs in the key: the h-half is the config only, so two
        # corpora at one config+seed must not collapse into one "duplicated run".
        # Agreements collapse to one statistic; disagreements do not.
        pairs = {}
        for r in recs:
            pairs.setdefault((_config_hash(r), r.get("seed"), r.get("corpus_id")), []).append(r)
        n = len(pairs)
        n_runs = len(recs)
        adv, det = [], []
        pair_curves = {}
        for (chash, seed, corpus), dups in pairs.items():
            sigs = [_eval_signature(r) for r in dups]
            collapsed = len(sigs) <= 1 or len(set(sigs)) == 1
            if not collapsed:
                names = ("adversarial.rate", "positives.rate", "threshold")
                differing = [f"{name} ({' vs '.join(repr(s[i]) for s in sigs)})"
                             for name, i in zip(names, range(3))
                             if len({s[i] for s in sigs}) > 1]
                for a, b in itertools.combinations(dups, 2):
                    print(f"  DETERMINISM REGRESSION: {a.get('tag')} and "
                          f"{b.get('tag')} are the same (config, seed, corpus) run but their "
                          f"evals differ - {', '.join(differing)}. The byte-identity "
                          f"bar is not holding; the pair does not collapse, so both "
                          f"values stay in the row",
                          file=sys.stderr)
            for r in (dups[:1] if collapsed else dups):
                if (r["eval_block"].get("adversarial") or {}).get("rate") is not None:
                    adv.append(r["eval_block"]["adversarial"]["rate"] * 100)
                if (r["eval_block"].get("positives") or {}).get("rate") is not None:
                    det.append(r["eval_block"]["positives"]["rate"] * 100)
                curve = _sweep_curve(r)
                if curve is not None:
                    pair_curves.setdefault((chash, seed, corpus), []).append(curve)
        # Name the threshold the rates were read at, or say mixed when the
        # group's records do not agree on one.
        thresholds = {_fmt((r.get("eval_block") or {}).get("threshold")) for r in recs}
        th = thresholds.pop() if len(thresholds) == 1 else "mixed"
        n_str = f"n={n}" if n_runs == n else f"n={n} ({n_runs} runs)"
        line = f"  {target or '?':<5} {label:<38} {n_str}"
        line += f"  adv FA@{th}  {_stat(adv)}".rstrip()
        line += f"  detection@{th}  {_stat(det)}".rstrip()
        # The matched-FA column - only when some record carries a sweep, so a
        # ledger with none renders exactly as before, byte for byte.
        if matched is not None:
            vals, fbs = [], []
            for curves in pair_curves.values():
                for curve in curves:
                    v, fb, fa = _at_most_budget(curve, matched)
                    vals.append(v)
                    if fb:
                        fbs.append((fa, v))
            if vals:
                cell = f"  det@FA<={matched:.1f}%  {_stat(vals)}"
                if fbs:
                    cell += " *"
                    worst = min(fbs, key=lambda p: p[0])
                    fallback_notes.append(
                        f"* {target or '?'} {label}: no curve point at adv FA <= "
                        f"{matched:.1f}% (its best point is FA {worst[0]:.1f}%). The "
                        f"marked value ({worst[1]:.1f}%) is that best point, read at "
                        "its own FA - a floor, not a reading at the budget, and not "
                        "an interpolation.")
            else:
                cell = f"  det@FA<={matched:.1f}%  -  (no sweep on file; " \
                       "@-threshold reading only)"
            line += cell
        lines.append(line)
    if fallback_notes:
        lines.append("")
        lines += [f"  {note}" for note in fallback_notes]
    # No eval block: config and tag only - there is no number to put in the
    # table, and pretending there was would be a lie.
    for rec, values in no_eval:
        label = "  ".join(f"{k}={_fmt(v)}" for k, v in zip(grid_keys, values)) or "-"
        lines.append(f"  {rec.get('target') or '?':<5} {label:<38} "
                     f"(no eval)  {rec.get('tag', '?')}")
    if any(len(recs) > 1 for recs in groups.values()):
        lines.append("")
        lines.append("  [min-max] is the repeat-to-repeat spread. A difference inside")
        lines.append("  it is not a result, no matter which side of it the mean lands on.")
    if groups:
        lines.append("")
        lines.append("  The @ value is the threshold the column was READ at. These are")
        lines.append("  one-threshold readings, not a matched-FA comparison: a detection")
        lines.append("  difference between rows is not a verdict - 'Never compare models")
        lines.append("  at a fixed threshold' (CLAUDE.md).")
    if matched is not None:
        lines.append("")
        lines += [
            "  det@FA<=B: the matched-FA reading. One budget for every",
            "  swept group; each reads its OWN curve, never another group's.  '-': the",
            "  records predate the sweep - @-threshold columns only, not comparable at B.",
            "  '*': the group's curve never reaches FA <= B; the marked value is its best",
            "  reachable point, at its own FA - a floor, not a reading at the budget.",
        ]
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
