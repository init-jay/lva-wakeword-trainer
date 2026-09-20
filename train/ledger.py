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
        if not rec.get("eval_block"):
            no_eval.append(rec)
            continue
        combo = tuple(_fmt((rec.get("grid") or {}).get(k)) for k in grid_keys)
        groups.setdefault((rec.get("target"), combo), []).append(rec)

    lines = [f"ledger: {path}  ({len(records)} record(s)]", ""]
    for (target, combo), recs in sorted(groups.items(),
                                        key=lambda kv: (kv[0][0] or "", kv[0][1])):
        label = "  ".join(f"{k}={v}" for k, v in zip(grid_keys, combo)) or "-"
        adv = [r["eval_block"]["adversarial"]["rate"] * 100
               for r in recs
               if r["eval_block"].get("adversarial", {}).get("rate") is not None]
        det = [r["eval_block"]["positives"]["rate"] * 100
               for r in recs
               if r["eval_block"].get("positives", {}).get("rate") is not None]
        line = f"  {target or '?':<5} {label:<38} n={len(recs):<3}"
        line += f"  adv FA  {_stat(adv)}".rstrip()
        line += f"  detection  {_stat(det)}".rstrip()
        lines.append(line)
    # Records that carry no eval block show their config and tag only: there is
    # no number to put in the table, and pretending there was would be a lie.
    for rec in no_eval:
        label = "  ".join(f"{k}={_fmt((rec.get('grid') or {}).get(k))}"
                          for k in grid_keys) or "-"
        lines.append(f"  {rec.get('target') or '?':<5} {label:<38} "
                     f"(no eval)  {rec.get('tag', '?')}")
    if any(len(recs) > 1 for recs in groups.values()):
        lines.append("")
        lines.append("  [min-max] is the repeat-to-repeat spread. The noise floor in")
        lines.append("  this repo is 10 points, measured at an identical config (77% and")
        lines.append("  67% on the same holdout) - a difference inside that band is not")
        lines.append("  a result, no matter which side of it the mean lands on.")
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
