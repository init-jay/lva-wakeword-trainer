#!/usr/bin/env python3
"""Cross-arm comparison over the run ledger, one grid key at a time.

    src/train/train-applesilicon/.venv/bin/python src/scripts/compare_arms.py \
        --wake-word "<wake word>" --grid-key real-vtlp [--ledger PATH]

Built for the corpus_axes sweeps: the sweep varies
the CORPUS, not the trainer, so its rows are not comparable by ledger's plain
summarise() - which groups on trainer config and would silently fold arms
whose configs are identical into one row. This tool keeps only the records
whose "grid" dict carries --grid-key and groups by the grid VALUE (the arm),
then prints, per arm:

  - corpus_id(s): a loud warning when one arm spans more than one corpus -
    cross-corpus rows must never be silently averaged;
  - per-speaker detection n/m with a Wilson CI (src/eval/src/eval_model.py's
    wilson_interval, imported - the ci95 values already in the ledger came
    from that exact function; per speaker, never pooled across speakers);
  - adversarial false accepts PER CATEGORY, extend and hey_other separately -
    the eval's ADVERSARIAL axis - never pooled with the other categories;
  - matched-FA detection: B is the SAME rule src/train/ledger.py's summarise
    prints - the median across swept arms of each arm's own median recorded-
    threshold adv FA, printed with that derivation. Each arm is read AT B on
    its OWN threshold_sweep curve via ledger._at_most_budget - a
    step-function point pick, never an interpolation; a curve that cannot
    reach B is marked '*' with its best-reachable point as a floor;
  - the voice_holdout_set rate when the eval block carries one, labelled
    'ranking signal, not a gate'.

The ledger's honesty rules are imported, not re-derived: _config_hash /
_eval_signature (n counts distinct (config-hash, seed) pairs; exact duplicates
collapse and print n=K (M runs); divergent duplicates print a DETERMINISM
WARNING naming both tags), _sweep_curve and _at_most_budget (point picks only). Records with no threshold_sweep
(every pre-sweep row) keep the labelled @-threshold reading and get
'- (no sweep on file)' in the matched cell - the mixed-vintage fallback
src/train/ledger.py carries, because the append-only ledger holds both vintages.

Read-only: nothing here opens the ledger for writing.
"""

import argparse
import itertools
import json
import statistics
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Import, don't reimplement: the ledger is the scorer of record (its
# docstring: a second copy of these semantics is how they drift). _fmt/_stat
# keep the table cells byte-compatible with `python -m train.ledger`.
from train import ledger  # noqa: E402

# src/eval/ has no __init__.py on the host: the eval IMAGE mounts src/eval/src as the
# package `eval` (src/eval/docker-compose.yml; tests/test_eval_stats.py replicates
# the mount with the same trick). Register it in-process so wilson_interval
# is imported rather than reimplemented - a second Wilson could drift from
# the ci95 values the eval block already recorded.
if "eval" not in sys.modules:
    _pkg = types.ModuleType("eval")
    _pkg.__path__ = [str(REPO_ROOT / "src" / "eval" / "src")]
    sys.modules["eval"] = _pkg
from eval.eval_model import wilson_interval  # noqa: E402

# The false-accept axis (src/eval/src/eval_model.py ADVERSARIAL): extend and
# hey_other, never pooled with the other negative categories (CLAUDE.md).
ADVERSARIAL = ("extend", "hey_other")


def _load(path):
    """Every record as a list of dicts in file order. Absent file -> [].

    Mirrors src/train/ledger.py load() for an arbitrary path - the --ledger flag
    is what makes this testable with synthetic files without ever touching
    the append-only history."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _arm_label(value):
    """The grid value as an arm name; the empty arm reads (none), not ''."""
    label = ledger._fmt(value)
    return label if label else "(none)"


def _arm_samples(recs):
    """(n_pairs, n_runs, samples, pair_tags) - the ledger's pair semantics, per arm.

    Two records sharing the tag's h-half (config hash), the seed AND the corpus are
    the same run computed at two commits: not two draws. Exact duplicates
    (identical eval signature) collapse to one sample; duplicates whose evals
    differ are a DETERMINISM WARNING naming both tags, and both values stay
    in the row - the pair does not collapse.

    corpus_id is part of the key because this tool is built for corpus_axes
    sweeps, where one (config, seed) legitimately spans two corpora and the two
    evals are EXPECTED to differ - that is the arm, not a regression. Keying on
    the pair alone shouted a determinism failure at the variable under test,
    which is the same false positive src/train/ledger.py widened its own key to
    stop; the two keys have to agree or this tool contradicts the ledger it
    reads.
    """
    pairs = {}
    for r in recs:
        pairs.setdefault(
            (ledger._config_hash(r), r.get("seed"), r.get("corpus_id")), []).append(r)
    samples = []
    pair_tags = {}
    for key, dups in pairs.items():
        pair_tags[key] = [r.get("tag", "?") for r in dups]
        sigs = [ledger._eval_signature(r) for r in dups]
        collapsed = len(sigs) <= 1 or len(set(sigs)) == 1
        if not collapsed:
            names = ("adversarial.rate", "positives.rate", "threshold")
            differing = [f"{name} ({' vs '.join(repr(s[i]) for s in sigs)})"
                         for name, i in zip(names, range(3))
                         if len({s[i] for s in sigs}) > 1]
            for a, b in itertools.combinations(dups, 2):
                print(f"  DETERMINISM WARNING: {a.get('tag')} and {b.get('tag')} "
                      f"are the same (config-hash, seed, corpus) run but their evals "
                      f"differ - {', '.join(differing)}. The pair does not "
                      "collapse; both values stay in the row - the "
                      "byte-identity bar is not holding",
                      file=sys.stderr)
        samples.extend(dups[:1] if collapsed else dups)
    return len(pairs), len(recs), samples, pair_tags


def _per_speaker(samples):
    """{speaker: (detected, n)} across the arm's samples, in ledger order.

    Sums within ONE speaker across the arm's repeats - the within-arm
    aggregation the footer says separates the lever from the redraw.
    Speakers stay separate: pooling across them is how a voice that fails
    gets averaged away."""
    out = {}
    for r in samples:
        for name, v in ((r.get("eval_block") or {}).get("per_speaker") or {}).items():
            agg = out.setdefault(name, [0, 0])
            agg[0] += (v.get("detected") or 0)
            agg[1] += (v.get("n") or 0)
    return out


def _categories(samples):
    """{category: (fired, n)} across the arm's samples, per category.

    The categories are printed separately (extend and hey_other are the
    axis); nothing here sums them - a pooled rate lets general's background
    dilute the extend signal (CLAUDE.md, never pool negative categories)."""
    out = {}
    for r in samples:
        for name, v in ((r.get("eval_block") or {})
                        .get("false_accepts_by_category") or {}).items():
            agg = out.setdefault(name, [0, 0])
            agg[0] += (v.get("fired") or 0)
            agg[1] += (v.get("n") or 0)
    return out


def render(records, grid_key, ledger_file):
    """The full report as a string. Warnings go to sys.stderr at call time."""
    kept = [(ledger._fmt((r.get("grid") or {}).get(grid_key)), r)
            for r in records if grid_key in (r.get("grid") or {})]
    lines = [f"compare-arms: {ledger_file}  ({len(records)} record(s); "
             f"grid key {grid_key!r} in {len(kept)})", ""]
    if not kept:
        return "\n".join(lines) + "  no records carry that grid key - nothing to compare"

    arms = {}
    for value, rec in kept:
        arms.setdefault(value, []).append(rec)

    # Per-arm structure: samples (C2-collapsed) plus each sample's sweep
    # curve and its recorded-threshold adv FA (the budget derivation).
    arm_info = {}
    for value in sorted(arms, key=lambda v: _arm_label(v)):
        recs = arms[value]
        n, n_runs, samples, _tags = _arm_samples(recs)
        corpora = sorted({str(r.get("corpus_id")) for r in recs})
        if len(corpora) > 1:
            print(f"  WARNING: arm {grid_key}={_arm_label(value)} spans {len(corpora)} "
                  f"corpus_id(s) {', '.join(corpora)} - cross-corpus rows must never be "
                  "silently averaged. The within-arm numbers below mix corpora; treat "
                  "them as one loud flag, not a statistic", file=sys.stderr)
        curves, fa0s = [], []
        for r in samples:
            curve = ledger._sweep_curve(r)
            if curve is not None:
                curves.append(curve)
            fa0 = (r.get("eval_block") or {}).get("adversarial") or {}
            if fa0.get("rate") is not None:
                fa0s.append(fa0["rate"] * 100)
        arm_info[value] = dict(recs=recs, n=n, n_runs=n_runs, samples=samples,
                               corpora=corpora, curves=curves, fa0s=fa0s)

    # The common FA budget: the SAME rule src/train/ledger.py summarise prints -
    # the median across swept arms of each arm's own median adv FA at its
    # recorded threshold. Read each arm AT that budget on its OWN curve.
    budget = None
    n_swept = 0
    swept = [v for v in arm_info if arm_info[v]["curves"]]
    if swept:
        per_arm = [statistics.median(arm_info[v]["fa0s"])
                   for v in swept if arm_info[v]["fa0s"]]
        n_swept = len(per_arm)
        if per_arm:
            budget = statistics.median(per_arm)
        if budget is not None:
            lines += [
                f"  matched-FA budget: adv FA <= {budget:.1f}% - the median, across the",
                f"  {n_swept} swept arm(s), of each arm's own median adv FA at its recorded",
                "  threshold. The rule src/train/ledger.py summarise prints.",
                "  Each arm is read AT that budget on its OWN curve - a step-function point",
                "  pick, never interpolated (CLAUDE.md: never compare models at a fixed threshold).",
                "",
            ]

    fallback_notes = []
    for value, info in arm_info.items():
        label = _arm_label(value)
        n_str = (f"n={info['n']} ({info['n_runs']} runs)"
                 if info["n_runs"] != info["n"] else f"n={info['n']}")
        lines.append(f"  ARM {grid_key}={label}   corpus {' + '.join(info['corpora'])}"
                     f"   {n_str}")
        tags = sorted({r.get("tag", "?") for r in info["recs"]})
        lines.append(f"    tags: {', '.join(tags)}")
        eb_any = any(r.get("eval_block") for r in info["recs"])
        if not eb_any:
            lines.append("    (no eval block on file for this arm)")
            continue

        # One-threshold readings, labelled with the recorded threshold (or
        # mixed): these are NOT a matched-FA comparison.
        thresholds = {ledger._fmt((r.get("eval_block") or {}).get("threshold"))
                      for r in info["recs"]}
        th = thresholds.pop() if len(thresholds) == 1 else "mixed"
        adv, det = [], []
        for r in info["samples"]:
            eb = r.get("eval_block") or {}
            if (eb.get("adversarial") or {}).get("rate") is not None:
                adv.append(eb["adversarial"]["rate"] * 100)
            if (eb.get("positives") or {}).get("rate") is not None:
                det.append(eb["positives"]["rate"] * 100)
        lines.append(f"    detection@{th}   {ledger._stat(det)}".rstrip())
        lines.append(f"    adv FA@{th}      {ledger._stat(adv)}".rstrip())

        # Adversarial false accepts PER CATEGORY - extend and hey_other are
        # the axis; the other categories are listed, never summed.
        cats = _categories(info["samples"])
        for name in ADVERSARIAL:
            if name in cats:
                fired, n = cats[name]
                lines.append(f"    FA {name:<10} {fired}/{n}  "
                             f"{(fired / n * 100 if n else float('nan')):.1f}%")
        for name in sorted(cats):
            if name in ADVERSARIAL:
                continue
            fired, n = cats[name]
            lines.append(f"    FA {name:<10} {fired}/{n}  "
                         f"{(fired / n * 100 if n else float('nan')):.1f}%  (context)")

        # Per speaker, with a Wilson CI - the interval exists because the
        # least-recorded speaker has few clips (one clip is a large share of
        # the rate; wilson_interval's docstring).
        lines.append("    per speaker (never pooled across speakers):")
        for name, (det_s, n_s) in sorted(_per_speaker(info["samples"]).items()):
            ci = wilson_interval(det_s, n_s)
            ci_s = (f"[{ci[0] * 100:.1f}-{ci[1] * 100:.1f}]" if ci else "-")
            rate_s = f"{det_s / n_s * 100:.1f}%" if n_s else "-"
            lines.append(f"      {name:<8} {det_s}/{n_s}  {rate_s}  {ci_s}")

        # voice_holdout_set: the synthetic TTS ranking set, when the eval
        # ran with the flag. A ranking signal for the sweep points - it never
        # stands in for the gates, which stay on the real held-out recordings.
        vhs = [v["rate"] * 100 for r in info["samples"]
               for v in [(r.get("eval_block") or {}).get("voice_holdout_set")]
               if v and v.get("rate") is not None]
        if vhs:
            lines.append(f"    voice_holdout  {ledger._stat(vhs)}   "
                         "(ranking signal, not a gate)")

        # The matched-FA cell, from this arm's OWN curve only.
        if info["curves"] and budget is not None:
            vals, fbs = [], []
            for curve in info["curves"]:
                v, fb, fa = ledger._at_most_budget(curve, budget)
                vals.append(v)
                if fb:
                    fbs.append((fa, v))
            cell = f"    det@FA<={budget:.1f}%   {ledger._stat(vals)}"
            if fbs:
                cell += " *"
                worst = min(fbs, key=lambda p: p[0])
                fallback_notes.append(
                    f"* {grid_key}={label}: no curve point at adv FA <= "
                    f"{budget:.1f}% (its best point is FA {worst[0]:.1f}%). The "
                    f"marked value ({worst[1]:.1f}%) is that best point, read at "
                    "its own FA - a floor, not a reading at the budget, and not "
                    "an interpolation.")
            lines.append(cell)
        else:
            # Mixed-vintage fallback (src/train/ledger.py): no sweep on file for
            # this arm - the @-threshold reading above is all it has.
            lines.append(f"    det@FA<={budget:.1f}%   -  (no sweep on file; "
                         "@-threshold reading only)"
                         if budget is not None else
                         "    det@FA<=B        -  (no sweep on file; "
                         "@-threshold reading only)")
        lines.append("")

    if fallback_notes:
        lines += [f"  {note}" for note in fallback_notes]
        lines.append("")

    # Footer: always printed - the sweep's whole honest-reading contract.
    lines += [
        "  ARMS DIFFER IN CORPUS IDENTITY: the sweep's corpus_axes rebuilds the TTS",
        "  corpus between arms, so a cross-arm gap carries",
        "  redraw noise on top of the lever. The within-arm repeats are what separate",
        "  the lever from the redraw; one repeat is not a result.",
        "  Run-to-run noise at these holdout sizes is wide enough that a small",
        "  per-speaker difference is not a result: two runs of an identical",
        "  config disagree at the same threshold, and that is the evidence.",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compare the arms of one grid key in the run ledger")
    p.add_argument("--wake-word", required=True)
    p.add_argument("--grid-key", required=True,
                   help="the key in each record's grid dict (e.g. real-vtlp)")
    p.add_argument("--ledger", default=None,
                   help="ledger file to read (default: output/<safe>/runs.jsonl; "
                        "a synthetic path makes the tool testable)")
    args = p.parse_args(argv)
    path = Path(args.ledger) if args.ledger else ledger.ledger_path(args.wake_word)
    print(render(_load(path), args.grid_key, path))


if __name__ == "__main__":
    main()
