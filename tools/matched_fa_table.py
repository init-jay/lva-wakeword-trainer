#!/usr/bin/env python3
"""Read matched-false-accept per-speaker tables out of score_margins CSV dumps.

The companion to tools/score_margins.py: that tool runs a model once and writes every
clip's peak; this reads those CSVs back and answers the question a comparison
actually gets asked - at the threshold that admits N adversarial fires, what does
EACH speaker score? One row per model, each read on its OWN curve; the pooled
column (shown last) is not a verdict - pooling across speakers is how a voice
that fails gets averaged away.

    python3 tools/matched_fa_table.py /tmp/margins/*.csv [--budgets 6,12]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_model as ev   # noqa: E402
import score_margins as sm  # noqa: E402

# The ONE grid definition, imported from the sweep that records it
# (eval/src/eval_model.py:205): a local copy could drift and then this table
# would read thresholds the eval sweep never recorded - a number that does not
# exist. (eval/src is not a package, hence the sys.path insert, the same way
# score_margins.py reaches it.)
GRID = ev.SWEEP_GRID
# The adversarial set the same way score_margins.py defines its FA axis
# (ev.ADVERSARIAL at eval/src/eval_model.py:117). This file used to carry its
# own tuple ("extend", "hey_other"); the two agreed, so they are one definition now.
ADV = ev.ADVERSARIAL


def resolve_budgets(adv, explicit):
    """Budgets for one CSV: the parsed --budgets list, or the constraint-derived one.

    Reuses score_margins.default_budget / ADV_FA_CONSTRAINT instead of re-deriving
    the arithmetic a third time. The old literal default "6,12" was stale: a budget
    that is a count silently changes meaning when the adversarial set changes size,
    and for the 298-clip set the 2% constraint derives 5 (298 * 0.02 = 5.96, strictly
    inside -> 5), which neither 6 nor 12 is.
    """
    if explicit is not None:
        return explicit
    return [sm.default_budget(len(adv))]


def load(path):
    pos, adv = defaultdict(list), []
    model = Path(path).name.replace(".csv", "")
    for row in csv.reader(open(path)):
        if row[0] == "model" or len(row) < 4:
            continue
        _, setname, clip, peak = row[0], row[1], row[2], float(row[3])
        if setname.startswith("pos/"):
            pos[setname[4:]].append(peak)
        elif setname.startswith("neg/") and setname[4:] in ADV:
            adv.append(peak)
    return model, pos, adv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--budgets", default=None,
                    help="comma-separated counts of extend+hey_other fires allowed "
                         "(default: derived from the adversarial set in the files "
                         "given - score_margins.default_budget, the largest count "
                         "strictly inside the 2%% constraint)")
    args = ap.parse_args()
    explicit = None if args.budgets is None else [int(b) for b in args.budgets.split(",")]

    rows = []
    for p in args.csvs:
        model, pos, adv = load(p)
        budgets = resolve_budgets(adv, explicit)
        speakers = sorted(pos)
        allpos = [v for s in speakers for v in pos[s]]
        for b in budgets:
            t = next((g for g in GRID if sum(1 for a in adv if a >= g) <= b), None)
            if t is None:
                rows.append((model, b, None, None, {}))
                continue
            fires = sum(1 for a in adv if a >= t)
            rows.append((model, b, t, (fires, len(adv)),
                         {s: (sum(1 for v in pos[s] if v >= t), len(pos[s]))
                          for s in speakers},
                         sum(1 for v in allpos if v >= t), len(allpos)))

    speakers = sorted({s for r in rows for s in r[4]})
    hdr = f"  {'model':<30}{'FA budget':>10}{'thr':>6}{'adv FA':>10}"
    for s in speakers:
        hdr += f"{s:>10}"
    hdr += f"{'pooled':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model, b, t, fa, per, *pooled in rows:
        line = f"  {model[:29]:<30}{b:>7}/{fa[1] if fa else 0:<3}"
        if t is None:
            print(line + f"{'never reached':>6}{'':>10}" +
                  "".join(f"{'-':>10}" for _ in speakers) + f"{'-':>10}")
            continue
        line += f"{t:>6.2f}{fa[0]:>7}/{fa[1]:<3}"
        for s in speakers:
            k, n = per.get(s, (0, 0))
            line += f"{k:>4}/{n:<5}"
        line += f"{pooled[0]:>5}/{pooled[1]:<4}"
        print(line + f"   (t={t:.2f})")
    print("\n  Same column across rows = the same false-accept budget, each model read on")
    print("  its OWN curve (never at another's threshold). The pooled column is shown")
    print("  last and is not a verdict: it averages the voice that fails.")


if __name__ == "__main__":
    main()
