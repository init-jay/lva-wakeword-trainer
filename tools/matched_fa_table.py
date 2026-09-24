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
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = [round(0.05 * i, 2) for i in range(1, 20)]
ADV = ("extend", "hey_other")


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
    ap.add_argument("--budgets", default="6,12",
                    help="comma-separated counts of extend+hey_other fires allowed")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    rows = []
    for p in args.csvs:
        model, pos, adv = load(p)
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
