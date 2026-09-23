#!/usr/bin/env python3
"""Per-clip score margins: WHICH clips set the matched-false-accept ceiling.

Why this exists. Every scorecard in this repo reports detection *at* a matched FA
budget and per-speaker counts, but never the thing between them: the interleaving of
positive peaks and adversarial peaks on one axis. "det@FA<=2.3% = 83%" says the
threshold that admits <=6 of 298 adversarial clips cuts off 17% of the positives; it
does not say whether those 17% sit at 0.49 (one nudge from the boundary) or at 0.02
(the model never saw the voice at all), nor which handful of adversarial clips is
holding the threshold up. Those two readings imply different work - one is a
positives/class-balance problem, the other is a near-miss-negative problem - and no
aggregate column distinguishes them.

The FA axis is extend+hey_other only, never pooled ('Never pool negative
categories'), and the threshold is chosen the way the ledger chooses it: the lowest
grid threshold whose adversarial FA is within budget, read on the model's OWN curve
('Never compare models at a fixed threshold').

    python3 tools/score_margins.py --model output/hey_seeree/oww/<tag>.onnx \
        [--model output/hey_seeree/mww/<tag>.json] [--adv-fa-budget 6] [--csv out.csv]

Runs on the eval env (host: eval/.venv; container: the eval image, module form).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

import backends                      # noqa: E402  (eval/src is the module home on a host)
import eval_model as ev              # noqa: E402
import paths                         # noqa: E402


def peaks_for(backend, clips, rng):
    """(name, peak) for each clip, through the same padding path eval_model uses."""
    out = []
    for name, data in clips:
        audio, _ = ev.with_noise_floor(data, ev.clip_rng(name))
        scores, _ = backend.score(audio)
        out.append((name, float(np.max(scores))))
    return out


def operating_threshold(adv_peaks, budget, grid=ev.SWEEP_GRID):
    """Lowest grid threshold with adversarial FA <= budget; None if the curve never gets there."""
    for t in grid:
        if sum(1 for p in adv_peaks if p >= t) <= budget:
            return t
    return None


def report(model, pos_dirs, neg_dir, budget, top_n, csv_path=None, window=None):
    backend = backends.load(model, sliding_window_size=window)
    rng = np.random.default_rng(0)

    by_speaker, _ = ev.load_by_speaker([str(d) for d in pos_dirs])
    pos_peaks = {s: peaks_for(backend, clips, rng) for s, clips in by_speaker.items()}
    negatives, _ = ev.load_dir(neg_dir)
    by_cat = {}
    for name, data in negatives:
        match = ev.CATEGORY_RE.match(name)
        by_cat.setdefault(match.group(1) if match else "(uncategorised)", []).append(
            (name, data))
    neg_peaks = {c: peaks_for(backend, clips, rng) for c, clips in by_cat.items()}
    adv = [p for c in ev.ADVERSARIAL for _, p in neg_peaks.get(c, [])]
    adv_names = [(c, n, p) for c in ev.ADVERSARIAL for n, p in neg_peaks.get(c, [])]
    all_pos = [p for rows in pos_peaks.values() for _, p in rows]

    t = operating_threshold(adv, budget)
    print("=" * 78)
    print(Path(model).name)
    print(backend.describe())
    if t is None:
        print(f"  curve never reaches adv FA <= {budget} on the grid; lowest grid point "
              f"reads {sum(1 for p in adv if p >= min(ev.SWEEP_GRID))} fires")
        t = min(ev.SWEEP_GRID)
    fired = sum(1 for p in adv if p >= t)
    print(f"  matched-FA operating point: threshold {t:.2f} at adv FA {fired}/{len(adv)} "
          f"({fired / len(adv):.1%}), budget {budget}")

    print("\n  PER SPEAKER at that threshold  (peak = median of the clip peaks)")
    det_all = 0
    for s, rows in pos_peaks.items():
        hit = [p for _, p in rows if p >= t]
        miss = sorted([p for _, p in rows if p < t])
        det_all += len(hit)
        print(f"    {s:<8} {len(hit):>3}/{len(rows):<3} median {np.median([p for _, p in rows]):.3f}"
              f"   misses: {' '.join(f'{m:.2f}' for m in miss) or '-'}")
    print(f"    {'POOLED':<8} {det_all:>3}/{len(all_pos):<3}")

    # The blocking set: adversarial clips that keep the threshold from coming down.
    blockers = sorted([r for r in adv_names if r[2] >= t], key=lambda r: -r[2])
    print(f"\n  BLOCKERS (adversarial clips at or above the threshold)  n={len(blockers)}")
    for cat, name, p in blockers[:top_n]:
        print(f"    {cat:<11}{name[:40]:<42}{p:.3f}")
    # The whole trade on one screen: what the budget is actually costing, per speaker.
    print("\n  CURVE (adv = extend+hey_other only)")
    print(f"    {'thr':>5}{'advFA':>8}{'pooled':>9}{'jay':>8}{'jen':>8}   (each speaker n fixed)")
    for g in ev.SWEEP_GRID:
        fa = sum(1 for p in adv if p >= g)
        det = sum(1 for p in all_pos if p >= g)
        cells = ""
        for s in ("jay", "jen"):
            rows = pos_peaks.get(s)
            if rows:
                k = sum(1 for _, p in rows if p >= g)
                cells += f"{k:>4}/{len(rows):<4}"
        mark = " <- operating" if abs(g - t) < 1e-9 else ""
        print(f"    {g:>5.2f}{fa:>5}/{len(adv):<3}{det / len(all_pos):>8.0%}  {cells}{mark}")
    # What one more point of FA budget buys at the operating point - the price of
    # shaving a single blocker, read off the same curve.
    below = sorted([p for p in adv if p < t], reverse=True)
    if below:
        t2 = below[0]
        gained = sum(1 for p in all_pos if t2 <= p < t)
        print(f"\n  dropping the cheapest blocker (peak {t2:.3f}) would let the threshold "
              f"fall to {t2:.3f}\n  and win back {gained} positive(s) at the same FA budget.")
    # Margin of the positives that just miss, and where the nearest blocker sits.
    near = sorted([(p, s, n) for s, rows in pos_peaks.items() for n, p in rows if p < t],
                  reverse=True)[:top_n]
    print(f"\n  JUST BELOW the threshold (the cheapest positives to win back)")
    for p, s, n in near:
        print(f"    {s:<8}{n[:40]:<42}{p:.3f}")

    if csv_path:
        with open(csv_path, "w") as fh:
            fh.write("model,set,clip,peak\n")
            for s, rows in pos_peaks.items():
                for n, p in rows:
                    fh.write(f"{Path(model).name},pos/{s},{n},{p:.6f}\n")
            for c, rows in neg_peaks.items():
                for n, p in rows:
                    fh.write(f"{Path(model).name},neg/{c},{n},{p:.6f}\n")
        print(f"\n  per-clip peaks written to {csv_path}")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable; .onnx (oww) or .json (mww manifest+weights)")
    ap.add_argument("--positives", nargs="+", default=None,
                    help="default: the held-out speaker dirs (same as eval_model.py)")
    ap.add_argument("--negatives", default=str(paths.NEGATIVES_DIR))
    ap.add_argument("--adv-fa-budget", type=int, default=6,
                    help="fires of extend+hey_other allowed (default 6 = 2.0%% of 298, "
                         "the budget the current oww sweep verdicts were read at)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sliding-window-size", type=int, default=None,
                    help="microWakeWord only, and only needed for a bare .tflite: "
                         "a cutoff means nothing without its window, so pass the "
                         "number the manifest says (5 for this corpus). A .json "
                         "manifest carries it and this overrides that.")
    ap.add_argument("--csv", default=None, help="dump per-clip peaks here (suffix .csv)")
    args = ap.parse_args()

    pos_dirs = [Path(d) for d in args.positives] if args.positives else paths.holdout_dirs(runon=False)
    for i, m in enumerate(args.model):
        report(m, pos_dirs, args.negatives, args.adv_fa_budget, args.top,
               csv_path=(f"{args.csv}.{i}" if args.csv and len(args.model) > 1
                         else args.csv), window=args.sliding_window_size)


if __name__ == "__main__":
    main()
