#!/usr/bin/env python3
"""Per-clip score margins: WHICH clips set the matched-false-accept ceiling.

Why this exists. Every scorecard in this repo reports detection *at* a matched FA
budget and per-speaker counts, but never the thing between them: the interleaving of
positive peaks and adversarial peaks on one axis. A matched-FA detection figure says
that the threshold admitting the budget's worth of adversarial clips cuts off some
share of the positives; it does not say whether those positives sit one nudge below
the boundary or down near zero (a model that never saw the voice at all), nor which
handful of adversarial clips is holding the threshold up. Those two readings imply
different work - one is a positives/class-balance problem, the other is a near-miss-
negative problem - and no aggregate column distinguishes them.

The FA axis is extend+hey_other only, never pooled ('Never pool negative
categories'), and the threshold is chosen the way the ledger chooses it: the lowest
grid threshold whose adversarial FA is within budget, read on the model's OWN curve
('Never compare models at a fixed threshold').

    python3 tools/score_margins.py --model output/<word>/oww/<tag>.onnx \
        [--model output/<word>/mww/<tag>.json] [--adv-fa-budget N] [--csv out.csv]

Runs on the eval env (host: eval/.venv; container: the eval image, module form).
"""
import argparse
import math
import sys
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

import backends                      # noqa: E402  (eval/src is the module home on a host)
import eval_model as ev              # noqa: E402
import paths                         # noqa: E402

# The ceiling a ship call is made under: adversarial false accepts must stay strictly
# inside this share of the adversarial set. It is a policy constant, not a measurement,
# and it is what makes --adv-fa-budget's default derivable instead of a count that only
# happens to be right for one negative set's size.
ADV_FA_CONSTRAINT = 0.02


def default_budget(n_adv):
    """Largest fire count that stays STRICTLY inside ADV_FA_CONSTRAINT of the set.

    Derived, not hardcoded: a budget that is a count silently changes meaning whenever
    the adversarial set changes size, and a tool whose out-of-the-box reading sits on
    the wrong side of the constraint it exists to enforce is worse than one that refuses.

    The 1e-9 is not decoration. "Strictly inside" has to survive n * c landing on an
    integer: a set whose size makes the constraint an exact count (one fire per fifty
    clips at 2%) must NOT be allowed that count, and a binary float product can sit a
    ulp either side of the integer it means to be. Shaving an epsilon before the ceil
    makes both directions read the same way.
    """
    return max(0, math.ceil(n_adv * ADV_FA_CONSTRAINT - 1e-9) - 1)


def peaks_for(backend, clips, rng):
    """(name, peak) for each clip, through the same padding path eval_model uses."""
    out = []
    for name, data in clips:
        audio, _ = ev.with_noise_floor(data, ev.clip_rng(name))
        scores, _ = backend.score(audio)
        out.append((name, float(np.max(scores))))
    return out


def artifact_of(model):
    """The file whose bytes get loaded, plus its md5 prefix.

    A microWakeWord ``.json`` names its ``.tflite`` by relative path, so the pair can
    come from two different runs and the report will still print numbers - they will
    just belong to neither model. A mismatched pair is therefore worth failing on
    rather than printing. Returns (path, md5_8).

    A bare path into ``data/corpus/`` is refused for the same reason with a different
    mechanism: the corpus tree keeps ONE ``tflite_stream_state_internal_quant.tflite``
    per corpus, and every run built against that corpus writes its weights there, so
    the file is the last run's, not the one being asked about. Scoring from there makes
    two runs that differ in nothing but their bytes read as one model, and an arm
    judged against another arm's weights produces a verdict that is not merely noisy
    but backwards. The run dir is the only per-run copy, so it is the only place a
    scored model may come from.
    """
    path = Path(model)
    if path.suffix == ".json":
        named = json.loads(path.read_text()).get("model")
        if not named:
            raise SystemExit(f"{path}: manifest names no model")
        path = path.parent / named
    if not path.exists():
        raise SystemExit(f"model artifact named by {model} does not exist: {path}")
    corpus = (REPO_ROOT / "data" / "corpus").resolve()
    resolved = path.resolve()
    if resolved == corpus or corpus in resolved.parents:
        raise SystemExit(
            f"refusing to score {resolved}: it is inside the corpus tree, which holds "
            f"one shared copy of the weights per corpus (last run wins), so its bytes "
            f"belong to whichever run was converted last, not to the model being asked "
            f"about. Score the run-dir copy under output/<word>/<target>/<tag>/.")
    return path, hashlib.md5(path.read_bytes()).hexdigest()[:8]


def operating_threshold(adv_peaks, budget, grid=ev.SWEEP_GRID):
    """Lowest grid threshold with adversarial FA <= budget; None if the curve never gets there."""
    for t in grid:
        if sum(1 for p in adv_peaks if p >= t) <= budget:
            return t
    return None


def curve_header(speakers):
    """The CURVE table's header: one column per speaker, in holdout order."""
    return (f"    {'thr':>5}{'advFA':>8}{'pooled':>9}"
            + "".join(f"{s:>9}" for s in speakers) + "   (each speaker n fixed)")


def curve_cells(pos_peaks, speakers, g):
    """`k/n` at threshold g for EVERY speaker named - a hardcoded pair here is how a
    holdout speaker gets dropped from the one table built to show what the FA budget
    costs the voice with the fewest clips."""
    return "".join(
        f"{sum(1 for _, p in pos_peaks[s] if p >= g):>4}/{len(pos_peaks[s]):<4}"
        for s in speakers)


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
    artifact, digest = artifact_of(model)
    print("=" * 78)
    # The resolved artifact, not the argument: every mww manifest is named <word>.json
    # and every model inside a run dir is named stream_state_internal_quant.tflite, so
    # the basename cannot tell one run's model from another's.
    print(f"{model}  ->  {artifact}  md5 {digest}")
    print(backend.describe())
    if budget is None:
        budget = default_budget(len(adv))
        print(f"  budget derived: {budget} fire(s) = the largest count strictly inside "
              f"{ADV_FA_CONSTRAINT:.0%} of the {len(adv)}-clip adversarial set")
    # A separate flag, not `t is None`: the fallback below overwrites t, and reading
    # the budget back off the threshold is what made an unreachable curve look
    # matched in the first place.
    reachable = t is not None
    if not reachable:
        print(f"  curve never reaches adv FA <= {budget} on the grid; lowest grid point "
              f"reads {sum(1 for p in adv if p >= min(ev.SWEEP_GRID))} fires")
        t = min(ev.SWEEP_GRID)
    fired = sum(1 for p in adv if p >= t)
    print(f"  {'matched-FA' if reachable else 'BEST-AVAILABLE'} operating point: "
          f"threshold {t:.2f} at adv FA {fired}/{len(adv)} ({fired / len(adv):.1%}), "
          f"budget {budget}")
    if not reachable:
        # Unreachable-budget fallback: the detection figure below is read at an FA
        # the caller did not ask for. Say so next to it, not eight lines earlier.
        print("  !! NOT AT BUDGET - the detection figures below are at this FA, not "
              f"{budget / len(adv):.1%}. Do not quote them as matched-FA.")

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
    # EVERY speaker gets a column - the set comes from the holdout dirs, so a hardcoded
    # pair here quietly drops whichever voice the holdout has that the author's did not,
    # and the dropped one is the interesting one: the table exists to show what the
    # budget costs the speaker with the fewest clips ('never pool per-speaker results').
    print("\n  CURVE (adv = extend+hey_other only)")
    speakers = list(pos_peaks)
    print(curve_header(speakers))
    for g in ev.SWEEP_GRID:
        fa = sum(1 for p in adv if p >= g)
        det = sum(1 for p in all_pos if p >= g)
        cells = curve_cells(pos_peaks, speakers, g)
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
        # Key on the artifact, not the argument's basename: every mww model inside a
        # run dir is named stream_state_internal_quant.tflite, so a CSV keyed on that
        # cannot tell two runs apart. The digest is the tie-breaker that cannot lie.
        key = artifact.name
        if key == "stream_state_internal_quant.tflite":
            key = f"{artifact.parents[1].name}/{key}"
        key += f"#{digest}"
        with open(csv_path, "w") as fh:
            fh.write("model,set,clip,peak\n")
            for s, rows in pos_peaks.items():
                for n, p in rows:
                    fh.write(f"{key},pos/{s},{n},{p:.6f}\n")
            for c, rows in neg_peaks.items():
                for n, p in rows:
                    fh.write(f"{key},neg/{c},{n},{p:.6f}\n")
        print(f"\n  per-clip peaks written to {csv_path}")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable; .onnx (oww) or .json (mww manifest+weights)")
    ap.add_argument("--positives", nargs="+", default=None,
                    help="default: the held-out speaker dirs (same as eval_model.py)")
    ap.add_argument("--negatives", default=str(paths.NEGATIVES_DIR))
    ap.add_argument("--adv-fa-budget", type=int, default=None,
                    help="fires of extend+hey_other allowed (default: derived - the "
                         "largest count that stays strictly inside the repo's standing "
                         "FA constraint for the adversarial set actually loaded, so the "
                         "out-of-the-box reading is a matched-FA reading and stays one "
                         "when the negative set changes size)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sliding-window-size", type=int, default=None,
                    help="microWakeWord only, and only needed for a bare .tflite: "
                         "a cutoff means nothing without its window, so pass the "
                         "number the manifest says. A .json manifest carries it and "
                         "this overrides that.")
    ap.add_argument("--csv", default=None, help="dump per-clip peaks here (suffix .csv)")
    args = ap.parse_args()

    pos_dirs = [Path(d) for d in args.positives] if args.positives else paths.holdout_dirs(runon=False)
    for i, m in enumerate(args.model):
        report(m, pos_dirs, args.negatives, args.adv_fa_budget, args.top,
               csv_path=(f"{args.csv}.{i}" if args.csv and len(args.model) > 1
                         else args.csv), window=args.sliding_window_size)


if __name__ == "__main__":
    main()
