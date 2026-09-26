#!/usr/bin/env python3
"""
Compare trained wake-word models at MATCHED false-accept rates.

Comparing models at a fixed threshold is misleading. What varies between runs is largely where
the score distribution sits, not how well the model separates the classes - a
fixed-threshold comparison measures the operating point rather than the model.

Every comparison here therefore tunes the threshold per model to hit the same
false-accept count, and reports detection at that point. A model is better only if
it detects more at the same precision.

Negatives are read PER CATEGORY, never pooled: the corpus from generate_negatives.py
is adversarial by construction - a fifth of it is phrase-extending - so a pooled rate
is meaningless. `extend` and `hey_other` are the adversarial categories the matched
comparison is keyed on; the rest are ordinary speech and should stay near zero.

MODELS FROM BOTH TRAINERS CAN BE COMPARED HERE, and the matched-false-accept method
is what makes that legitimate. An openWakeWord score and a microWakeWord
sliding-window average are not the same quantity and share no threshold scale - but
"detection at the operating point that admits N adversarial false accepts" is the
same question asked of both, on the same corpus, through the same code. The
fixed-0.5 table is meaningless ACROSS backends as well as across runs.

Two things travel with a microWakeWord number and are printed with it: the sliding
window size, without which a cutoff means nothing, and the score resolution, because
an int8 output has 256 levels and the sweep goes to 0.01.

Usage, from the repo root:
    # every held-out speaker, plain and run-on found automatically
    python -m eval.compare_models --models output/hey_seeree/oww/*.onnx

    # or name the directories explicitly
    python -m eval.compare_models --models M \\
        --positives data/recordings/holdout/speaker1 \\
        --runon data/recordings/holdout/speaker1_runon

    # one model, with a threshold sweep for choosing a deployment operating point
    python -m eval.compare_models \\
        --models output/hey_seeree/oww/hey_seeree_705c23b.tflite --sweep

    # openWakeWord ship candidate against the microWakeWord model, on the Mac.
    # Pass the mWW .json, not its .tflite: the manifest carries the cutoff and the
    # sliding window, so scoring it puts those under test too.
    docker compose run --rm eval python -m eval.compare_models --models \\
        output/hey_seeree/oww/hey_seeree_705c23b.onnx \\
        output/hey_seeree/mww/hey_seeree_705c23b.json

    # the same, on the host env (src/scripts/setup-eval-host.sh) - plain-path form,
    # from the repo root: the `-m eval.X` module form is the image's mount
    src/eval/.venv/bin/python src/eval/src/compare_models.py --models M M2

POSITIVES MUST BE RECORDINGS THE MODEL HAS NOT TRAINED ON, which is why the defaults
come from `src/eval/src/paths.py` rather than being spelled out here: the trainer globs
data/recordings/samples/ recursively, so scoring against that tree reports training
accuracy - it overstated detection by ~10 points during this work. Passing a
directory inside samples/ anyway is warned about, not blocked.

Needs onnxruntime and an importable openwakeword for .onnx, plus a TFLite runtime
and pymicro-features for microWakeWord. The `eval` compose service carries all of
it and builds native on the Mac; the host env (src/eval/pyproject.toml) carries the
same pins.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the scoring path from eval_model.py so both tools agree exactly: same
# streaming, same noise-floor padding, same per-clip RNG seed.
# Runnable as `python src/eval/src/compare_models.py` as well as `python -m
# eval.compare_models` - the plain-path form is the host invocation
# (src/scripts/setup-eval-host.sh). The try/except is the same guard the other
# scripts here carry.
try:
    from eval import backends, eval_model as ev, paths
except ImportError:
    import backends, eval_model as ev, paths

ADVERSARIAL_PREFIXES = ("extend_", "hey_other_")
FA_POINTS = (2, 4, 6, 8, 10, 12)

# 1000 resamples of the adversarial clips: the set is small (32 clips, one clip is
# 3.1 points), so the matched-FA threshold is a noisy draw from it. Enough
# resamples that the percentile edges are stable to the 1% point; the rng is fixed
# (seed 0, below) so a comparison rerun reads the same intervals - the comparison
# is over models, not over the bootstrap.
N_BOOTSTRAP = 1000

# Spans both backends' useful ranges, which do not overlap much. openWakeWord's
# operating point sits LOW - tuning run 17 ships at 0.15 - while microWakeWord's
# scores are a sliding-window mean of an int8 output and saturate: measured, every
# adversarial negative and every ordinary one fires at 0.15 and below, so its usable
# range is 0.25 upwards. A sweep that stopped at 0.5 would show the mWW model only
# where it is already too permissive to deploy.
SWEEP_THRESHOLDS = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02, 0.01)


def peak_scores(backend, clips):
    """Highest streaming score per clip - what a detector would actually see."""
    return np.array([
        ev.stream(backend, ev.with_noise_floor(data, ev.clip_rng(clip_name))[0])[0].max()
        for clip_name, data in clips
    ])


def threshold_for_fa(adversarial, count):
    """Lowest threshold admitting exactly `count` adversarial false accepts."""
    ordered = np.sort(adversarial)[::-1]
    return float(ordered[count]) + 1e-6 if count < len(ordered) else 0.0


def detection_at_fa(positive_scores, adversarial_scores, count):
    """Total detection rate (summed over the positive sets) at `count` adversarial
    false accepts - the quantity the matched table prints."""
    thr = threshold_for_fa(adversarial_scores, count)
    return sum((v >= thr).mean() for v in positive_scores.values()) / len(positive_scores)


def bootstrap_diff_ci(pos_w, adv_w, pos_s, adv_s, count, rng, n_boot=N_BOOTSTRAP):
    """95% bootstrap CI on the matched-FA rate DIFFERENCE (model w minus model s),
    as (point, low, high).

    Resample the ADVERSARIAL clips with replacement (one draw per resample, shared
    by both models - paired, so the CI is on the difference, not on two separate
    wobbly numbers) and re-derive each model's operating point from the resample;
    the positives stay fixed - the noise being quantified is the one that lives in
    the small adversarial set, which is also what moves the threshold a model is
    read at. The rng is the caller's fixed seed, so a rerun reads the same
    intervals: the comparison is over models, not over the bootstrap.

    This is the mechanical form of the 'never compare at a fixed threshold' rule:
    two runs of an identical config have measured 10 points apart here, so a
    difference smaller than this interval is not a result.
    """
    m = len(adv_w)
    if count >= m:
        return None

    def diff(draw):
        if draw is None:
            draw = slice(None)
        thr_w = threshold_for_fa(adv_w[draw], count)
        thr_s = threshold_for_fa(adv_s[draw], count)
        rate_w = sum((v >= thr_w).mean() for v in pos_w.values()) / len(pos_w)
        rate_s = sum((v >= thr_s).mean() for v in pos_s.values()) / len(pos_s)
        return rate_w - rate_s

    boots = np.array([
        diff(np.array(rng.choices(range(m), k=m)))          # with replacement
        for _ in range(n_boot)
    ])
    return diff(None), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Compare wake-word models at matched false-accept rates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help=".onnx or .tflite models to compare")
    parser.add_argument("--positives", nargs="+", default=None,
                        help="Held-out clips of the phrase alone (default: every "
                             "speaker directory under data/recordings/holdout/ "
                             "that is not a _runon one)")
    parser.add_argument("--runon", nargs="+", default=None,
                        help="Held-out clips of the phrase running into a command "
                             "(default: the _runon directories under "
                             "data/recordings/holdout/)")
    parser.add_argument("--negatives", default=str(paths.NEGATIVES_DIR),
                        help="Corpus from generate_negatives.py (default: %(default)s)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also print a threshold sweep per model, for choosing a "
                             "deployment operating point")
    parser.add_argument("--per-speaker-fa", type=int, default=4,
                        help="Adversarial false-accept count the per-speaker table is "
                             "read at (default: %(default)s). One point, not a sweep: "
                             "it answers whether a speaker falls off, not where to "
                             "set the threshold")
    parser.add_argument("--label-width", type=int, default=22)
    parser.add_argument("--sliding-window-size", type=int, default=None,
                        help="microWakeWord only: probabilities averaged before "
                             "thresholding. FIX THIS PER COMPARISON and sweep the "
                             "cutoff - varying both makes the table a 2D surface "
                             "read as a line. Default: the manifest's value")
    parser.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                        help="Also write every number printed here as machine-readable "
                             "JSON to PATH (same values, for the run ledger - see "
                             "improvement.md P0.5). Does not change the report.")
    args = parser.parse_args()

    negatives, _ = ev.load_dir(args.negatives)
    if not negatives:
        print(f"No negatives in {args.negatives}; generate them with generate_negatives.py")
        sys.exit(1)
    adversarial = [(n, d) for n, d in negatives if n.startswith(ADVERSARIAL_PREFIXES)]
    ordinary = [(n, d) for n, d in negatives if not n.startswith(ADVERSARIAL_PREFIXES)]

    # Resolved separately so the two sets stay disjoint. Both loaders recurse, so a
    # single --positives pointed at the holdout root would swallow the _runon
    # directories too and report them as clean detections.
    plain_dirs = args.positives or [str(d) for d in paths.holdout_dirs(runon=False)]
    runon_dirs = args.runon or [str(d) for d in paths.holdout_dirs(runon=True)]
    paths.warn_if_trained_on(plain_dirs + runon_dirs)

    # `spans` records where each speaker's clips sit in the flat list, so the
    # per-speaker view below is slicing done after one scoring pass - never a second
    # pass, and so never able to disagree with the pooled row.
    sets, sources, spans = {}, {}, {}
    for key, dirs in (("plain", plain_dirs), ("run-on", runon_dirs)):
        clips, used, key_spans = [], [], []
        for path in dirs:
            if not Path(path).is_dir():
                continue
            found, _ = ev.load_dir(path)
            if found:
                key_spans.append(
                    (paths.speaker_label(path), len(clips), len(clips) + len(found)))
                clips.extend(found)
                used.append(path)
        if clips:
            sets[key] = clips
            sources[key] = used
            spans[key] = key_spans
    if not sets:
        print(f"No held-out positives found in {paths.describe(plain_dirs + runon_dirs)}. "
              f"These must be recordings the model did NOT train on - record them with "
              f"`record_samples.py --holdout --speaker NAME`, and see the module "
              f"docstring.")
        sys.exit(1)

    print("=" * 78)
    print(f"{len(args.models)} model(s) | "
          + " ".join(f"{k} {len(v)}" for k, v in sets.items())
          + f" | adversarial negatives {len(adversarial)}, ordinary {len(ordinary)}")
    # Which recordings produced the numbers. Worth a line: the defaults now span
    # every held-out speaker on disk, so the set changes as speakers are added.
    for key, used in sources.items():
        print(f"  {key:<7} {paths.describe(used)}")
    print("=" * 78)

    # Models are usually named <wake_word>_<commit>, so strip the shared prefix and
    # keep the START of what remains - the commit hash is how runs are referred to.
    stems = [Path(p).stem for p in args.models]
    prefix = len(os.path.commonprefix(stems)) if len(stems) > 1 else 0

    # Disambiguate labels. Comparing the .onnx and .tflite of one model - the check
    # that a conversion is faithful - gives both the same stem, and a colliding key
    # would silently drop one from every table.
    labels, seen = [], {}
    for path in args.models:
        base = (Path(path).stem[prefix:] or Path(path).stem)[:args.label_width]
        if base in seen or any(Path(o).stem == Path(path).stem and o != path
                               for o in args.models):
            base = f"{base}{Path(path).suffix}"[:args.label_width]
        while base in seen:
            seen[base] += 1
            base = f"{base}#{seen[base]}"[:args.label_width]
        seen[base] = 0
        labels.append(base)

    scores, described = {}, {}
    for path, label in zip(args.models, labels):
        backend = backends.load(path, sliding_window_size=args.sliding_window_size)
        described[label] = backend.describe()
        scores[label] = {k: peak_scores(backend, v) for k, v in sets.items()}
        scores[label]["adv"] = peak_scores(backend, adversarial)
        scores[label]["ord"] = peak_scores(backend, ordinary)

    # What each label actually is. Mixing backends in one table is supported and is
    # also the easiest way to read a number as if it meant the same thing twice.
    print()
    for label, description in described.items():
        print(f"  {label:<{args.label_width}}  {description}")

    # Reference only: this is the number that misleads, so it is labelled as such.
    print("\nAt the default threshold 0.5 (reference - do NOT compare on this):")
    header = f"  {'model':<{args.label_width}}" + "".join(f"{k:>10}" for k in sets) + f"{'adv FA':>9}"
    print(header)
    for label, s in scores.items():
        row = f"  {label:<{args.label_width}}"
        for k in sets:
            row += f"{(s[k] >= 0.5).mean():>9.0%} "
        row += f"{int((s['adv'] >= 0.5).sum()):>6}/{len(adversarial)}"
        print(row)

    print(f"\nAt MATCHED false-accept counts (threshold tuned per model) -- "
          f"{' / '.join(sets)}:")
    print(f"  {'adv FA':<9}" + "".join(f"{lbl:>{args.label_width + 4}}" for lbl in scores))
    for count in FA_POINTS:
        if count >= len(adversarial):
            continue
        row = f"  {f'{count}/{len(adversarial)}':<9}"
        for label, s in scores.items():
            thr = threshold_for_fa(s["adv"], count)
            cells = "/".join(f"{(s[k] >= thr).mean() * 100:.0f}" for k in sets)
            row += f"{cells:>{args.label_width + 4}}"
        print(row)

    # PER SPEAKER, at one matched point. Every row above is an average over speakers,
    # so a model that fails one voice and carries the rest reads as merely slightly
    # worse - the failure mode that produced the 24%/97% split in src/train/corpus/
    # augment.py. One FA count rather than all of them, because the question here is
    # "does any speaker fall off", not "where is the operating point".
    fa = args.per_speaker_fa
    if fa < len(adversarial) and any(len(v) > 1 for v in spans.values()):
        print(f"\nPER SPEAKER, at {fa}/{len(adversarial)} adversarial false accepts:")
        for key in sets:
            if len(spans[key]) < 2:
                continue
            print(f"  {key}")
            print(f"    {'speaker':<18}{'n':>4}"
                  + "".join(f"{lbl:>{args.label_width + 2}}" for lbl in scores))
            for speaker, start, end in spans[key]:
                row = f"    {speaker[:17]:<18}{end - start:>4}"
                for label, s in scores.items():
                    thr = threshold_for_fa(s["adv"], fa)
                    hits = int((s[key][start:end] >= thr).sum())
                    n = end - start
                    rate = hits / n
                    # The Wilson CI travels with the rate (wilson_interval says why):
                    # at n=6 one clip is 16.7 points, so a bare rate is noise.
                    ci = ev.wilson_interval(hits, n)
                    cell = f"{rate:.0%}"
                    if ci:
                        cell += f"[{ci[0]:.0%}-{ci[1]:.0%}]"
                    row += f"{cell:>{args.label_width + 2}}"
                print(row)

        # Name the spread rather than leaving it to be spotted in the table.
        for label, s in scores.items():
            for key in sets:
                if len(spans[key]) < 2:
                    continue
                thr = threshold_for_fa(s["adv"], fa)
                rates = [((s[key][a:b] >= thr).mean(), name) for name, a, b in spans[key]]
                low, high = min(rates), max(rates)
                if high[0] - low[0] >= 0.25:
                    print(f"\n  {label} on {key}: {low[1]} at {low[0]:.0%} vs "
                          f"{high[1]} at {high[0]:.0%} - a {(high[0] - low[0]) * 100:.0f}"
                          f" point spread across speakers, not a threshold problem.")

    if len(scores) > 1:
        fa_counts = [c for c in FA_POINTS if c < len(adversarial)]
        # One fixed rng for every pair: the draws are the same noise, which is what
        # makes the intervals comparable between pairs and between reruns.
        rng = random.Random(0)
        diff_cis = {}
        labels_list = list(scores)
        for count in fa_counts:
            pairs = {}
            # Each unordered pair once: (w, s) and (s, w) are the same test with the
            # sign flipped, and printing both would read as twice the evidence.
            for i, w in enumerate(labels_list):
                for s in labels_list[i + 1:]:
                    pairs[(w, s)] = bootstrap_diff_ci(
                        {k: scores[w][k] for k in sets}, scores[w]["adv"],
                        {k: scores[s][k] for k in sets}, scores[s]["adv"],
                        count, rng)
            diff_cis[count] = pairs

        print(f"\n95% BOOTSTRAP CI ON THE MATCHED-FA DIFFERENCE "
              f"({N_BOOTSTRAP} resamples of the {len(adversarial)} adversarial clips, "
              f"with replacement, fixed seed):")
        for count in fa_counts:
            for (w, s), ci in diff_cis[count].items():
                if ci is None:
                    continue
                point, lo, hi = ci
                flag = "" if lo > 0 or hi < 0 else "   (includes 0 - not a result)"
                # int(round(...)) before formatting: a hair-below-zero bound would
                # otherwise print "-0".
                print(f"  {f'{count}/{len(adversarial)}':<9} {w} - {s}:",
                      f"{point * 100:+.0f} pts, 95% CI {int(round(lo * 100)):+d}.."
                      f"{int(round(hi * 100)):+d}{flag}")

        best = {}
        for count in fa_counts:
            for label, s in scores.items():
                thr = threshold_for_fa(s["adv"], count)
                total = sum((s[k] >= thr).mean() for k in sets)
                best[label] = best.get(label, 0) + total
        winner = max(best, key=best.get)
        # The verdict is GATED on the bootstrap: 'better' is only printed if the
        # winner's lead over the next model excludes 0 at at least one matched
        # point. Two runs of an identical config have measured 10 points apart, so
        # an un-gated 'best' would be a coin flip dressed up as a ranking.
        decisive = 0
        for count in fa_counts:
            points = {l: detection_at_fa({k: scores[l][k] for k in sets},
                                         scores[l]["adv"], count) for l in scores}
            ordered = sorted(points, key=points.get, reverse=True)
            if ordered[0] != winner:
                continue
            # The pair is stored once per unordered pair, in either direction.
            ci = diff_cis[count].get((ordered[0], ordered[1])) or \
                diff_cis[count].get((ordered[1], ordered[0]))
            if ci is None:
                continue
            if ci[1] > 0 or ci[2] < 0:
                decisive += 1
        if decisive == 0:
            print(f"\n  NOT DISTINGUISHABLE: the 95% CI on the difference includes 0 at "
                  f"every matched point - do not call {winner} better.")
            print("  A difference of a few points is not meaningful - two runs of the same")
            print("  configuration have measured 10 points apart at a fixed threshold.")
        else:
            print(f"\n  Best across the matched points: {winner} "
                  f"(lead excludes 0 at {decisive} of {len(fa_counts)} points)")

    if args.sweep:
        for label, s in scores.items():
            print(f"\nThreshold sweep - {label}")
            print(f"  {'thr':>6}" + "".join(f"{k:>10}" for k in sets)
                  + f"{'adv FA':>10}{'ordinary':>11}")
            for thr in SWEEP_THRESHOLDS:
                row = f"  {thr:>6.2f}"
                for k in sets:
                    row += f"{(s[k] >= thr).mean():>10.0%}"
                row += f"{int((s['adv'] >= thr).sum()):>7}/{len(adversarial)}"
                row += f"{int((s['ord'] >= thr).sum()):>8}/{len(ordinary)}"
                print(row)
            print("  The 'ordinary' column is the realistic false-accept rate. It is only")
            print("  a few minutes of audio, while a wake word runs continuously - so")
            print("  validate a low threshold against a long recording of the deployment")
            print("  room before committing to it.")

    # --- machine-readable copy of the same numbers -------------------------------
    # --json is the hook the run ledger (improvement.md P0.5) reads: the values are
    # exactly what the report above printed, computed with the same functions.
    if args.json_path:
        counts = [c for c in FA_POINTS if c < len(adversarial)]
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": [
                {"path": str(p), "label": lbl, "backend": described[lbl]}
                for p, lbl in zip(args.models, labels)
            ],
            "positive_sets": {key: len(v) for key, v in sets.items()},
            "adversarial_negatives": len(adversarial),
            "ordinary_negatives": len(ordinary),
            "at_0p5": {
                lbl: {**{k: float((s[k] >= 0.5).mean()) for k in sets},
                      "adv_fa": int((s["adv"] >= 0.5).sum())}
                for lbl, s in scores.items()
            },
            "matched_fa": {
                str(c): {
                    lbl: {**{k: float((s[k] >= threshold_for_fa(s["adv"], c)).mean())
                            for k in sets},
                          "threshold": threshold_for_fa(s["adv"], c)}
                    for lbl, s in scores.items()
                }
                for c in counts
            },
            "per_speaker": (
                {
                    key: {
                        speaker: {
                            "n": end - start,
                            **{lbl: {
                                "detected": int((s[key][start:end]
                                                 >= threshold_for_fa(s["adv"], fa)).sum()),
                                "ci95": list(ev.wilson_interval(
                                    int((s[key][start:end]
                                         >= threshold_for_fa(s["adv"], fa)).sum()),
                                    end - start)),
                            } for lbl, s in scores.items()},
                        }
                        for speaker, start, end in spans[key]
                    }
                    for key in sets if len(spans[key]) >= 2
                }
                if fa < len(adversarial) else {}
            ),
            "bootstrap_diff_ci": {
                str(c): {f"{w} - {s}": {"diff": point,
                                       "ci95": [lo, hi]}
                         for (w, s), (point, lo, hi) in diff_cis[c].items()}
                for c in diff_cis
            } if len(scores) > 1 else {},
        }
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nJSON written to {path}")


if __name__ == "__main__":
    main()
