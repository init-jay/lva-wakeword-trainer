#!/usr/bin/env python3
"""
Compare trained wake-word models at MATCHED false-accept rates.

This exists because comparing models at a fixed threshold is misleading, and that
mistake cost several wrong conclusions during the tuning documented in tuning.md.
Two runs of an IDENTICAL configuration measured 77% and 67% on held-out run-on
speech at threshold 0.5, and both reached 95% at 8/32 false accepts. What varies
between training runs is largely where the score distribution sits, not how well the
model separates the classes - so a fixed-threshold comparison measures the operating
point rather than the model.

Every comparison here therefore tunes the threshold per model to hit the same
false-accept count, and reports detection at that point. A model is better only if it
detects more at the same precision.

Negatives are read PER CATEGORY, never pooled: the corpus from generate_negatives.py
is adversarial by construction - a fifth of it is phrase-extending - so a pooled rate
is meaningless. `extend` and `hey_other` are the adversarial categories the matched
comparison is keyed on; the rest are ordinary speech and should stay near zero.

MODELS FROM BOTH TRAINERS CAN BE COMPARED HERE, and the matched-false-accept method
is what makes that legitimate. An openWakeWord score and a microWakeWord
sliding-window average are not the same quantity and share no threshold scale - but
"detection at the operating point that admits N adversarial false accepts" is the
same question asked of both, on the same corpus, through the same code. Read the
matched table; the fixed-0.5 table above it is meaningless ACROSS backends as well as
across runs.

Two things travel with a microWakeWord number and are printed with it: the sliding
window size, without which a cutoff means nothing, and the score resolution, because
an int8 output has 256 levels and the sweep goes to 0.01.

Usage:
    python compare_models.py --models my_custom_model/hey_seeree/*.onnx \\
        --positives my_real_samples_holdout/speaker1 \\
        --runon my_real_samples_holdout/speaker1_runon \\
        --negatives negatives_tts

    # one model, with a threshold sweep for choosing a deployment operating point
    python compare_models.py --models my_custom_model/hey_seeree.tflite --sweep

    # openWakeWord ship candidate against the microWakeWord model, on the Mac
    docker compose run --rm eval python compare_models.py --models \\
        my_custom_model/hey_seeree/hey_seeree_d1bb9f4.onnx \\
        my_custom_model/hey_seeree/mww/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite

Positives MUST be recordings the model has not trained on. train.py trains on
everything under my_real_samples/, so scoring against that directory reports training
accuracy - it overstated detection by ~10 points during this work.

Needs onnxruntime and an importable openwakeword for .onnx, plus a TFLite runtime and
pymicro-features for microWakeWord. The `eval` compose service carries all of it and
builds native on the Mac.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the scoring path from eval_model.py so both tools agree exactly: same
# streaming, same noise-floor padding, same per-clip RNG seed.
from eval import backends, eval_model as ev

ADVERSARIAL_PREFIXES = ("extend_", "hey_other_")
FA_POINTS = (2, 4, 6, 8, 10, 12)

# Spans both backends' useful ranges, which do not overlap much. openWakeWord's
# operating point sits LOW - tuning.md ships run 17 at 0.15 - while microWakeWord's
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare wake-word models at matched false-accept rates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help=".onnx or .tflite models to compare")
    parser.add_argument("--positives", default="my_real_samples_holdout/speaker1",
                        help="Held-out clips of the phrase alone (default: %(default)s)")
    parser.add_argument("--runon", default="my_real_samples_holdout/speaker1_runon",
                        help="Held-out clips of the phrase running into a command")
    parser.add_argument("--negatives", default="negatives_tts",
                        help="Corpus from generate_negatives.py")
    parser.add_argument("--sweep", action="store_true",
                        help="Also print a threshold sweep per model, for choosing a "
                             "deployment operating point")
    parser.add_argument("--label-width", type=int, default=22)
    parser.add_argument("--sliding-window-size", type=int, default=None,
                        help="microWakeWord only: probabilities averaged before "
                             "thresholding. FIX THIS PER COMPARISON and sweep the "
                             "cutoff - varying both makes the table a 2D surface "
                             "read as a line. Default: the manifest's value")
    args = parser.parse_args()

    negatives, _ = ev.load_dir(args.negatives)
    if not negatives:
        print(f"No negatives in {args.negatives}; generate them with generate_negatives.py")
        sys.exit(1)
    adversarial = [(n, d) for n, d in negatives if n.startswith(ADVERSARIAL_PREFIXES)]
    ordinary = [(n, d) for n, d in negatives if not n.startswith(ADVERSARIAL_PREFIXES)]

    sets = {}
    for key, path in (("plain", args.positives), ("run-on", args.runon)):
        clips, _ = ev.load_dir(path) if Path(path).is_dir() else ([], 0)
        if clips:
            sets[key] = clips
    if not sets:
        print("No held-out positives found. These must be recordings made AFTER the "
              "model trained - see the module docstring.")
        sys.exit(1)

    print("=" * 78)
    print(f"{len(args.models)} model(s) | "
          + " ".join(f"{k} {len(v)}" for k, v in sets.items())
          + f" | adversarial negatives {len(adversarial)}, ordinary {len(ordinary)}")
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

    if len(scores) > 1:
        best = {}
        for count in FA_POINTS:
            if count >= len(adversarial):
                continue
            for label, s in scores.items():
                thr = threshold_for_fa(s["adv"], count)
                total = sum((s[k] >= thr).mean() for k in sets)
                best[label] = best.get(label, 0) + total
        winner = max(best, key=best.get)
        print(f"\n  Best across the matched points: {winner}")
        print("  A difference of a few points is not meaningful - two runs of the same")
        print("  configuration have measured 10 points apart at a fixed threshold.")

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


if __name__ == "__main__":
    main()
