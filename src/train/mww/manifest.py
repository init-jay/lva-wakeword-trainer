#!/usr/bin/env python3
"""Emit the ESPHome manifest for a trained microWakeWord model.

microWakeWord produces only the .tflite. ESPHome needs a JSON manifest beside it,
and the values in it are not cosmetic - two of them change detection behaviour and
one has to match how the model was trained.

    python -m train.mww.manifest --wake-word "hey seeree" --run 20260902-063606

THE CUTOFF COMES FROM THE MEASUREMENT, NOT FROM A DEFAULT. Training writes
tflite_streaming_roc.txt: false rejection rate and false accepts per hour at every
cutoff. Picking a number without reading it is the same mistake as evaluating an
openWakeWord model at threshold 0.5, which the tuning log opens by warning about - two
runs of one configuration read 77% and 67% there.

    Cutoff 0.86: frr=0.0245; faph=0.000     <- zero false accepts, 97.6% recall
    Cutoff 0.70: frr=0.0168; faph=0.187
    Cutoff 0.60: frr=0.0116; faph=0.281

`--max-faph` picks the lowest cutoff meeting a false-accepts-per-hour budget, which
maximises recall subject to that budget. Default 0.0, the most conservative choice
the measurement supports - but see `choose_cutoff` for why that default nearly always
lands on a synthetic row, and shipped one manifest that could not fire.

AND THE ROC IS A GUIDE, NOT THE MEASUREMENT. Two reasons to confirm the cutoff with
`src/eval/src/backends.py` against held-out recordings before deploying it:

  * The ROC is scored on the ambient evaluation sets, not on this repo's adversarial
    negatives. `extend` false accepts - the failure mode unsolved since run 6 - are
    not in it at all.
  * The training repo dequantizes probabilities with a hardcoded 1/255
    (microwakeword/inference.py); the deployment runtime uses the output tensor's
    own scale, 1/256 here. A cutoff read off this table is ~0.4% away from the number
    the device compares against.

SLIDING_WINDOW_SIZE AND THE CUTOFF ARE A PAIR. The ROC is computed with a sliding
window average of 5 (microwakeword/test.py:301), so a cutoff read off that table is
only valid at `sliding_window_size: 5`. Changing one without re-deriving the other
silently moves the operating point.

FEATURE_STEP_SIZE MUST MATCH TRAINING. It is how often ESPHome's preprocessor emits
a feature vector, and the model was trained on features at `window_step_ms`. A
mismatch feeds the model a different time base than it learned - it will not error,
it will just detect badly. Emitted from mww.config.WINDOW_STEP_MS so the two cannot
drift: change the training step and the manifest follows.

TENSOR_ARENA_SIZE IS A GUESS AND HAS TO BE CHECKED ON DEVICE. It is the working
memory TFLite Micro allocates, and it cannot be computed here - it depends on the
arena planner in the ESPHome build. Too small and ESPHome fails to allocate at boot
with a clear error, so the failure is loud. Start with the default and raise it if
the device complains.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# The import root src/. This package sits at src/train/mww/, two levels
# below it, so src/ is two levels up, not the git root the pre-reorg value
# (parents[2]) points at.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.mww import config as mww_config  # noqa: E402

QUANT_DIR = "tflite_stream_state_internal_quant"
MODEL_FILE = "stream_state_internal_quant.tflite"
ROC_FILE = "tflite_streaming_roc.txt"

# microwakeword/test.py:301 - the ROC is computed with this window, so a cutoff
# taken from it is only meaningful alongside the same value here.
SLIDING_WINDOW_SIZE = 5

# Not computable from here; see the module docstring. ESPHome fails loudly at boot
# if it is too small.
DEFAULT_TENSOR_ARENA_SIZE = 30000
MINIMUM_ESPHOME_VERSION = "2024.7.0"

# Attribution, shown by ESPHome and by Home Assistant's wake-word picker. Defaults
# rather than flags because a manifest without them is an anonymous model, and the
# one thing nobody remembers to pass is the one that identifies who built it.
DEFAULT_AUTHOR = "init-jay"
DEFAULT_WEBSITE = "https://github.com/init-jay/lva-wakeword-trainer"

ROC_LINE = re.compile(r"Cutoff\s+([\d.]+)\s*:\s*frr\s*=\s*([\d.]+)\s*;\s*faph\s*=\s*([\d.]+)")


def parse_roc(path: Path):
    """[(cutoff, frr, faph), ...] from tflite_streaming_roc.txt."""
    rows = []
    for line in path.read_text().splitlines():
        m = ROC_LINE.search(line)
        if m:
            rows.append(tuple(float(g) for g in m.groups()))
    return rows


def choose_cutoff(rows, max_faph):
    """Lowest cutoff whose faph is within budget - i.e. best recall for that budget.

    Lower cutoff means more detections, so more false accepts AND fewer false
    rejects. Walking up from the lowest cutoff and taking the first that fits the
    budget gives the most sensitive operating point that still meets it.

    ROWS WITH frr == 1.0 ARE NOT OPERATING POINTS AND ARE DISCARDED FIRST. When no
    measured cutoff reaches the faph floor, microwakeword's generate_roc_curve
    (test.py:192-196) APPENDS A SYNTHETIC POINT at (faph 0, frr 1) to close the
    curve. It is a plotting terminator - a model that rejects everything - and it is
    the only row that satisfies the default --max-faph 0.0, so it won every time.
    That is how `hey_seeree.json` shipped with probability_cutoff 1.0, a manifest
    whose model cannot fire; measured on held-out recordings, the same model detects
    97% of one speaker's clips at 0.5. Nothing downstream could have caught it: a cutoff of
    1.0 is a legal value and ESPHome loads it without complaint.
    """
    real = [r for r in rows if r[1] < 1.0]
    if not real:
        raise SystemExit(
            f"{len(rows)} cutoffs in the ROC and every one has frr 1.0 - the model "
            f"detects nothing at any threshold. Do not ship this.")

    eligible = [r for r in real if r[2] <= max_faph]
    if not eligible:
        best = min(real, key=lambda r: r[2])
        raise SystemExit(
            f"no cutoff achieves faph <= {max_faph} while detecting anything. The "
            f"lowest measured is {best[2]} at cutoff {best[0]} (frr {best[1]}). "
            f"Raise --max-faph, or treat this as the model not being good enough "
            f"to deploy.")
    return min(eligible, key=lambda r: r[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--run", required=True, help="run directory name under the model dir")
    p.add_argument("--models-dir", default="output")
    p.add_argument("--max-faph", type=float, default=0.0,
                   help="false accepts per hour budget (default: %(default)s)")
    p.add_argument("--tensor-arena-size", type=int, default=DEFAULT_TENSOR_ARENA_SIZE)
    p.add_argument("--author", default=DEFAULT_AUTHOR)
    p.add_argument("--website", default=DEFAULT_WEBSITE)
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--languages", default="en")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    run_dir = Path(args.models_dir) / safe / "mww" / args.run / QUANT_DIR
    model = run_dir / MODEL_FILE
    roc = run_dir / ROC_FILE

    for path in (model, roc):
        if not path.is_file():
            sys.exit(f"not found: {path}")
    if model.stat().st_size < 1024:
        sys.exit(f"{model} is {model.stat().st_size} bytes - not a usable model")

    rows = parse_roc(roc)
    if not rows:
        sys.exit(f"no 'Cutoff ...: frr=...; faph=...' lines in {roc}")

    cutoff, frr, faph = choose_cutoff(rows, args.max_faph)
    print(f"{len(rows)} cutoffs measured; {roc}")
    print(f"  chosen cutoff {cutoff}: {100 * (1 - frr):.2f}% recall, "
          f"{faph} false accepts/hour (budget {args.max_faph})")

    manifest = {
        "type": "micro",
        "wake_word": args.wake_word,
        "author": args.author,
        "website": args.website,
        "model": model.name,
        "trained_languages": args.languages.split(","),
        "version": args.version,
        "micro": {
            "probability_cutoff": cutoff,
            # Must equal the window_step_ms the model was TRAINED with.
            "feature_step_size": mww_config.WINDOW_STEP_MS,
            # Must equal the window the ROC above was computed with.
            "sliding_window_size": SLIDING_WINDOW_SIZE,
            "tensor_arena_size": args.tensor_arena_size,
            "minimum_esphome_version": MINIMUM_ESPHOME_VERSION,
        },
    }
    out = run_dir / f"{safe}.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"\nDeploy both, side by side:\n  {model}\n  {out}")
    print(f"\ntensor_arena_size is {args.tensor_arena_size}, which is a starting "
          f"guess.\nIf ESPHome fails to allocate at boot, raise it and reflash.")


if __name__ == "__main__":
    main()
