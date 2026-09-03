#!/usr/bin/env python3
"""Run a microWakeWord training pass and report where the model landed.

The openWakeWord equivalent is run-training.sh, and this carries over the two
lessons from it that cost the most:

  * WHETHER THE MODEL WAS WRITTEN IS THE REAL SIGNAL, not the exit code. A stale
    model was evaluated twice on the openWakeWord side before identical checksums
    gave it away, so the output is checksummed before and after.
  * AN EMPTY FEATURE SET IS SILENT. microwakeword/data.py logs "No spectrograms
    found in a configured feature set" and carries on, so a corpus that failed to
    build trains a model on nothing and only shows up as a bewildering evaluation.
    The config is checked before training starts.

    python -m mww.train --wake-word "hey seeree" \\
        --ambient data/mww_ambient/speech data/mww_ambient/no_speech

Everything after --  is passed through to microwakeword.model_train_eval.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# The REPO ROOT. This package sits at train/mww/ since the reorg, so the root is
# two levels up, not one - the old value now points at train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from train.mww import config as mww_config  # noqa: E402

# The quantized streaming model is the one that ships. model_train_eval writes up to
# four variants; only this one is a TFLite Micro streaming model with internal state,
# which is what ESPHome loads. Its flag defaults to 1 upstream, the others to 0.
SHIPPED = ("tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite")
ROC_FILE = "tflite_streaming_roc.txt"

# THE ARCHITECTURE IS AN ARGPARSE SUBCOMMAND, NOT A CONFIG KEY. model_train_eval
# registers `inception` and `mixednet` as subparsers and raises
# "Unknown model type: None" if neither is given - which is what a YAML-only
# invocation gets, because none of these values live in the YAML at all.
#
# `config["stride"]` and the derived spectrogram lengths are computed FROM these
# flags (model_train_eval.py:60-93), so the architecture and the feature geometry
# are set in the same place, and changing one silently changes the other.
#
# Values below are upstream's notebook defaults, kept verbatim as a starting point -
# they differ from mixednet.py's own argparse defaults, which are narrower
# (pointwise_filters "48, 48, 48, 48", kernels "[5], [9], [13], [21]", stride 1).
# Change one at a time and record it, tuning.md style.
MODEL = "mixednet"
MODEL_FLAGS = [
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1,1,1,1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    # FOUR entries, matching the other three lists. mixednet.model asserts all
    # four are the same length (mixednet.py:298-305), and upstream's own
    # argparse default is "0,0,0,0,0" against four pointwise filters - so the
    # bare defaults fail that assert too.
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]


# data.py:170-190 globs <features_dir>/<split>/**/*_mmap/ for exactly these splits.
# Anything outside them is invisible to the trainer, however many mmap directories it
# contains - which is why a laxer "**/*_mmap" check passes a set that then loads zero
# spectrograms.
SPLITS = ("training", "validation", "testing", "testing_ambient", "validation_ambient")


def check_mmap_set(d: Path):
    """Problems with one mmap feature set, phrased so the fix is obvious."""
    if not d.is_dir():
        return [f"{d} does not exist"]
    if any((d / split).glob("**/*_mmap") for split in SPLITS
           if (d / split).is_dir()):
        return []

    # Nothing under the split names. The usual cause is an extra directory level
    # from unzipping an archive that already had a top-level folder, so look for
    # somewhere below that IS shaped correctly and name it.
    for candidate in sorted(p for p in d.glob("**/") if p != d):
        if any((candidate / split).is_dir() and
               any((candidate / split).glob("**/*_mmap")) for split in SPLITS):
            return [f"{d} has no <split>/**/*_mmap - but {candidate} does. "
                    f"Point --ambient at that instead."]

    found = len(list(d.glob("**/*_mmap")))
    return [f"{d} has no {'/'.join(SPLITS[:3])}/... subdirectories containing "
            f"*_mmap ({found} *_mmap dirs elsewhere under it, which the trainer "
            f"cannot see)"]


def run_tag():
    """Name this run after the commit that produced it, as run-training.sh does."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        tag = out.stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
        return tag + ("-dirty" if dirty else "")
    except Exception:
        return datetime.now().strftime("%Y%m%d-%H%M%S")


def checksum(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--ambient", nargs="*", default=[],
                   help="RaggedMmap dirs from setup-mww-data.sh")
    p.add_argument("--corpus-root", default="my_custom_model")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="mww_models")
    p.add_argument("--training-steps", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=mww_config.DEFAULT_BATCH_SIZE)
    p.add_argument("--tag", default=None,
                   help="name for this run's output directory (default: the git "
                        "short commit, or a timestamp outside a repo). Each run "
                        "gets its own - model_train_eval refuses to train into an "
                        "existing directory.")
    p.add_argument("--force", action="store_true",
                   help="delete this run's output directory if it already exists")
    p.add_argument("--config", default=None,
                   help="use an existing YAML instead of generating one")
    p.add_argument("--model", default=MODEL, choices=("mixednet", "inception"),
                   help="architecture subcommand (default: %(default)s)")
    p.add_argument("--model-flags", nargs=argparse.REMAINDER, default=None,
                   help="override the architecture flags entirely; everything after "
                        "this is passed through verbatim")
    p.add_argument("passthrough", nargs="*", default=[],
                   help="extra args for model_train_eval, after --")
    args = p.parse_args()
    if args.model_flags is None:
        args.model_flags = MODEL_FLAGS if args.model == "mixednet" else []

    safe = args.wake_word.replace(" ", "_").lower()

    if args.config:
        config_path = Path(args.config)
        cfg = yaml.safe_load(config_path.read_text())
    else:
        corpus = Path(args.corpus_root) / safe / "mww" / "features"
        cfg = mww_config.build(
            args.wake_word, corpus / "positives", corpus / "negatives",
            args.ambient, args.output_dir, data_dir=args.data_dir,
            training_steps=args.training_steps, batch_size=args.batch_size,
            run_tag=args.tag or run_tag())
        # SIBLING OF train_dir, NOT INSIDE IT. model_train_eval calls
        # os.makedirs(train_dir) and fails if anything is there - a config file
        # written into it is enough to stop the run.
        train_dir = Path(cfg["train_dir"])
        config_path = train_dir.with_suffix(".yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {config_path}")

    # Check every feature set has something in it BEFORE spending a training run.
    # An empty one is a warning upstream, not an error.
    problems = []
    for fs in cfg["features"]:
        if fs["type"] == "clips":
            d = Path(fs["clips_settings"]["input_directory"])
            n = len(list(d.glob(fs["clips_settings"].get("file_pattern", "*.wav"))))
            if n == 0:
                problems.append(f"no clips in {d}")
        elif fs["type"] == "mmap":
            problems.extend(check_mmap_set(Path(fs["features_dir"])))
    # AMBIENT *EVALUATION* DATA IS WHAT MODEL SELECTION RUNS ON. The maximization
    # metric is average_viable_recall, computed from false accepts per hour on
    # validation_ambient. With no such data the metric is 0.000 at every step, the
    # "best" checkpoint never improves on anything, and the exported model is
    # whichever one happened to be current - while the ordinary accuracy/recall
    # numbers still look excellent. Training sets alone are not enough.
    ambient_eval = []
    for fs in cfg["features"]:
        if fs["type"] != "mmap":
            continue
        d = Path(fs["features_dir"])
        for split in ("validation_ambient", "testing_ambient"):
            if (d / split).is_dir() and any((d / split).glob("**/*_mmap")):
                ambient_eval.append(f"{d.name}/{split}")
    if not ambient_eval:
        problems.append(
            "no validation_ambient or testing_ambient data in any feature set. "
            "average_viable_recall will be 0.000 at every step and model selection "
            "will not work - the *_eval archives are the ones that carry these "
            "splits (e.g. data/mww_ambient/dinner_party_eval)")
    if problems:
        print("\nREFUSING TO TRAIN:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    # Check the int8 calibration constraint NOW. It is asserted after training
    # completes, so getting it wrong costs a full run and leaves a 0-byte model.
    flags = dict(zip([f.lstrip("-") for f in args.model_flags[::2]],
                     args.model_flags[1::2]))
    if args.model == "mixednet" and "stride" in flags:
        ok, length, message = mww_config.check_quantization_constraint(
            flags, cfg["clip_duration_ms"], cfg["window_step_ms"])
        print(f"  {message}")
        if not ok:
            sys.exit("\nREFUSING TO TRAIN: the run would complete and then fail "
                     "during TFLite conversion.")

    train_dir = Path(cfg["train_dir"])
    if train_dir.exists():
        if args.force:
            print(f"removing existing {train_dir}")
            shutil.rmtree(train_dir)
        else:
            sys.exit(f"\n{train_dir} already exists, and model_train_eval will not "
                     f"train into it.\nUse --tag NAME for a fresh directory, or "
                     f"--force to delete this one.")
    model_path = train_dir / SHIPPED[0] / SHIPPED[1]
    before = checksum(model_path)

    # The subcommand and its flags go LAST - argparse subparsers consume everything
    # after the subcommand name, so any top-level flag placed after `mixednet` would
    # be swallowed and then rejected as an unknown argument.
    cmd = ([sys.executable, "-m", "microwakeword.model_train_eval",
            "--training_config", str(config_path),
            "--train", "1",
            "--test_tflite_streaming_quantized", "1"]
           + list(args.passthrough) + [args.model] + args.model_flags)
    print("\n" + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)

    after = checksum(model_path)
    if after is None:
        sys.exit(f"\nTRAINING FAILED: {model_path} does not exist "
                 f"(model_train_eval exited {result.returncode})")
    # A FAILED CONVERSION LEAVES AN EMPTY FILE. TFLite opens the output before
    # converting, so a crash during quantization calibration leaves 0 bytes behind -
    # which existed, and had changed, and so passed both checks here until this was
    # added. It reported "DONE ... (0 KB, md5 d41d8cd9)", d41d8cd9 being the md5 of
    # nothing at all.
    size = model_path.stat().st_size
    if size < 1024:
        sys.exit(f"\nTRAINING FAILED: {model_path} is {size} bytes - the TFLite "
                 f"conversion did not produce a model (model_train_eval exited "
                 f"{result.returncode}). Look for the traceback above; a failure in "
                 f"quantization calibration is the usual cause.")
    if before is not None and before == after:
        sys.exit(f"\nTRAINING FAILED: {model_path} is unchanged from before this run - "
                 "it is the PREVIOUS model. Do not evaluate or deploy it.")

    size_kb = model_path.stat().st_size / 1024
    print(f"\nDONE  {model_path}  ({size_kb:.0f} KB, md5 {after[:8]})")

    roc = train_dir / SHIPPED[0] / ROC_FILE
    if roc.is_file():
        print(f"\nFalse accepts per hour vs cutoff: {roc}")
        print("  Pick probability_cutoff from THIS, not from a default. It is the")
        print("  same job as the threshold sweep on the openWakeWord side, and the")
        print("  ESPHome manifest needs the number.")
    else:
        print(f"\nNOTE: no {ROC_FILE} - rerun with --test_tflite_streaming_quantized 1")


if __name__ == "__main__":
    main()
