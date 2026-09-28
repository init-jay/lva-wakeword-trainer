#!/usr/bin/env python3
"""Run a microWakeWord training pass and report where the model landed.

Wrapped by src/scripts/run-mww-training.sh, which chains the four stages; the
openWakeWord equivalent is run-oww-training.sh. Two lessons carried over from it:

  * WHETHER THE MODEL WAS WRITTEN IS THE REAL SIGNAL, not the exit code - so the
    output is checksummed before and after.
  * AN EMPTY FEATURE SET IS SILENT. microwakeword/data.py logs "No spectrograms
    found in a configured feature set" and carries on, so a corpus that failed
    to build trains a model on nothing - the config is checked first.

    python -m train.mww.train --wake-word "<wake word>" \\
        --ambient data/external/mww_ambient/speech \\
                  data/external/mww_ambient/no_speech

Everything after --  is passed through to microwakeword.model_train_eval.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# The import root src/: this package sits at src/train/mww/, so src/ is two
# levels up from this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from train import ownership, provenance  # noqa: E402
from train.mww import config as mww_config  # noqa: E402

# The quantized streaming model is the one that ships: only this variant is a
# TFLite Micro streaming model with internal state, which is what ESPHome loads.
SHIPPED = ("tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite")
ROC_FILE = "tflite_streaming_roc.txt"

# THE ARCHITECTURE IS AN ARGPARSE SUBCOMMAND, NOT A CONFIG KEY. model_train_eval
# registers `inception` and `mixednet` as subparsers and raises
# "Unknown model type: None" if neither is given - which is what a YAML-only
# invocation gets, because none of these values live in the YAML at all.
#
# `config["stride"]` and the derived spectrogram lengths are computed FROM these
# flags (model_train_eval.py:60-93), so changing one silently changes the other.
#
# Values below are upstream's notebook defaults, kept verbatim as a starting
# point - they differ from mixednet.py's own narrower argparse defaults. Change
# one at a time and record it.
MODEL = "mixednet"
MODEL_FLAGS = [
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1,1,1,1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    # FOUR entries, matching the other three lists. mixednet.model asserts all
    # four are the same length (mixednet.py:298-305); upstream's own argparse
    # default is "0,0,0,0,0" against four pointwise filters and fails it too.
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]


# data.py:170-190 globs <features_dir>/<split>/**/*_mmap/ for exactly these
# splits. Anything outside them is invisible to the trainer - which is why a
# laxer "**/*_mmap" check passes a set that then loads zero spectrograms.
SPLITS = ("training", "validation", "testing", "testing_ambient", "validation_ambient")

# The smoke-mode sizes. 200 steps against the 10,000-step default: the full
# host run measured 14m14s (docs/SPEED.md) and the training loop is a minority
# of it, so 200 steps is seconds. The eval interval moves 500 -> 50
# deliberately: with the default interval a 200-step run would evaluate only
# at the last step (`step % interval == 0 or is_last_step`), never exercising
# the evaluation path; at 50 it fires four times. Batch size stays default.
SMOKE_TRAINING_STEPS = 200
SMOKE_EVAL_STEP_INTERVAL = 50


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


def run_tag(wake_word, config=None, fallback=None):
    """Name this run after the code AND the audio AND the config.

    Safe to compute before training here: the mWW corpus is built by separate
    commands (mww.corpus then mww.features), so it is already on disk and the
    tag names the audio that will actually be trained on.

    The config half (the `-h` tail) lets two sweep points at the same commit
    and same frozen corpus coexist: without it they share a tag and the second
    dies at the 'directory already exists' check below. `config` is the
    resolved hyperparameters (seed included - two runs differing only in seed
    must not share a tag); paths and feature-set layout stay out of it."""
    return provenance.run_tag(
        wake_word, target="mww", config=config,
        fallback=fallback or datetime.now().strftime("%Y%m%d-%H%M%S"))


def tag_input(cfg, model, model_flags, seed):
    """The resolved config to hash into the tag's config half.

    Every hyperparameter that distinguishes two runs must appear here (or at
    the top level of cfg) - adding a knob to mww_config.build without adding it
    here is how two different training runs start sharing a tag again.
    `features` is excluded on purpose: its directories are machine layout (the
    code half) and the audio they point at is the corpus half.
    model/model_flags/seed are merged in from argparse state: the per-set
    sampling weights sit at build()'s top level, but the architecture flags
    cannot - they are the model_train_eval subcommand, not config keys - so
    `model_flags` is the MERGED list (--model-flags or the MODEL_FLAGS default,
    with the repeatable --model-flag entries applied over it), and a sweep
    moving one geometry flag hashes differently."""
    base = {k: v for k, v in cfg.items()
            if k not in ("train_dir", "summaries_dir", "features")}
    base.update({"model": model, "model_flags": list(model_flags), "seed": seed})
    return base


def checksum(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", required=True)
    p.add_argument("--ambient", nargs="*", default=[],
                   help="RaggedMmap dirs from `download-external-data.sh mww`, "
                        "under data/external/mww_ambient/")
    p.add_argument("--corpus-root", default="data/corpus")
    p.add_argument("--data-dir", default="data/external")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--training-steps", type=int, nargs="+")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: the full pipeline SHAPE with the expensive "
                        "parts minified and the corpus REUSED, not regenerated - "
                        "the end-to-end check for a changed src/train/ tree, in a "
                        "few minutes instead of the 14m14s measured full host run. "
                        "What changes: training_steps 200 (default: 10,000), "
                        "eval_step_interval 50 (default: 500, so the evaluation "
                        "path still fires), and the run is named "
                        "smoke-<timestamp> - a name no real run can take, which "
                        "keeps a smoke model out of the archive's meaning. The "
                        "corpus and pre-built features take the reuse path "
                        "(catalog probes still run for the reuse check). The "
                        "results are NOT measurable: do not evaluate or deploy a "
                        "smoke model. Combining --smoke with an explicit real-size "
                        "flag (--training-steps, --eval-step-interval, "
                        "--batch-size, --config) is an error: a smoke and a real "
                        "size are two different runs.")
    p.add_argument("--batch-size", type=int, default=mww_config.DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rates", type=float, nargs="+",
                   help="learning rate schedule, one value per training stage "
                        "(default: upstream's [0.001] - mww_config's "
                        "DEFAULT_LEARNING_RATES, unchanged in this repo)")
    p.add_argument("--positive-class-weight", type=int, nargs="+",
                   help="positive class weight (default: upstream's [1])")
    p.add_argument("--negative-class-weight", type=int, nargs="+",
                   default=mww_config.DEFAULT_NEGATIVE_CLASS_WEIGHT,
                   help="negative class weight (default: [20], upstream's value). "
                        "A big lever that was never reachable from this CLI: "
                        "every mww run to date silently trained at [20] "
                        "regardless of any config file.")
    p.add_argument("--eval-step-interval", type=int,
                   help="steps between evaluation/validation steps (default: "
                        "upstream's 500, mww_config.DEFAULT_EVAL_STEP_INTERVAL)")
    p.add_argument("--positive-sampling-weight", type=float,
                   help="sampling weight of the positive feature set (default: "
                        "build()'s 2.0)")
    p.add_argument("--negative-sampling-weight", type=float,
                   help="sampling weight of the adversarial-negative feature set "
                        "(default: build()'s 2.0) - the per-set lever this repo's "
                        "config.py named, previously hardcoded and wired to "
                        "nothing")
    p.add_argument("--ambient-sampling-weight", type=float,
                   help="sampling weight of the ambient feature sets (default: "
                        "build()'s 1.0)")
    p.add_argument("--positive-penalty-weight", type=float,
                   help="penalty weight of the positive feature set (default: "
                        "build()'s 1.0)")
    p.add_argument("--negative-penalty-weight", type=float,
                   help="penalty weight of the adversarial-negative feature set "
                        "(default: build()'s 1.0)")
    p.add_argument("--ambient-penalty-weight", type=float,
                   help="penalty weight of the ambient feature sets (default: "
                        "build()'s 1.0)")
    p.add_argument("--model-flag", action="append", default=[], metavar="K=V",
                   help="one architecture flag, repeatable (e.g. --model-flag "
                        "stride=4), merged over the effective flag list "
                        "(--model-flags if given, else the MODEL_FLAGS default): "
                        "an already-present flag's value is REPLACED, otherwise "
                        "the pair is appended - a sweep wants to move one "
                        "geometry flag without re-typing all of MODEL_FLAGS. "
                        "The merged list is what the int8 quantization pre-check "
                        "reads and the run tag's config half hashes, so a "
                        "--model-flag point is a distinct sweep point.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the training stage (default: %(default)s = "
                        "unseeded). Seeded into the TF subprocess via "
                        "TF_SET_RANDOM_SEED (how TensorFlow accepts a seed "
                        "without patching model_train_eval) and into this "
                        "process's random/numpy. The seed is part of the run's "
                        "tag: two runs differing only in seed must not be filed "
                        "under one name.")
    p.add_argument("--tag", default=None,
                   help="name for this run's output directory (default: "
                        "<commit>[-dirty]-c<corpus>-h<config>, see "
                        "src/train/provenance.py). Each run gets its own - "
                        "model_train_eval refuses to train into an existing "
                        "directory.")
    p.add_argument("--force", action="store_true",
                   help="delete this run's output directory if it already exists")
    p.add_argument("--config", default=None,
                   help="use an existing YAML instead of generating one")
    p.add_argument("--model", default=MODEL, choices=("mixednet", "inception"),
                   help="architecture subcommand (default: %(default)s)")
    p.add_argument("--model-flags", nargs=argparse.REMAINDER, default=None,
                   help="override the architecture flags entirely; everything after "
                        "this is passed through verbatim")
    p.add_argument("--print-tag", action="store_true",
                   help="print the run tag for these arguments and exit without "
                        "training. The wrapper scripts use it: the config half is "
                        "only knowable here, where the resolved config is built - "
                        "computing the tag in two places is how the archive and "
                        "the run directory would drift apart.")
    p.add_argument("passthrough", nargs="*", default=[],
                   help="extra args for model_train_eval, after --")
    args = p.parse_args()


    if args.model_flags is None:
        args.model_flags = MODEL_FLAGS if args.model == "mixednet" else []
    # The repeatable --model-flag k=v entries, merged OVER the list above: an
    # already-present flag's value is replaced, otherwise the pair is appended.
    # The merge goes into a FRESH list - MODEL_FLAGS is a module constant and
    # mutating it in place would leak one sweep point into the next run. It
    # happens BEFORE the quantization check further down, which builds its
    # flags dict from args.model_flags: a stride=... that breaks the
    # divisibility must cost seconds, not a full run.
    for item in args.model_flag:
        key, sep, value = item.partition("=")
        if not sep or not key:
            sys.exit(f"--model-flag takes k=v, got {item!r}")
        flag = key if key.startswith("--") else "--" + key
        merged = list(args.model_flags)
        if flag in merged:
            i = merged.index(flag)
            if i + 1 < len(merged):
                merged[i + 1] = value
            else:  # a flag with no value: complete it rather than append a second
                merged.append(value)
        else:
            merged.extend([flag, value])
        args.model_flags = merged

    # === SMOKE MODE: decided here, before any stage - a contradiction costs
    # zero seconds.
    if args.smoke:
        # The smoke minifies a FIXED set of sizes; an explicit real size on the
        # command line is a contradiction, not an override - error out.
        explicit = []
        if args.training_steps is not None:
            explicit.append("--training-steps")
        if args.eval_step_interval is not None:
            explicit.append("--eval-step-interval")
        # get_default takes the DEST, not the option string, on recent 3.12
        # ("batch_size", not "--batch-size"): the option-string form silently
        # returns None and this check fires on every smoke run.
        if args.batch_size != p.get_default("batch_size"):
            explicit.append("--batch-size")
        if args.config:
            explicit.append("--config")
        if explicit:
            sys.exit(f"ERROR: --smoke is combined with explicit real-size flags "
                     f"({', '.join(explicit)}). It minifies them itself; a smoke and "
                     f"a real size are two different runs. Drop the flag for the "
                     f"run you actually want.")
        args.training_steps = [SMOKE_TRAINING_STEPS]
        args.eval_step_interval = SMOKE_EVAL_STEP_INTERVAL
        # The run directory IS the archive's guard here: mww files every run
        # under output/<wake>/mww/<tag>/ and the wrapper copies the model out
        # named after <tag>, so a smoke-named directory and file can never be
        # read as a real run (a real tag is <commit>[-dirty]-c<hex>[-h<hex>]).
        if args.tag is None:
            args.tag = f"smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        short, _, _, _, _ = provenance.corpus_tag(args.wake_word, "mww")
        print("=" * 60)
        print(f"SMOKE MODE: training minified ({args.training_steps[0]} steps, "
              f"eval every {args.eval_step_interval}), corpus and features REUSED "
              f"(tag c{short}), results are NOT measurable")
        print(f"  run directory: {args.output_dir}/"
              f"{args.wake_word.replace(' ', '_').lower()}/mww/{args.tag}")
        print("=" * 60)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        # The training runs in the model_train_eval subprocess; TF_SET_RANDOM_SEED
        # is the one seed hook upstream accepts without a patch.
        os.environ["TF_SET_RANDOM_SEED"] = str(args.seed)

    safe = args.wake_word.replace(" ", "_").lower()

    if args.config:
        config_path = Path(args.config)
        cfg = yaml.safe_load(config_path.read_text())
        resolved = None      # a hand-written config is filed as given, not re-resolved
    else:
        corpus = Path(args.corpus_root) / safe / "mww" / "features"
        # The placeholder tag is replaced below: the tag's config half is a hash
        # OF this config, so the config has to exist before the tag does.
        cfg = mww_config.build(
            args.wake_word, corpus / "positives", corpus / "negatives",
            args.ambient, args.output_dir, data_dir=args.data_dir,
            training_steps=args.training_steps,
            learning_rates=args.learning_rates,
            batch_size=args.batch_size,
            negative_class_weight=args.negative_class_weight,
            positive_class_weight=args.positive_class_weight,
            eval_step_interval=args.eval_step_interval,
            positive_sampling_weight=args.positive_sampling_weight,
            negative_sampling_weight=args.negative_sampling_weight,
            ambient_sampling_weight=args.ambient_sampling_weight,
            positive_penalty_weight=args.positive_penalty_weight,
            negative_penalty_weight=args.negative_penalty_weight,
            ambient_penalty_weight=args.ambient_penalty_weight,
            run_tag="run")
        resolved = tag_input(cfg, args.model, args.model_flags, args.seed)
        tag = args.tag or run_tag(args.wake_word, config=resolved)
        if args.print_tag:
            print(tag)
            sys.exit(0)
        cfg["train_dir"] = str(Path(args.output_dir) / safe / "mww" / tag)
        cfg["summaries_dir"] = str(
            Path(args.output_dir) / safe / "mww" / tag / "summaries")
        # SIBLING OF train_dir, NOT INSIDE IT. model_train_eval's
        # os.makedirs(train_dir) fails if anything is there - a config file
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
    # AMBIENT *EVALUATION* DATA IS WHAT MODEL SELECTION RUNS ON. The
    # maximization metric is average_viable_recall, from false accepts per
    # hour on validation_ambient: with no such data the metric is 0.000 at
    # every step, the "best" checkpoint never improves, and the exported model
    # is whichever one happened to be current - while the ordinary
    # accuracy/recall numbers still look excellent.
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
            "splits (e.g. data/external/mww_ambient/dinner_party_eval)")
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
    # converting, so a crash during quantization calibration leaves 0 bytes
    # behind - which existed, and had changed, and so passed both checksums
    # above. It reported "DONE ... (0 KB, md5 d41d8cd9)" - the md5 of nothing.
    size = model_path.stat().st_size
    if size < 1024:
        sys.exit(f"\nTRAINING FAILED: {model_path} is {size} bytes - the TFLite "
                 f"conversion did not produce a model (model_train_eval exited "
                 f"{result.returncode}). Look for the traceback above; a failure in "
                 f"quantization calibration is the usual cause.")
    if before is not None and before == after:
        sys.exit(f"\nTRAINING FAILED: {model_path} is unchanged from before this run - "
                 "it is the PREVIOUS model. Do not evaluate or deploy it.")

    # FILE THE RESOLVED CONFIG UNDER THE TAG, only now that the model exists:
    # a failed run must not leave a config that names a model that was never
    # written. This is what the sweep ledger reads, so it is the FULL resolved
    # hyperparameters - the same dict the tag's config half hashes - not a
    # hand-picked subset.
    if resolved is not None:
        train_dir.with_name(f"{tag}.config.json").write_text(
            json.dumps(resolved, indent=2, default=str) + "\n")

    # Give the run directory back to the host user before anything on the host
    # has to touch it - run-mww-training.sh's collection step copies these files
    # out, and would otherwise hit Permission denied. See src/train/ownership.py.
    ownership.hand_back(Path(args.output_dir), work_dir=Path.cwd())

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
