"""Emit a microWakeWord training YAML pointing at this repo's corpus.

WHY A GENERATOR AND NOT A CHECKED-IN YAML. The same reason train.py builds
openWakeWord's config from a template rather than shipping one: the paths, the
per-set weights and the clip counts all move together, and a hand-edited YAML drifts
from what the corpus actually contains. This mirrors create_config in train.py.

EVERYTHING IS `type: mmap`, INCLUDING OUR OWN CORPUS. A feature set can also be
`type: clips`, which reads a directory of WAVs and generates spectrograms on the fly,
and that looked like it removed the need for a conversion step. It does not:
ClipsHandlerWrapperGenerator.get_mode_size returns 0 for every mode except
"training" (data.py:357-362), so a clips set supplies no validation or testing data
and the first validation step fails on whatever shape the ambient sets yield instead.

A clips training set would also LEAK. It is constructed with
spectrogram_generator(random=True), which draws from Clips.clips - every clip in the
directory, ignoring the train/validation/test split - so training would sample the
same clips that validation is scored on.

So mww/features.py writes all three splits to RaggedMmap up front, and this config
points at those. The ambient negatives from Hugging Face arrive in the same form.

DERIVED KEYS ARE NOT WRITTEN HERE. `spectrogram_length`,
`spectrogram_length_final_layer`, `training_input_shape` and `stride` are computed by
model_train_eval.py:60-93 from `clip_duration_ms`, `window_step_ms` and the model
flags. Writing them here would be duplicating a calculation that the trainer will
redo, and a stale copy is worse than none.

    python -m mww.config --wake-word "hey seeree" --out training_parameters.yaml
"""

import argparse
from pathlib import Path

import yaml

# 1500 ms, against openWakeWord's 2000 ms window.
#
# THE FRONTEND. A clip that sits comfortably in a 2 s window may not in 1.5 s.
# corpus/augment.py's trimming makes that survivable - it is why the phrase is flush
# to the end - but the alignment reasoning in tuning.md does NOT carry over unchecked.
#
# THE QUANTIZATION CONSTRAINT. `spectrogram_length` must be divisible by `stride`,
# and int8 calibration asserts it AFTER training completes (utils.py:321) - so
# getting it wrong costs a full run and leaves a 0-byte .tflite behind. At 10 ms
# steps and stride 3, 1500 ms gives 204, which divides cleanly.
#
# THIS VALUE IS COUPLED TO WINDOW_STEP_MS. At the earlier 20 ms step, 1500 gave 179
# and had to move to 1560. Changing either requires re-deriving the other - do not
# hand-pick it. check_quantization_constraint() at the bottom of this file derives it
# from the model flags and names a working value, and mww/train.py calls it before
# launching so the failure costs seconds rather than a full run.
CLIP_DURATION_MS = 1500

# 10 ms, MATCHING ESPHOME'S PREPROCESSOR. Not a free parameter.
#
# ESPHome's micro_wake_word generates 40 features every 10 ms, and mWW assumes the
# preprocessor step equals this value:
#
#     preprocessor_window_step = config["window_step_ms"]   # model_train_eval.py:66
#
# so training at 20 ms builds a model expecting features at half the rate the device
# delivers. That does not error on device - it detects badly, which is worse. The
# manifest's `feature_step_size` is emitted from this constant (mww/manifest.py) so
# the two cannot drift apart.
#
# It also sets the frame step for the quantization constraint, which is
# `stride * window_step_ms` = 30 ms here. See CLIP_DURATION_MS.
WINDOW_STEP_MS = 10

# Augmentation corpora already on disk from setup-data.sh. mWW's Augmentation takes
# the same two things openWakeWord's does, from the same downloads.
IMPULSE_DIRS = ["data/mit_rirs"]
BACKGROUND_DIRS = ["data/audioset_16k", "data/fma"]

# Upstream notebook defaults, kept verbatim as the starting point. Change one at a
# time and record it, tuning.md style - the notebook's own README says a usable model
# takes a lot of experimentation, which is seventeen runs of this repo restated.
DEFAULT_TRAINING_STEPS = [10000]
DEFAULT_LEARNING_RATES = [0.001]
DEFAULT_BATCH_SIZE = 128
DEFAULT_EVAL_STEP_INTERVAL = 500
DEFAULT_POSITIVE_CLASS_WEIGHT = [1]
DEFAULT_NEGATIVE_CLASS_WEIGHT = [20]


def clips_feature_set(directory, truth, sampling_weight, penalty_weight,
                      impulse_dirs, background_dirs, truncation_strategy="default",
                      slide_frames=10, step_ms=WINDOW_STEP_MS):
    """A feature set generated on the fly from a directory of WAVs.

    `slide_frames` is 10 for training and validation and 1 for testing upstream: >1
    yields several overlapping spectrograms per clip by dropping end frames, which
    simulates the sequential inputs a streaming model actually sees. Testing wants the
    real thing, so it uses 1.
    """
    return {
        "type": "clips",
        "truth": truth,
        "sampling_weight": sampling_weight,
        "penalty_weight": penalty_weight,
        "truncation_strategy": truncation_strategy,
        "clips_settings": {
            "input_directory": str(directory),
            "file_pattern": "*.wav",
            # No remove_silence here: corpus/augment.py has already trimmed, with a
            # method calibrated on this corpus. Letting webrtcvad trim a second time
            # would stack two different silence definitions on the same clips.
            "remove_silence": False,
            "random_split_seed": 10,
            "split_count": 0.1,
        },
        "augmentation_settings": {
            "augmentation_duration_s": CLIP_DURATION_MS / 1000.0,
            "impulse_paths": [str(p) for p in impulse_dirs],
            "background_paths": [str(p) for p in background_dirs],
            "background_min_snr_db": -5,
            "background_max_snr_db": 10,
            "min_jitter_s": 0.195,
            "max_jitter_s": 0.205,
        },
        "spectrogram_generation_settings": {
            "step_ms": step_ms,
            "slide_frames": slide_frames,
        },
    }


def mmap_feature_set(features_dir, truth, sampling_weight, penalty_weight,
                     truncation_strategy="truncate_start"):
    """A pre-generated RaggedMmap set - the Hugging Face ambient negatives.

    Layout is <features_dir>/{training,validation,testing,testing_ambient,
    validation_ambient}/**/*_mmap/ (data.py:170-190).
    """
    return {
        "type": "mmap",
        "features_dir": str(features_dir),
        "truth": truth,
        "sampling_weight": sampling_weight,
        "penalty_weight": penalty_weight,
        "truncation_strategy": truncation_strategy,
    }


def build(wake_word, positives_dir, negatives_dir, ambient_dirs, output_dir,
          data_dir=".", training_steps=None, learning_rates=None,
          batch_size=DEFAULT_BATCH_SIZE, negative_class_weight=None, run_tag=None):
    safe = wake_word.replace(" ", "_").lower()
    # Augmentation now happens in mww/features.py, when the spectrograms are
    # written, so no augmentation settings appear in this config at all.
    del data_dir

    features = [
        # Positives from mww/corpus.py: synthetic voices plus real recordings,
        # trimmed and carrying the child-range copies, in mWW's own directory.
        #
        # Real recordings are NOT duplicated here the way --real-copies duplicates
        # them for openWakeWord. That trick exists because openWakeWord augments by
        # globbing the directory once, so N copies become N augmented variants. mWW
        # augments on every read, so copies would only bias sampling - and
        # `sampling_weight` below is the honest knob for that. See corpus/real.py.
        mmap_feature_set(positives_dir, truth=True, sampling_weight=2.0,
                         penalty_weight=1.0, truncation_strategy="default"),
        # This repo's ADVERSARIAL negatives - "hey serious", "hey Sienna". ~100 clips
        # against ambient sets orders of magnitude larger, so they need a sampling
        # weight that keeps them visible. This is the per-set lever openWakeWord did
        # not have: there, max_negative_weight applied to the whole negative class.
        mmap_feature_set(negatives_dir, truth=False, sampling_weight=2.0,
                         penalty_weight=1.0, truncation_strategy="default"),
    ]
    for d in ambient_dirs:
        features.append(mmap_feature_set(d, truth=False, sampling_weight=1.0,
                                         penalty_weight=1.0))

    return {
        "window_step_ms": WINDOW_STEP_MS,
        "clip_duration_ms": CLIP_DURATION_MS,
        "batch_size": batch_size,
        "training_steps": training_steps or DEFAULT_TRAINING_STEPS,
        "learning_rates": learning_rates or DEFAULT_LEARNING_RATES,
        "positive_class_weight": DEFAULT_POSITIVE_CLASS_WEIGHT,
        "negative_class_weight": negative_class_weight or DEFAULT_NEGATIVE_CLASS_WEIGHT,
        "eval_step_interval": DEFAULT_EVAL_STEP_INTERVAL,
        # Upstream's two-step selection: get false accepts per hour under target
        # first, then maximise recall. The same shape as this repo's rule that
        # detection is only comparable at matched false accepts.
        "minimization_metric": None,
        "target_minimization": 0.5,
        "maximization_metric": "average_viable_recall",
        # ONE DIRECTORY PER RUN. model_train_eval does os.makedirs(train_dir) and
        # raises "model already exists in folder ..." if it is there at all
        # (model_train_eval.py:111-120) - it will not train into an existing
        # directory, not even an empty one holding only a config file.
        #
        # Tagging per run also keeps history, which the openWakeWord side did not:
        # its rmtree deleted every archived model on each run until the corpus was
        # nested under oww/.
        "train_dir": str(Path(output_dir) / safe / (run_tag or "run")),
        "summaries_dir": str(Path(output_dir) / safe / (run_tag or "run") / "summaries"),
        "features": features,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    # mWW owns its corpus, built by mww/corpus.py. Reading the openWakeWord one in
    # place was tried and rejected: train.py rmtree's it at the start of every run.
    p.add_argument("--positives", default=None,
                   help="default: <corpus-root>/<wake_word>/mww/positives")
    p.add_argument("--negatives", default=None,
                   help="default: <corpus-root>/<wake_word>/mww/negatives")
    p.add_argument("--corpus-root", default="my_custom_model",
                   help="corpora live at <root>/<wake_word>/{oww,mww}/ "
                        "(default: %(default)s)")
    p.add_argument("--ambient", nargs="*", default=[],
                   help="RaggedMmap feature dirs for the ambient negatives")
    p.add_argument("--data-dir", default=".",
                   help="where mit_rirs/audioset_16k/fma live (default: %(default)s)")
    p.add_argument("--output-dir", default="mww_models")
    p.add_argument("--training-steps", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--out", default="training_parameters.yaml")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    corpus = Path(args.corpus_root) / safe / "mww" / "features"
    positives = Path(args.positives) if args.positives else corpus / "positives"
    negatives = Path(args.negatives) if args.negatives else corpus / "negatives"

    for label, d in (("positives", positives), ("negatives", negatives)):
        n = len(list(d.glob("*/*_mmap"))) if d.is_dir() else 0
        print(f"  {label:<10} {d}  ({n} mmap sets)")
        if n == 0:
            print("           EMPTY OR MISSING. Build it first:")
            print(f'             python -m mww.corpus   --wake-word "{args.wake_word}"')
            print(f'             python -m mww.features --wake-word "{args.wake_word}"')

    cfg = build(args.wake_word, positives, negatives, args.ambient,
                args.output_dir, data_dir=args.data_dir,
                training_steps=args.training_steps, batch_size=args.batch_size)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.out}: {len(cfg['features'])} feature sets, "
          f"clip_duration_ms={cfg['clip_duration_ms']}")
    if not args.ambient:
        print("  WARNING: no ambient negative sets. Training against this repo's ~100")
        print("           adversarial negatives alone will produce a model that fires")
        print("           on ordinary speech - those sets are what teach silence and")
        print("           background. Fetch them first (plan.md phase 2).")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The quantization constraint, checkable before a run instead of after one.
#
# int8 calibration feeds the representative dataset in stride-sized slices and
# asserts the spectrogram divides evenly (utils.py:321). It runs AFTER training
# completes, so getting this wrong costs a full run and leaves a 0-byte .tflite.
#
# Two attempts got it wrong by reasoning from part of the formula. The whole of it
# is model_train_eval.py:60-88, and the part that matters is that the frame step
# includes the STRIDE:
#
#     window_step_samples = stride * 16000 * window_step_ms / 1000
#
# so one frame is stride x window_step_ms = 60 ms here, not 20. Changing the clip
# duration by 20 ms moves nothing; it takes 60 ms to move the length by one.
# ---------------------------------------------------------------------------

PREPROCESSOR_SAMPLE_RATE = 16000
PREPROCESSOR_WINDOW_SIZE_MS = 30


def _parse_list(text):
    """mixednet's flag format: '1,1,1,1' or '[5], [7,11], [9,15], [23]'."""
    import ast
    text = text.strip()
    if "[" in text:
        return list(ast.literal_eval(f"[{text}]"))
    return [int(x) for x in text.split(",") if x.strip()]


def mixednet_slices_dropped(flags: dict) -> int:
    """Frames lost to valid padding - mixednet.py:108-128, reimplemented."""
    dropped = 0
    if int(flags["first_conv_filters"]) > 0:
        dropped += int(flags["first_conv_kernel_size"]) - 1
    stride = int(flags["stride"])
    for repeat, ksize in zip(_parse_list(flags["repeat_in_block"]),
                             _parse_list(flags["mixconv_kernel_sizes"])):
        dropped += (repeat * (max(ksize) - 1)) * stride
    return dropped


def spectrogram_length(clip_duration_ms, window_step_ms, stride, slices_dropped):
    """model_train_eval.py:66-88, reimplemented so it can be checked up front."""
    desired = int(PREPROCESSOR_SAMPLE_RATE * clip_duration_ms / 1000)
    window = int(PREPROCESSOR_SAMPLE_RATE * PREPROCESSOR_WINDOW_SIZE_MS / 1000)
    step = int(stride * PREPROCESSOR_SAMPLE_RATE * window_step_ms / 1000)
    remainder = desired - window
    final_layer = 0 if remainder < 0 else 1 + int(remainder / step)
    return final_layer + slices_dropped


def check_quantization_constraint(flags: dict, clip_duration_ms=None,
                                  window_step_ms=None):
    """(ok, length, message). Suggests a clip duration when it does not divide."""
    clip_duration_ms = clip_duration_ms or CLIP_DURATION_MS
    window_step_ms = window_step_ms or WINDOW_STEP_MS
    stride = int(flags["stride"])
    dropped = mixednet_slices_dropped(flags)
    length = spectrogram_length(clip_duration_ms, window_step_ms, stride, dropped)
    if length % stride == 0:
        return True, length, f"spectrogram_length {length}, divisible by stride {stride}"

    frame_ms = stride * window_step_ms
    for delta in range(1, 40):
        for candidate in (clip_duration_ms + delta * frame_ms,
                          clip_duration_ms - delta * frame_ms):
            if candidate <= 0:
                continue
            if spectrogram_length(candidate, window_step_ms, stride,
                                  dropped) % stride == 0:
                return False, length, (
                    f"spectrogram_length {length} is not divisible by stride "
                    f"{stride} - int8 calibration will assert AFTER training "
                    f"completes. Try CLIP_DURATION_MS = {candidate} "
                    f"(one frame is stride x window_step_ms = {frame_ms} ms).")
    return False, length, (f"spectrogram_length {length} is not divisible by stride "
                           f"{stride}, and no nearby clip duration fixes it")
