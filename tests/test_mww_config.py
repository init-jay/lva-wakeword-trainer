"""Guards for src/train/mww/config.py.

Each test here guards a decision that once cost a full mWW training run (tens
of minutes) when it broke:

* The int8 calibration asserts the spectrogram length divides by the stride
  only AFTER training completes - a wrong CLIP_DURATION_MS therefore costs a
  full run and leaves a 0-byte .tflite behind (config.py, "THE QUANTIZATION
  CONSTRAINT"). The formula's trap is that one frame is stride x window_step
  ms, not window_step ms; the tests pin the documented 204-frame case so the
  reimplementation cannot drift from the one in model_train_eval.py.
* Callers pass None to mean "use the default"; an explicit None used to
  override the per-set DEFAULT_* weights and land in the YAML and the tag.
* The run tag's config half hashes the TOP-LEVEL weight keys (tag_input drops
  features), so a weight moved in one build() call must move in the emitted
  dict or the run shares its tag with the default.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from train.mww import config as mww  # noqa: E402

# The upstream notebook defaults src/train/mww/train.py actually launches with -
# the "known-good" architecture. Kept in lockstep with MODEL_FLAGS there.
KNOWN_GOOD_FLAGS = {
    "pointwise_filters": "64,64,64,64",
    "repeat_in_block": "1,1,1,1",
    "mixconv_kernel_sizes": "[5], [7,11], [9,15], [23]",
    "residual_connection": "0,0,0,0",
    "first_conv_filters": "32",
    "first_conv_kernel_size": "5",
    "stride": "3",
}


def test_mixednet_slices_dropped_known_good():
    # 4 (first conv) + 3 x (4 + 10 + 14 + 22) (the four blocks).
    assert mww.mixednet_slices_dropped(KNOWN_GOOD_FLAGS) == 154


def test_spectrogram_length_documented_204_frame_case():
    # 1500 ms at 10 ms steps, stride 3: 24000 samples - 480 window = 23520,
    # 23520 / (3 x 480) = 49 -> 1 + 49 + 154 = 204, the value the config
    # comment documents as the one that divides cleanly.
    length = mww.spectrogram_length(1500, 10, 3, 154)
    assert length == 204
    assert 204 % 3 == 0


def test_quantization_constraint_passes_with_known_good_defaults():
    ok, length, msg = mww.check_quantization_constraint(KNOWN_GOOD_FLAGS)
    assert ok, msg
    assert length == 204
    assert "divisible" in msg


def test_quantization_constraint_fails_and_names_a_working_value():
    # 1560 = 1500 + two 30 ms frames (stride 3 x 10 ms step): the length
    # moves 204 -> 206, which does not divide by 3. The failure must name a
    # working duration, not just say "wrong" - and the suggested value must
    # actually check out, since a wrong suggestion would cost the next run
    # the whole point of this check.
    ok, length, msg = mww.check_quantization_constraint(
        KNOWN_GOOD_FLAGS, clip_duration_ms=1560)
    assert not ok
    assert length == 206
    assert "Try CLIP_DURATION_MS" in msg
    suggested = int(msg.split("Try CLIP_DURATION_MS = ")[1].split()[0])
    ok2, _, _ = mww.check_quantization_constraint(
        KNOWN_GOOD_FLAGS, clip_duration_ms=suggested)
    assert ok2


def _build(**overrides):
    # Directories are opaque here: build() only stringifies them into the
    # emitted config, so a throwaway path per call keeps the assertions
    # about the weights and the tag-relevant top level, not the layout.
    kwargs = dict(
        wake_word="hey seeree",
        positives_dir=Path("tmp") / "pos",
        negatives_dir=Path("tmp") / "neg",
        ambient_dirs=[],
        output_dir="output",
    )
    kwargs.update(overrides)
    return mww.build(**kwargs)


def test_build_defaults_survive_explicit_none():
    # The live bug class: the CLI passes None for "use the default" and the
    # old code let it through into the YAML and the tag instead of the
    # DEFAULT_* value.
    cfg = _build(training_steps=None, learning_rates=None,
                 negative_class_weight=None, positive_class_weight=None,
                 eval_step_interval=None,
                 positive_sampling_weight=None, positive_penalty_weight=None,
                 negative_sampling_weight=None, negative_penalty_weight=None,
                 ambient_sampling_weight=None, ambient_penalty_weight=None)
    assert cfg["training_steps"] == mww.DEFAULT_TRAINING_STEPS
    assert cfg["learning_rates"] == mww.DEFAULT_LEARNING_RATES
    assert cfg["negative_class_weight"] == mww.DEFAULT_NEGATIVE_CLASS_WEIGHT
    assert cfg["positive_class_weight"] == mww.DEFAULT_POSITIVE_CLASS_WEIGHT
    assert cfg["eval_step_interval"] == mww.DEFAULT_EVAL_STEP_INTERVAL
    for key, default in (
            ("positive_sampling_weight", mww.DEFAULT_POSITIVE_SAMPLING_WEIGHT),
            ("positive_penalty_weight", mww.DEFAULT_POSITIVE_PENALTY_WEIGHT),
            ("negative_sampling_weight", mww.DEFAULT_NEGATIVE_SAMPLING_WEIGHT),
            ("negative_penalty_weight", mww.DEFAULT_NEGATIVE_PENALTY_WEIGHT),
            ("ambient_sampling_weight", mww.DEFAULT_AMBIENT_SAMPLING_WEIGHT),
            ("ambient_penalty_weight", mww.DEFAULT_AMBIENT_PENALTY_WEIGHT)):
        assert cfg[key] == default, key


def test_build_weight_change_changes_the_top_level_config():
    # The tag's config half hashes the top level; if one weight moved
    # without the dict changing, a sweep point would share its tag with the
    # default run and mww training would refuse to run into the directory it
    # already claimed.
    a = _build(negative_sampling_weight=2.0)
    b = _build(negative_sampling_weight=3.0)
    assert a["negative_sampling_weight"] == 2.0
    assert b["negative_sampling_weight"] == 3.0
    assert a != b


def test_the_clips_factory_does_not_hand_the_split_back_to_upstream():
    """random_split_seed is None: the split stays ours.

    A seed there makes microwakeword/audio/clips.py:145-157 build split_clips itself, and
    upstream splits per FILE - so the N copies of one human recording scatter across train
    and validation. That is the leak src/train/mww/split.py exists to close, and it returns
    silently: every count still looks right, only the held-out recordings stop being held
    out. The factory has no caller today (every feature set the trainers assemble is
    mmap_feature_set), which is precisely why nothing else would notice it being wired up
    with a seed later.
    """
    fs = mww.clips_feature_set("dir", "hey seeree", 1.0, 0.1, [], [])
    settings = fs["clips_settings"]
    assert settings["random_split_seed"] is None, settings
    assert settings["split_count"] == 0.1, settings
    assert settings["remove_silence"] is False, settings


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
