#!/usr/bin/env python3
"""
Train OpenWakeWord model using Kokoro TTS synthetic voices + real recordings.

Usage:
    python train.py --wake-word "hey seeree"
    python train.py --wake-word "okay jarvis" --samples-per-voice 300 --training-steps 75000

Docker:
    docker compose run --rm oww-trainer python -m train.oww.train \\
        --wake-word "hey seeree"
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml
from tqdm import tqdm

# THE REPO ROOT, NOT THIS FILE'S DIRECTORY - the chdir below anchors every
# relative path in this module. Getting this wrong does not raise: it builds the
# corpus under train/oww/ and trains on nothing.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The plain-path form (`python train/oww/train.py`) puts train/oww/ on sys.path, not the root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The engine-agnostic half of corpus construction, shared with the microWakeWord trainer.
from train.corpus.augment import (CHILD_STRETCH, CHILD_STRETCH_FRACTION,  # noqa: E402
                                  add_child_range_copies, trim_directory,
                                  trim_silence)
from train.corpus.kokoro import (KokoroPool,  # noqa: E402
                                 generate_kokoro_samples, get_kokoro_voices,
                                 kokoro_tts, kokoro_tts_batch,
                                 kokoro_tts_timed, phrase_end_sample,
                                 probe_kokoro_servers, run_jobs)
from train.corpus.negatives import (LEGACY_VOICE_MARKER,  # noqa: E402
                                    MISPRONOUNCING_VOICES,
                                    TRAINING_COMMANDS, build_negative_phrases)
from train.corpus.piper import (generate_piper_samples,  # noqa: E402
                                select_piper_voices)
from train.corpus.positives import (PLAIN_SPEED_GRID, PLAIN_SPEEDS,  # noqa: E402
                                    plain_positive_texts)
from train.corpus.real import copy_real_samples  # noqa: E402
from train import ownership, provenance  # noqa: E402
from train.corpus import manifest as corpus_manifest  # noqa: E402
from wordlists import exclude_voice_holdout, load_voice_holdout  # noqa: E402
from wordlists import voice_holdout_path  # noqa: E402

warnings.filterwarnings("ignore", message="Reached EOF prematurely")

WORK_DIR = REPO_ROOT
os.chdir(WORK_DIR)

# The run's seed, set in main() from --seed. Module-level because the two
# subprocess launchers (run_augmentation / run_training) pass it to the
# openwakeword train.py subprocess as PYTHONHASHSEED without it being threaded
# through every call site.
_SEED: int = 0

# The smoke-mode training size: the smallest at which the full pipeline still
# runs end to end. 200 steps is ~30 s of the torch loop on an M-series Mac,
# against ~8 min at the 50,000-step default (SPEED.md: the full host run's
# training loop, 50k steps, is the minority of its ~35 minutes). The feature
# arrays are NOT minified - upstream sizes them from the corpus directories and
# exposes no size knob - so a smoke run still pays the full recompute (~12 min
# measured, the augmentation+features stage). The corpus is REUSED, never
# regenerated: generation is the 30-54 minute TTS stage, and its server health
# is probed at the start of a normal run anyway - the smoke exists to exercise
# the pipeline between the probes, not the probes.
SMOKE_TRAINING_STEPS = 200

# Speed coverage of the positives, widened at the top for run 9.
#
# Measured failure: a synthetic sweep of the run 4 model detected 6/6 up to 1.25x,
# then 3/6 at 1.40x, 2/6 at 1.60x - training rendered nothing above 1.3x, and it was
# fine below (6/6 at 0.55x). Widen the top only.
#
# Matches a real failure too: four of the five held-out clips run 4 missed were the
# fast ones, and the shortest (300 ms) shorter than every clip it detected. Kokoro
# at 1.6x renders "hey seeree" in 390-590 ms, exactly that range - checked for
# intelligibility, since degraded audio is worse than no coverage of the speed.
#
# Both lists move together - one variable, "how fast can the phrase be" - since
# fast run-on speech is the commonest real case.
#
# Stays discrete and five long so the fallback path can cache its phrase-alone
# reference per (voice, speed).
RUNON_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6]

# How much of the command's onset to keep after the wake word ends, in ms.
#
# The value that matters is where the phrase ends relative to the END OF THE ARRAY,
# because create_fixed_size_clip aligns that with the window. Plain positives sit at
# ~80 ms (30 ms trim pad + ~50 ms residual); run-ons must match or the positive set
# is bimodal and the model learns the later mode.
#
# The boundary comes from Kokoro's /dev/captioned_speech word timestamps, so this is
# the whole overshoot, not jitter on an estimate. Two earlier attempts inferred it
# from a phrase-alone rendering:
#
#   v1, cut at phrase_len + U(50,250): kept 270-470 ms of command. Alignment peak
#      160 -> 480 ms, median latency 70 -> 130 ms, extend false accepts 4/32 -> 7/32 -
#      a trailing region holding speech in BOTH classes stops discriminating.
#   v2, correcting for the 30 ms trim pad: still median +153 ms late and
#      voice-dependent (af_bella ~0, bf_lily +348..+459), and 2/18 clips cut inside
#      the wake word.
#
# The timestamps remove both the bias and the variance. The fallback path still uses
# v2, which is why it reports itself loudly.
#
# THE RANGE MUST NOT START AT ZERO. The margin is not padding; it is what lets the
# model hear the word ENDED rather than continued - the whole discrimination between
# "hey seeree" and "hey serious". Measured against held-out real recordings:
#
#   effective margin   held-out run-on   extend+hey_other FA   latency
#     ~50 ms (run 5)         28%              12/32             -20 ms
#    ~140 ms (run 6)         40%               8/32              48 ms
#    ~200 ms (run 4)         46%               6/32              77 ms
#    ~225 ms (run 7)         56%               7/32              83 ms
#
# Runs 6 and 7 share an identical real-sample corpus and differ only in this
# constant: +85 ms of margin bought +16 points of real run-on detection. That is the
# relationship this value exploits.
#
# The false-accept column is NOT a gradient: it plateaus at 7-8/32 across a 3x range
# of margin (the early monotonic reading was mostly escaping the pathological zero
# case; run 4 also had half the real data). Do not raise this expecting fewer false
# accepts.
#
# The cost is latency, which tracks margin and sits at 83 ms against a 120 ms gate.
# Roughly one more step of headroom, for diminishing returns.
RUNON_TAIL_MS = (150.0, 300.0)


def report_onnx_providers():
    """Say plainly whether feature computation will run on the GPU.

    The failure is silent and expensive: onnxruntime falls back to CPU with a
    warning rather than erroring, and openwakeword picks its thread count from torch
    rather than onnxruntime - so a box with a working GPU and the CPU build computes
    features single-threaded on CPU. That cost 36 minutes of an 83-minute run before
    anyone noticed the warning in the scrollback.

    get_available_providers() is not enough: it reports what the build supports, not
    what will load. So open a real session against the model that will be used.
    """
    try:
        import onnxruntime
    except ImportError:
        print("  onnxruntime not importable - feature computation will fail")
        return

    available = onnxruntime.get_available_providers()
    melspec = WORK_DIR / "openwakeword/openwakeword/resources/models/melspectrogram.onnx"

    actual = "CPUExecutionProvider"
    if "CUDAExecutionProvider" in available and melspec.exists():
        try:
            session = onnxruntime.InferenceSession(
                str(melspec), providers=["CUDAExecutionProvider"])
            actual = session.get_providers()[0]
        except Exception as e:
            print(f"  Could not open a CUDA session: {e}")

    print(f"  onnxruntime {onnxruntime.__version__}, using {actual}")
    if actual != "CUDAExecutionProvider":
        print("  WARNING: features will be computed on CPU. This is the slowest stage")
        print("           of the run. Install onnxruntime-gpu (>=1.19 for CUDA 12) and")
        print("           make sure cuDNN is on the library path.")


# generate_runon_samples stays here: its cut logic needs RUNON_SPEEDS and
# RUNON_TAIL_MS above, which stay openWakeWord-local. microWakeWord has no run-on
# positives yet (the gap is documented in train/mww/corpus.py).


def generate_runon_samples(pool: "KokoroPool", voices: list, output_dir: Path,
                           per_voice: int, wake_word: str, desc: str,
                           reference: dict = None, workers: int = 2,
                           batch: int = 16):
    """Positives where the phrase runs straight into a command.

    The model measured in the tuning log detects 97% of "hey seeree, what's the time?"
    (comma, so the TTS puts a pause in) but only 83% of "hey seeree what's the time?"
    spoken as one breath. Splicing a command onto a separately-recorded phrase does not
    reproduce that - the final syllable has to actually be coarticulated into the next
    word, which means rendering the whole thing as one utterance.

    The clip is then CUT shortly after the phrase, and that is the careful part.
    create_fixed_size_clip aligns the END OF THE ARRAY with the end of the detection
    window, so a whole "hey seeree what's the time" would land the wake word ~1.5s
    before the window end - outside the window once truncated to 2s. Cutting just past
    the phrase leaves the command's onset as trailing context and keeps the phrase
    where the window expects it.

    The cut point comes from a phrase-alone rendering at the same voice and speed,
    cached per (voice, speed). Coarticulation makes the phrase slightly shorter inside
    the run-on than alone, so the cut lands a little way into the command - the
    intent, with deliberate jitter on top.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = {} if reference is None else reference
    # The fallback cache is read and written from several threads, and a miss costs
    # a TTS call, so guard it rather than racing to make the same call twice.
    reference_lock = threading.Lock()
    fallbacks = []

    # Group by (voice, speed) so a batch can share one request, then chunk. Each
    # clip keeps its own tail jitter, drawn here so the corpus stays a function of
    # the seed rather than of thread scheduling.
    from collections import defaultdict
    buckets = defaultdict(list)
    total = 0
    for voice in voices:
        for i in range(per_voice):
            speed = RUNON_SPEEDS[i % len(RUNON_SPEEDS)]
            command = TRAINING_COMMANDS[(i // len(RUNON_SPEEDS)) % len(TRAINING_COMMANDS)]
            tail = int(16000 * np.random.uniform(*RUNON_TAIL_MS) / 1000)
            buckets[(voice, speed)].append((command, tail))
            total += 1

    jobs = []
    for (voice, speed), group in buckets.items():
        for i in range(0, len(group), batch):
            jobs.append((voice, speed, group[i:i + batch]))

    def cut_and_write(data, timestamps, tail, voice, speed, kokoro_url, text):
        """Cut one run-on clip just past the wake word and write it."""
        cut = phrase_end_sample(timestamps, wake_word) if data is not None else None

        if cut is None:
            # Fallback for a server without /dev/captioned_speech: infer the
            # boundary from a phrase-alone rendering, cached per (voice, speed).
            # Measured at a median +153 ms late and voice-dependent, so it is a
            # degraded mode rather than an equivalent one.
            if data is None:
                data = kokoro_tts(kokoro_url, voice, text, speed)
            key = (voice, speed)
            with reference_lock:
                if key not in reference:
                    alone = kokoro_tts(kokoro_url, voice, wake_word, speed)
                    reference[key] = len(trim_silence(alone)) if alone is not None else None
                phrase_len = reference[key]
            if data is not None and phrase_len:
                data = trim_silence(data)
                cut = phrase_len - int(16000 * 30.0 / 1000)   # drop the trim pad
                fallbacks.append(1)

        if data is None or not cut:
            return 0
        data = data[:cut + tail] if cut + tail < len(data) else data
        scipy.io.wavfile.write(
            str(output_dir / f"runon_{voice}_{uuid.uuid4().hex}.wav"), 16000, data)
        return 1

    def render(job):
        voice, speed, group = job
        kokoro_url = pool.next()
        texts = [f"{wake_word} {command}" for command, _ in group]

        if batch == 1:
            results = [kokoro_tts_timed(kokoro_url, voice, texts[0], speed)]
        else:
            results = kokoro_tts_batch(kokoro_url, voice, texts, speed)

        written = 0
        for (data, timestamps), (command, tail), text in zip(results, group, texts):
            written += cut_and_write(data, timestamps, tail, voice, speed,
                                     kokoro_url, text)
        return written

    success = run_jobs(jobs, render, desc, workers * len(pool),
                       weights=[len(g) for _, _, g in jobs])

    print(f"  Generated {success}/{total} run-on samples "
          f"({len(jobs)} request(s), batch {batch})")
    if fallbacks:
        print(f"  NOTE: {len(fallbacks)} clip(s) fell back to the phrase-alone estimate "
              f"({len(reference)} reference renderings).")
        print("        /dev/captioned_speech was unavailable or its words did not match")
        print("        the wake phrase, so those cuts sit later than they should.")
    return success


# copy_real_samples, time_stretch, vocal_tract_shift, add_child_range_copies,
# trim_silence and trim_directory live in corpus/real.py and corpus/augment.py.


def _tag_input(resolved):
    """The config half of the run tag: every hyperparameter, no paths.

    Paths describe the machine, not the run: hashing an absolute path would move
    the tag when the repo moves, and the corpus paths are the corpus half's job
    anyway. What remains is exactly what a sweep point varies - steps, geometry,
    weights, the augmentation rounds, and the seed.
    """
    not_hyper = {"output_dir", "corpus_dir", "rir_paths", "background_paths",
                 "feature_data_files", "false_positive_validation_data_path",
                 "target_phrase", "model_name"}
    return {k: v for k, v in resolved.items() if k not in not_hyper}


def convert_to_tflite(model_path: Path):
    """Convert the exported .onnx with this repo's converter. Returns the path or None.

    UPSTREAM'S CONVERSION CANNOT RUN HERE. openwakeword's train.py finishes by
    calling convert_onnx_to_tflite, which imports onnx_tf - part of the
    tensorflow-cpu 2.8.1 / tensorflow_probability / onnx_tf trio this image
    deliberately does not install (it never resolved against protobuf >= 3.20). So it
    exits 1 AFTER the .onnx is safely written - the whole reason the freshness check
    exists rather than trusting the exit code.

    Doing it here means the .tflite arrives in the same run, from the converter that
    VERIFIES the result: onnx2tflite tries each axis adaptation, scores it against the
    source ONNX on random inputs, and refuses a model that disagrees. That check is
    not optional care - onnx2tf's output axis order varies by version, and a wrong-axis
    tflite loads cleanly, reports a plausible shape, and returns plausible 0-1 scores
    while detecting nothing at all.

    A FAILURE HERE DOES NOT FAIL THE RUN. The .onnx is what eval/ and
    run-oww-training.sh work with; the .tflite is for preflight and the deployment
    runtime, and can be produced later from the same .onnx without retraining.
    """
    tflite_path = model_path.with_suffix(".tflite")
    print(f"Converting to tflite: {tflite_path.name}")
    try:
        # A SUBPROCESS, not an in-process import: the converter pulls in
        # tensorflow, and loading it into this process after a long torch/OpenMP
        # run aborts the whole process on macOS arm64 (measured twice: 2026-09-07
        # 12:43 run, and the 2026-09-21 bar-test run A - "mutex lock failed" SIGABRT,
        # which a try/except cannot catch). In a child the abort dies with the
        # child; the .onnx is already what this run produced.
        r = subprocess.run(
            [sys.executable, "-m", "train.oww.onnx2tflite", str(model_path),
             "-o", str(tflite_path)],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
        if r.returncode != 0:
            raise RuntimeError(
                f"converter exited {r.returncode}: "
                + (r.stderr.strip().splitlines() or ["no output"])[-1])
        for line in reversed(r.stdout.strip().splitlines()):
            if "max diff" in line:
                print(f"  verified against the source ONNX, {line.split('max diff')[-1].strip()}")
                break
        return tflite_path
    except Exception as exc:                                         # noqa: BLE001
        print(f"  WARNING: tflite conversion failed - {type(exc).__name__}: {exc}")
        print("  The .onnx is unaffected and is this run's model. Convert later with:")
        print(f"    python -m train.oww.onnx2tflite {model_path}")
        return None


def setup_training_dirs(wake_word: str, skip_corpus: bool = False) -> Path:
    """Set up training directory structure.

    data/corpus/<wake_word>/oww/ - beside the microWakeWord corpus at .../mww/,
    without either pipeline reaching into the other's directory.

    skip_corpus keeps what is already there, for a re-run after a failure downstream
    of generation - the OOM at the feature array is the motivating case. It VERIFIES
    rather than trusts: an empty or partial corpus trains a model on nothing and
    reports excellent accuracy, so a missing class is a hard error here.

    THE CORPUS AND THE MODELS ARE SEPARATE TREES, and this function is why. It
    rmtree's its base directory on every run. The base used to share a tree with the
    models run-oww-training.sh archives, so a run could delete the archive - it did
    until the corpus nested a level deeper. With the corpus under data/ and models
    under output/, this rmtree is confined to generated audio the next run would
    rebuild anyway and cannot reach a model at all.
    """
    safe_name = wake_word.replace(" ", "_").lower()
    base_dir = WORK_DIR / "data" / "corpus" / safe_name / "oww"
    subdirs = ["positive_train", "positive_test", "negative_train", "negative_test"]

    if skip_corpus:
        counts = {d: len(list((base_dir / d).glob("*.wav"))) for d in subdirs}
        empty = [d for d, n in counts.items() if n == 0]
        if empty:
            print(f"ERROR: --skip-corpus, but {base_dir} has no clips in: "
                  f"{', '.join(empty)}")
            print("       Re-run without --skip-corpus to build it.")
            sys.exit(1)
        print("Reusing existing corpus (--skip-corpus):")
        for d in subdirs:
            print(f"  {d}: {counts[d]} clips")
        return base_dir

    if base_dir.exists():
        print("Clearing previous training outputs...")
        shutil.rmtree(base_dir)

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir


def create_config(wake_word: str, n_samples: int, training_steps: int,
                  layer_size: int, data_dir: str, augmentation_rounds: int = 3,
                  max_negative_weight: int = 2000, seed: int = 0,
                  lr: float = 0.0001, batch_n_per_class: int = 1024,
                  target_fp_per_hour: float = 0.1,
                  target_accuracy: float = 0.7, target_recall: float = 0.5,
                  n_samples_val: int = None,
                  augmentation_batch_size: int = 16,
                  output_dir: Path = None,
                  overrides: list = None):
    """Create training configuration."""
    safe_name = wake_word.replace(" ", "_").lower()

    # Load default config from OpenWakeWord
    default_path = WORK_DIR / "openwakeword/examples/custom_model.yml"
    with open(default_path, 'r') as f:
        config = yaml.load(f.read(), yaml.Loader)

    config["target_phrase"] = [safe_name]
    config["model_name"] = safe_name
    config["n_samples"] = n_samples
    # None = the pre-P1.5 behaviour: validation set is a tenth of training, floored
    # at 1000. An explicit value is what a sweep point moves.
    config["n_samples_val"] = (n_samples_val if n_samples_val is not None
                               else max(1000, n_samples // 10))
    config["steps"] = training_steps
    config["layer_size"] = layer_size
    config["target_accuracy"] = target_accuracy
    config["target_recall"] = target_recall
    # Two consumers inside auto_train: checkpoint selection, and the weight
    # doubling - each sequence that ends with best_val_fp above this target
    # doubles max_negative_weight (it can fire twice, so a run requesting 2000
    # can train at 8000). Upstream's default is 0.2; this repo wrote 0.1 at
    # creation and has never measured the departure. The EFFECTIVE per-sequence
    # weights are now audited into <tag>.config.json (patches/
    # log-weight-and-merge.py), so a sweep against this target finally has
    # its actual weights on the record.
    config["target_false_positives_per_hour"] = target_fp_per_hour
    # auto_train's base learning rate. Upstream hardcoded 0.0001 inside the
    # method - not a config key, not a flag - so this key and
    # patches/configurable-lr.py exist for it: the method now reads
    # config.get("lr", 0.0001) and the per-sequence /10 lines scale from it.
    # 0.0001 is upstream's value, untouched in this repo, so the default is a
    # no-change.
    config["lr"] = lr
    # Per-batch class balance: every step draws this many ACAV100M negatives
    # against 50 positives and 50 adversarial negatives (the latter two stay at
    # custom_model.yml's values; --set batch_n_per_class={...} is how a sweep
    # moves those). 1024/50/50 is upstream's number, never touched here - the
    # balance is a major lever and the key had to match feature_data_files
    # (custom_model.yml's batch_n_per_class, not a top-level int). The total
    # batch size is the sum of the dict values.
    config["batch_n_per_class"] = {
        "ACAV100M_sample": batch_n_per_class,
        "adversarial_negative": 50,
        "positive": 50,
    }
    # Explicit rather than riding the yml default (16): a config value a sweep
    # can see is a config value it can move, and the yml's own comment cautions
    # against making it large - variety in the augmentation is the point.
    config["augmentation_batch_size"] = augmentation_batch_size
    # MODELS OUT, CORPUS ELSEWHERE. Upstream uses output_dir for exactly three things
    # (openwakeword/train.py:652, 905, 909): the .onnx export, the .tflite conversion
    # beside it, and an empty <output_dir>/<model_name>/ it creates unconditionally -
    # that last one is where the corpus WOULD have gone; its being empty is the
    # visible sign corpus_dir took over. Everything else formerly derived from
    # output_dir is re-pointed by patches/configurable-corpus-dir.py, which is what
    # makes corpus_dir exist.
    # BOTH ABSOLUTE, AND corpus_dir MUST BE. Upstream runs os.path.abspath() on
    # output_dir (train.py:649) but knows nothing about corpus_dir, so a relative
    # value survives into trim_mmap, which builds its temp file like this:
    #
    #     output_file2 = mmap_path.strip(".npy") + "2.npy"       # data.py:876
    #
    # str.strip takes a CHARACTER SET and strips BOTH ends, so a leading "./" loses
    # its dot and a relative path silently becomes absolute at the filesystem root:
    #
    #     ./data/corpus/hey_seeree/oww/positive_features_train.npy
    #     ->  /data/corpus/hey_seeree/oww/positive_features_trai2.npy
    #
    # which fails with FileNotFoundError during feature computation - after corpus
    # generation and augmentation have already run. (The mangled "trai" is the same
    # bug eating the "n"; harmless once the directory is right.)
    config["output_dir"] = (str(output_dir) if output_dir is not None
                            else str(WORK_DIR / "output" / safe_name / "oww"))
    config["corpus_dir"] = str(WORK_DIR / "data" / "corpus" / safe_name / "oww")

    # CREATE output_dir OURSELVES, ALL OF IT. Upstream makes it with os.mkdir
    # (train.py:650-651), which creates ONE level - fatal now that it is three deep:
    #
    #     FileNotFoundError: [Errno 2] No such file or directory:
    #         '/app/output/hey_seeree/oww'
    #
    # It fails inside the augmentation subprocess, after corpus generation has
    # already run, so the cost is the whole generation stage. corpus_dir needs no
    # equivalent: patches/configurable-corpus-dir.py creates it with os.makedirs.
    (WORK_DIR / config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # End of a linear ramp: the negative-class loss weight grows from 1 to this over
    # training (openwakeword/train.py:274), so higher penalises false positives harder.
    #
    # Run 8 tried 4000: fewer false accepts (7/32 -> 3/32 at threshold 0.5) but worse
    # detection (held-out plain 89% -> 77%, run-on 56% -> 37%). At MATCHED
    # false-accept counts the two models trade places without either dominating - the
    # weight mostly moved the operating point along the same curve, which the
    # detection threshold does for free and without a retrain. Back at 2000.
    #
    # Raise this only if the deployment cannot tune its threshold.
    config["max_negative_weight"] = max_negative_weight

    # Each round re-augments every clip with a different impulse response, background
    # and gain, multiplying the distinct feature vectors at no extra TTS. It matters
    # because training draws 50 positives per step for 50,000 steps - 2.5M draws
    # against ~14k clips, ~180 revisits each, and at one round those are 180 views
    # of an identical vector.
    #
    # Needs patches/honour-augmentation-rounds.py: upstream multiplies the clip list
    # by this value but sizes the output array from the unmultiplied directory, so
    # without the patch the extra rounds are computed and discarded.
    config["augmentation_rounds"] = augmentation_rounds
    config["rir_paths"] = [f'{data_dir}/mit_rirs']
    config["background_paths"] = [f'{data_dir}/audioset_16k', f'{data_dir}/fma']
    config["false_positive_validation_data_path"] = f"{data_dir}/validation_set_features.npy"
    config["feature_data_files"] = {"ACAV100M_sample": f"{data_dir}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
    config.pop("piper_sample_generator_path", None)  # We use Kokoro, not Piper

    # SEED THE SUBPROCESS. openwakeword's train.py (via patches/seed-augment.py)
    # seeds its random/numpy/torch from this key and passes it to augment_clips.
    # 0 = unseeded = upstream behaviour, which is what a config without the patch
    # also does, so the two cannot diverge.
    config["seed"] = seed

    # The --set escape hatch, applied LAST: it overrides every key set above,
    # so a sweep runner needs no per-knob plumbing for anything in this config.
    # Values parse as JSON, falling back to the raw string when that fails
    # ("--set target_phrase=hey seeree" keeps the spaces). Whatever lands here
    # is part of the resolved config, so _tag_input hashes it into the run tag
    # and it is filed verbatim in <tag>.config.json - a --set point is a
    # first-class sweep point, named like the explicit flags.
    if overrides:
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep or not key:
                sys.exit(f"--set takes key=value, got {item!r}")
            try:
                config[key] = json.loads(raw)
            except ValueError:
                config[key] = raw
        print(f"Config overrides applied last: {', '.join(overrides)}")

    config_path = WORK_DIR / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Config saved: {config_path}")
    return config


def run_augmentation(overwrite: bool = False):
    """Run OpenWakeWord augmentation pipeline.

    `overwrite` passes upstream's --overwrite: the feature arrays get recomputed
    even though they exist. The caller decides from the features.json sidecar
    (see main): the .npy files are keyed on (corpus, rounds, total_length, seed),
    and a sidecar that does not match is stale, not a cache hit.
    """
    print("\n" + "=" * 60)
    print("Running augmentation pipeline...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    cmd = [sys.executable, train_script,
           "--training_config", "training_config.yaml",
           "--augment_clips"]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, env=_trainer_env())


def _trainer_env():
    """The subprocess environment: PYTHONHASHSEED pinned to the run's seed.

    hash() randomization is on by default and changes dict/set iteration order
    per process, which upstream's generators turn into a different draw order. 0
    (the deterministic value) when the run itself is unseeded is still a choice:
    two unseeded runs should at least not differ in the one place they can be
    made to agree for free.
    """
    return {**os.environ, "PYTHONHASHSEED": str(_SEED)}


def wait_for_kokoro_shutdown(timeout: int = 120):
    """Block until the Kokoro servers have released their VRAM.
    """
    import socket

    import torch
    if not torch.cuda.is_available():
        print("  no CUDA - not waiting for Kokoro (nothing holds VRAM here)")
        return

    # The NATIVE FastAPI ports, not the protocol ones (8899/8901): this wait is
    # about VRAM, and VRAM is held by the in-image FastAPI process, which still
    # listens on 8880 in BOTH kokoro containers - 8881 is only the host-side
    # mapping of kokoro2's 8880 ("8881:8880" in docker-compose.yml), and the
    # names here resolve on the compose network, where both answers come off
    # 8880. The wrapper in front of it holds nothing worth waiting for.
    servers = [("kokoro", 8880), ("kokoro2", 8880)]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = []
        for host, port in servers:
            try:
                socket.create_connection((host, port), 1).close()
                alive.append(f"{host}:{port}")
            except OSError:
                pass
        if not alive:
            return
        time.sleep(2)

    print(f"  WARNING: {', '.join(alive)} still answering after {timeout}s.")
    print("           Training will allocate with their VRAM still held, which is")
    print("           what caused the 16.09 GiB OOM this wait exists to prevent.")


def report_free_vram():
    """Print free VRAM going into training, so the next OOM is diagnosable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        print(f"  VRAM: {free / gib:.2f} GiB free of {total / gib:.2f} GiB")
    except Exception as e:
        print(f"  Could not read VRAM: {e}")


# The machine-readable audit lines patches/log-weight-and-merge.py makes the
# upstream trainer print at the moments of its two hidden decisions - the
# per-sequence negative-weight (the doubling, improvement.md P1.3) and the
# checkpoint merge (P1.4). The wrapper captures them in-process as the
# subprocess streams; reading the run log after the fact is not an option:
# the log exists only when the run script's `script -q` wrote it, through a
# pty (with escape codes), and the direct-invocation path has no log at all.
_AUDIT_RE = re.compile(r"^#\s*(WEIGHT_AUDIT|MERGE_AUDIT)\s+(\S.*)$")


def run_training():
    """Run OpenWakeWord model training.

    Returns (returncode, audit). audit maps "weight"/"merge" to the raw
    audit lines captured from the subprocess stdout; main() files them in
    <tag>.config.json AFTER computing the run tag (outcomes, not inputs -
    see main).
    """
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)

    # Both BEFORE the subprocess: it allocates the feature array on its first
    # breath, so anything reported afterwards describes a decision already made.
    wait_for_kokoro_shutdown()
    report_free_vram()

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    # stdout is CAPTURED and echoed line by line: the echo keeps the bar and
    # audit lines in the run log exactly as before (stdout inherited), and the
    # capture is what lets main() file the audit lines in <tag>.config.json.
    # stderr is inherited: torch's dataloader-fork warnings are the only
    # output there and the run script's log already records them.
    proc = subprocess.Popen(
        [
            sys.executable, train_script,
            "--training_config", "training_config.yaml",
            "--train_model",
        ],
        env=_trainer_env(),
        stdout=subprocess.PIPE,
        text=True,
    )
    audit = {"weight": [], "merge": []}
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            m = _AUDIT_RE.match(line)
            if m:
                (audit["weight"] if m.group(1) == "WEIGHT_AUDIT"
                 else audit["merge"]).append(m.group(2).strip())
    finally:
        proc.stdout.close()
    returncode = proc.wait()
    return returncode, audit


def _parse_audit_lines(audit):
    """The captured audit lines as <tag>.config.json values.

    effective_max_negative_weight is the PER-SEQUENCE list (requested /
    doubled / effective per sequence): the honest form, since sequences 2
    and 3 each see a different weight (P1.3). merged_checkpoints is the
    steps list from the merge summary line (P1.4) - a present-but-EMPTY
    list means the gate ran and nothing cleared it (upstream then exports
    the live model as-is), which is a different record from the null filed
    when the line is missing entirely (patch not applied).
    Returns (weight, steps, merge_seen).
    """
    weight = []
    for line in audit["weight"]:
        kv = dict(p.split("=", 1) for p in line.split())
        weight.append({
            "sequence": int(kv["sequence"]),
            "requested": int(kv["requested"]),
            "doubled": kv["doubled"].lower() == "true",
            "effective": int(kv["effective"]),
        })
    steps, merge_seen = [], False
    for line in audit["merge"]:
        if line.startswith("cleared="):
            merge_seen = True
            kv = dict(p.split("=", 1) for p in line.split())
            steps = [int(s) for s in kv["steps"].split(",") if s]
    return weight, steps, merge_seen


def main():
    parser = argparse.ArgumentParser(description="Train a custom OpenWakeWord model")
    parser.add_argument("--wake-word", default="hey seeree", help="Wake word/phrase to train")
    parser.add_argument("--samples-per-voice", type=int, default=300,
                        help="Samples per Kokoro voice (default: %(default)s). Raised from\n                             200 when the wordlists grew: 59 negative phrases and 60\n                             run-on command/speed combinations need more renderings each\n                             to keep per-item density up.")
    parser.add_argument("--training-steps", type=int, default=50000,
                        help="Steps for the first training sequence (default: "
                             "%(default)s). openwakeword scales warmup, hold and the "
                             "negative-weight ramp from this, so raising it also "
                             "slows how fast the negative weight climbs. Run 11 tried "
                             "100k: detection looked better at threshold 0.5, but at "
                             "MATCHED false-accept rates it was worse everywhere. It "
                             "moved the operating point, it did not improve the "
                             "model. Measured in tuning run 11.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: the full pipeline SHAPE with the expensive "
                             "parts minified and the corpus REUSED, not regenerated "
                             "- the end-to-end check for a changed train/ tree, in a "
                             "few minutes instead of the ~35 min full host run. "
                             "What changes: the corpus is REUSED (implies "
                             "--skip-corpus: the corpus stage is skipped, though the "
                             "catalog probe still runs for the reuse check - the "
                             "engines must be reachable, nothing is rendered - and "
                             "the manifest check still applies: a differently-shaped "
                             "or differently-voiced corpus is refused exactly as "
                             "with any reuse), the feature "
                             "arrays are RECOMPUTED even when the features.json "
                             "sidecar matches (the feature stage is the one this "
                             "exists to run; generation is the 30-54 min TTS stage "
                             "and its health is probed at the start of a normal "
                             "run anyway), training runs 200 steps instead of the "
                             "50,000-step default (~30 s against ~8 min), and the "
                             "tflite conversion still runs for real. Output goes to "
                             "a clearly-named smoke directory (output/<wake>/oww/"
                             "smoke-<timestamp>/ by default, --smoke-output to "
                             "pick); the canonical <wake>.onnx, its .last_run_tag "
                             "and the archive that the container script keys on "
                             "that file are all left untouched, so a smoke can "
                             "never be mistaken for a real run or clobber one. "
                             "The run tag's config half differs from a real run's "
                             "because the step count changed - expected, and the "
                             "smoke directory name says what it is anyway. The "
                             "results are NOT measurable: do not evaluate or "
                             "deploy a smoke model. Combining --smoke with an "
                             "explicit real-size flag (--training-steps, "
                             "--n-samples-val, --set, --corpus rebuild) is an "
                             "error: you asked for both a smoke and a real size.")
    parser.add_argument("--smoke-output", default=None,
                        help="Where a --smoke run writes its model (default: "
                             "output/<wake>/oww/smoke-<timestamp>/). A SIBLING of "
                             "the canonical <wake>.onnx, never its replacement: "
                             "a smoke run cannot clobber the model the archive "
                             "points at, no matter how long the smoke lives.")
    parser.add_argument("--layer-size", type=int, default=64, choices=[32, 64, 128], help="Network layer size")
    parser.add_argument("--kokoro-url", default=os.environ.get("KOKORO_URL", "tcp://127.0.0.1:8899"),
                        help="Kokoro TTS server(s) as tcp:// protocol URL(s), "
                             "comma-separated to split the work across them: one "
                             "Kokoro process is single-threaded and saturates one "
                             "core, so more PROCESSES scale where more client "
                             "threads do not. Each URL is a tts-protocol server "
                             "(tts-service/), which fronts the actual engine - "
                             "kokoro-mlx on a Mac, Kokoro-FastAPI in Docker. "
                             "The old http:// and mlx:// forms are rejected: they "
                             "used to mean different backends with different "
                             "audio.")
    parser.add_argument("--data-dir", default="data/external",
                        help="Where the third-party downloads live: the ACAV100M "
                        "and validation feature .npy files, audioset_16k, fma, "
                        "mit_rirs (default: %(default)s)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Skip silence trimming before augmentation (not recommended)")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Reuse data/corpus/<wake>/oww/ instead of regenerating it. "
                             "For resuming a run that failed AFTER generation - the "
                             "CUDA OOM at the feature array is the usual reason. Skips "
                             "TTS, real-clip copying and trimming. Augmentation and the "
                             "feature arrays are reused TOO when the .npy files exist "
                             "and the features.json sidecar matches this run's "
                             "(corpus, augmentation_rounds, seed) - use "
                             "--rebuild-features to force recomputation. "
                             "Sample-shaping flags (--samples-per-voice, --piper-fraction, "
                             "--child-fraction, --runon-fraction) are IGNORED: the clips "
                             "already exist and this does not rebuild them. When a "
                             "corpus.json manifest exists it is CHECKED against the "
                             "requested shaping flags and the effective voice set and "
                             "a mismatch refuses the run - reusing a differently-shaped "
                             "corpus or one built from a different voice set silently "
                             "would be the measurement the flag exists to prevent (the "
                             "voice set is checked because the cf9c065b reuse, "
                             "2026-09-22, matched on every shaping flag and still "
                             "trained on held-out voices). Checking it means the TTS "
                             "catalogs must be REACHABLE for a resume - the probe is "
                             "a cheap catalog fetch, but it is what the check reads.")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Recompute the augmentation/feature .npy arrays even "
                             "though they exist and their sidecar matches. Needed "
                             "after a code change to the feature path; the sidecar "
                             "keys on (corpus digest, augmentation_rounds, seed), not "
                             "on the code, which the run tag's code half covers.")
    parser.add_argument("--negatives-file",
                        help="Text file of confusable negative phrases, one per line "
                             "(# comments allowed). Overrides the built-in list for "
                             "this wake word; base negatives are always included.")
    parser.add_argument("--tts-workers", type=int, default=2,
                        help="Concurrent requests PER SERVER (default: %(default)s). "
                             "A Kokoro process handles one at a time, so this only "
                             "covers the gap between responses; total concurrency is "
                             "this times the number of servers.")
    parser.add_argument("--tts-batch", type=int, default=16,
                        help="Utterances per Kokoro request (default: %(default)s). "
                             "A short request is ~75%% fixed overhead, so batching "
                             "is ~3x faster at realistic bucket sizes; clips are split apart exactly, on the "
                             "server's word timestamps. 1 disables it. Every clip in "
                             "a batch shares a voice and speed, which is why plain "
                             "speeds come from a grid rather than a continuous draw.")
    parser.add_argument("--real-copies", type=int, default=10,
                        help="How many times each real recording is duplicated into "
                             "the positive set (default: %(default)s). Weighting, "
                             "not augmentation - watch the held-out set for "
                             "overfitting to the specific clips.")
    parser.add_argument("--max-negative-weight", type=int, default=2000,
                        help="How hard false positives are penalised by the end of "
                             "training (default: %(default)s). Higher trades "
                             "detection for precision - but so does the detection "
                             "threshold, for free. Measured in tuning run 8, which "
                             "compared requested 2000 vs 4000; the weight doubling "
                             "is unconditional (best_val_fp is never updated - see "
                             "patches/log-weight-and-merge.py and improvement.md "
                             "P1.3), so that was effective 2000/4000/8000 vs "
                             "4000/8000/16000 per sequence: a fair comparison, 4000 "
                             "was not better. Recommend to leave at default.")
    parser.add_argument("--augmentation-rounds", type=int, default=3,
                        help="How many differently-augmented copies of each clip to "
                             "compute features for (default: %(default)s). Multiplies "
                             "training data at no TTS cost.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Base learning rate for auto_train's first sequence "
                             "(default: %(default)s). Upstream hardcodes it inside "
                             "the method - not a config key, not a flag - until "
                             "patches/configurable-lr.py, so it was the one "
                             "hyperparameter a sweep had no flag for. The "
                             "per-sequence /10 decay is kept: sequence 2 trains at "
                             "lr/10 and sequence 3 at lr/100, so this moves the "
                             "whole schedule, not just the first third. "
                             "0.0001 is upstream's value, never touched in this "
                             "repo, so the default is a no-change. The resolved "
                             "value is part of the run tag (provenance.config_tag).")
    parser.add_argument("--batch-n-per-class", type=int, default=1024,
                        help="ACAV100M negative draws per training batch (default: "
                             "%(default)s, the upstream custom_model.yml value). "
                             "Every step draws this many background negatives "
                             "against 50 positives and 50 adversarial negatives - "
                             "the per-batch class balance is a major lever nobody "
                             "in this repo has touched. Those two stay at the "
                             "yml's 50/50; move them with "
                             "--set batch_n_per_class={\"positive\": N} - the "
                             "total batch size is the sum of the three.")
    parser.add_argument("--target-fp-per-hour", type=float, default=0.1,
                        help="Target false accepts per hour on the validation set "
                             "(default: %(default)s). auto_train uses it two ways: "
                             "it gates checkpoint selection, and a sequence that "
                             "ends with best_val_fp above it DOUBLES "
                             "max_negative_weight - it can fire twice, so a run "
                             "requesting --max-negative-weight 2000 can train at "
                             "8000. The effective per-sequence weights are audited "
                             "into <tag>.config.json. Upstream's default is 0.2; "
                             "this repo's 0.1 is an unmeasured departure from it.")
    parser.add_argument("--target-accuracy", type=float, default=0.7,
                        help="Checkpoint-merge gate for validation accuracy "
                             "(default: %(default)s, the value this repo has "
                             "always written).")
    parser.add_argument("--target-recall", type=float, default=0.5,
                        help="Checkpoint-merge gate for validation recall "
                             "(default: %(default)s, the value this repo has "
                             "always written).")
    parser.add_argument("--n-samples-val", type=int, default=None,
                        help="Validation clips to generate (default: "
                             "%(default)s = max(1000, n_samples/10), the "
                             "pre-P1.5 behaviour).")
    parser.add_argument("--augmentation-batch-size", type=int, default=16,
                        help="Batch size for the augmentation pass over the "
                             "generated clips (default: %(default)s, the upstream "
                             "yml value). Upstream's comment cautions against "
                             "making it large - variety in the augmentation is "
                             "the point.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Generic override, repeatable: KEY=VALUE merged into "
                             "the generated config LAST, so it overrides every "
                             "flag above. VALUE parses as JSON (true, 3, [1, 2], "
                             "{\"a\": 1}), falling back to the raw string when "
                             "that fails. Escape hatch: a sweep runner needs no "
                             "per-knob plumbing - the resolved result is what "
                             "gets hashed into the run tag (provenance.config_tag) "
                             "and filed as <tag>.config.json, so a --set point is "
                             "a first-class sweep point, named like the explicit "
                             "flags.")
    parser.add_argument("--runon-fraction", type=float, default=0.4,
                        help="Fraction of positives where the phrase runs straight "
                             "into a command instead of being followed by quiet "
                             "(default: %(default)s). 0 disables them.")
    parser.add_argument("--exclude-voices", default="",
                        help="Comma-separated Kokoro voices to skip, added to the "
                             "built-in MISPRONOUNCING_VOICES list for this wake "
                             "word. Use for a wake word with no built-in entry.")
    parser.add_argument("--include-legacy-voices", action="store_true",
                        help="Keep Kokoro's v0 voices, which are skipped by default. "
                             "They are older renderings of speakers already in the "
                             "set (af_v0bella beside af_bella), so they cost ~36%% of "
                             "the Kokoro clips and measured no accuracy gain - a "
                             "22-voice corpus scored 82/65 against a 36-voice one's "
                             "76/56 at 4 adversarial false accepts. Use this to "
                             "reproduce a corpus generated before they were dropped.")
    parser.add_argument("--child-fraction", type=float,
                        default=CHILD_STRETCH_FRACTION,
                        help="Fraction of Kokoro positives that get an ADDITIONAL "
                             "pitch/formant-shifted copy, to cover child and "
                             "adolescent voices (default: %(default)s). 0 disables "
                             "it, restoring the pre-run-12 adult-only corpus.")
    parser.add_argument("--piper-fraction", type=float, default=0.0,
                        help="Fraction of the PHRASE-ALONE positive budget rendered "
                             "by Piper instead of Kokoro (default: %(default)s = "
                             "off). SUBSTITUTES rather than adds, so corpus size, "
                             "the plain/run-on split and real-clip density are all "
                             "unchanged - which is what makes a comparison against "
                             "the previous run mean anything. Run-ons stay Kokoro: "
                             "their cut point comes from Kokoro's word timestamps, "
                             "and Wyoming has no equivalent, so Piper would fall "
                             "back to inferring it from a phrase-alone rendering - "
                             "measured at a median +153 ms late, against a "
                             "RUNON_TAIL_MS of 150-300 ms. That is run 14's "
                             "alignment regression waiting to happen.")
    parser.add_argument("--piper-url",
                        default=os.environ.get("PIPER_URL", "tcp://127.0.0.1:8898"),
                        help="Piper protocol server(s), tcp:// URLs, "
                             "comma-separated to run a fleet (scripts/"
                             "start-tts-fleet.sh N launches N on this Mac): "
                             "the corpus is sharded BY VOICE, each model pinned "
                             "to one instance for the whole run, because an "
                             "instance holds one model resident and reloads on "
                             "a switch - round-robin would buy nothing "
                             "(corpus/piper.py, PiperFleet) "
                             "(default: %%(default)s)")
    parser.add_argument("--piper-speakers", type=int, default=12,
                        help="Speakers to sample per multi-speaker Piper voice, "
                             "evenly spaced (default: %(default)s). "
                             "en_US-libritts_r-medium alone carries 904, which "
                             "would swamp the corpus with one model's g2p.")
    parser.add_argument("--piper-languages", default="en_US,en_GB",
                        help="Language prefixes to include (default: %(default)s)")
    parser.add_argument("--corpus", choices=["auto", "reuse", "rebuild"], default="auto",
                        help="Whether to rebuild the TTS corpus or reuse the one on "
                             "disk. auto (default): reuse when a corpus.json manifest "
                             "exists and matches the requested shaping flags AND the "
                             "effective voice set (live catalog minus exclusions "
                             "minus the voice holdout, resolved from the TTS "
                             "catalogs before the decision) - during a sweep the "
                             "corpus is a held-fixed independent variable, and "
                             "regenerating it redraws the TTS noise every point. "
                             "reuse: require that (a mismatch exits with a diff, no "
                             "silent rebuild - including a different voice set, the "
                             "axis the cf9c065b reuse, 2026-09-22, was blind on). "
                             "rebuild: regenerate unconditionally. "
                             "Replaces --skip-corpus as the front door; that flag "
                             "still works and means reuse.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for every stage of the run: corpus drawing, "
                             "augmentation, model init (default: %(default)s = "
                             "unseeded, and the log says so). Two runs at the same "
                             "seed, same corpus and same config must be comparable "
                             "bit for bit - until this landed, the 10-point "
                             "run-to-run variance measured at an identical config "
                             "was the seed, not the run. The seed is part of the "
                             "run's tag (provenance.config_tag). Note: the TTS "
                             "engines are NOT seedable, so a seeded run still "
                             "renders different audio - reuse the corpus instead "
                             "(--skip-corpus / --corpus reuse).")
    args = parser.parse_args()

    wake_word = args.wake_word
    safe_name = wake_word.replace(" ", "_").lower()

    # === SMOKE MODE: decided here, before any stage, so a contradiction costs
    # zero seconds. The whole contract is loud and local to this block.
    smoke_dir = None
    if args.smoke:
        # The smoke minifies a FIXED set of sizes; an explicit real size on the
        # command line is a contradiction, not an override - error out and do not
        # guess which one was meant. --set counts too: it is the generic size
        # escape hatch and would silently undo the minification.
        explicit = []
        if args.training_steps != parser.get_default("training_steps"):
            explicit.append("--training-steps")
        if args.n_samples_val is not None:          # the default is None (= derived)
            explicit.append("--n-samples-val")
        if args.set:
            explicit.append("--set")
        if args.corpus == "rebuild":
            explicit.append("--corpus rebuild")
        if explicit:
            sys.exit(f"ERROR: --smoke is combined with explicit real-size flags "
                     f"({', '.join(explicit)}). It minifies them itself; a smoke and "
                     f"a real size are two different runs. Drop the flag for the run "
                     f"you actually want.")
        args.training_steps = SMOKE_TRAINING_STEPS
        # Reuse is IMPLIED, not auto: auto would rebuild a corpus whose manifest
        # does not match, which is exactly the hour-long spend a smoke must not
        # make. A missing manifest fails here, as reuse does.
        args.corpus = "reuse"
        if args.smoke_output:
            smoke_dir = Path(args.smoke_output)
        else:
            smoke_dir = (WORK_DIR / "output" / safe_name / "oww"
                         / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        short, _, _, _, _ = provenance.corpus_tag(wake_word, "oww")
        print("=" * 60)
        print(f"SMOKE MODE: training minified ({args.training_steps} steps), "
              f"feature arrays RECOMPUTED, corpus REUSED (tag c{short}), "
              f"results are NOT measurable")
        print(f"  model goes to {smoke_dir}")
        print(f"  the canonical output/{safe_name}/oww/{safe_name}.onnx, its "
              f".last_run_tag and the archive are untouched")
        print("=" * 60)

    # Seed the parent process's draws (corpus stage: run-on tail jitter, child
    # stretch, Piper speed draws) BEFORE the corpus stage, so the drawing is a
    # function of the seed rather than of the clock. The openwakeword train.py
    # subprocess seeds itself from config["seed"] (patches/seed-augment.py); the
    # rendering is not covered - the TTS engines cannot be seeded, and that is
    # why a sweep reuses the corpus rather than regenerating it.
    global _SEED
    _SEED = args.seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # === VOICE SET: resolved BEFORE the corpus-mode decision ====================
    # The reuse check diffs the EFFECTIVE voice set, not only the shaping flags.
    # The cf9c065b reuse, 2026-09-22: every shaping flag matched the manifest,
    # but the tracked voice holdout (seven Kokoro voices + two Piper pairs)
    # postdated the corpus build, and matches_requested never compared
    # manifest["voices"] - so a post-reservation run silently trained on the
    # pre-reservation corpus. The probe is the cheap `voices` op (a catalog
    # fetch: no rendering, no model load), so a reuse run pays for it too;
    # deferring it to the build stage is exactly what kept the check
    # voice-blind, and it is the only way to know a catalog that has moved
    # since the build cannot be reused as if it had not. The consequence: a
    # --skip-corpus resume needs the TTS fleet UP (an unreachable catalog
    # exits here, as it does for a build).
    #
    # Computed ONCE and consumed by BOTH the check and the build below: two
    # computations of "live catalog minus exclusions" could drift from each
    # other and re-open the same hole with the check fixed.
    print("\n[Kokoro servers]")
    pool = KokoroPool(args.kokoro_url.split(","))
    kokoro_voices = probe_kokoro_servers(pool)
    if not kokoro_voices:
        print("ERROR: No Kokoro voices available!")
        sys.exit(1)
    print(f"  {len(pool)} server(s), {len(kokoro_voices)} shared English voices")

    excluded = set(MISPRONOUNCING_VOICES.get(safe_name, []))
    excluded.update(v.strip() for v in args.exclude_voices.split(",") if v.strip())

    # The v0 legacy voices, dropped by default. Reported separately from the
    # mispronouncing ones: those are excluded to protect accuracy, these to save
    # time that buys nothing. See LEGACY_VOICE_MARKER for the corpora that
    # measured it.
    legacy = sorted(v for v in kokoro_voices if LEGACY_VOICE_MARKER in v)
    if legacy and not args.include_legacy_voices:
        excluded.update(legacy)
        print(f"  Skipping {len(legacy)} v0 legacy voice(s) - older renderings of "
              f"speakers already in the set, {len(legacy) * 100 // len(kokoro_voices)}% "
              f"of the clips for no measured gain (--include-legacy-voices keeps them)")
    elif legacy:
        print(f"  Including {len(legacy)} v0 legacy voice(s) by request")

    if excluded:
        present = sorted(v for v in kokoro_voices if v in excluded)
        kokoro_voices = [v for v in kokoro_voices if v not in excluded]
        mispronouncing = sorted(set(present) - set(legacy))
        if mispronouncing:
            print(f"  Excluding {len(mispronouncing)} voice(s) that mispronounce "
                  f"the wake word: {', '.join(mispronouncing)}")
        print(f"  {len(kokoro_voices)} voices remain")
        missing = sorted(excluded - set(present))
        if missing:
            print(f"  NOTE: {', '.join(missing)} not offered by these servers anyway")
        if not kokoro_voices:
            print("ERROR: every available voice is excluded!")
            sys.exit(1)

    # THE VOICE HOLDOUT (improvement.md P1.2): the voices wordlists/
    # voice_holdout.yaml reserves for the synthetic ranking set are excluded
    # from every corpus build, so that set stays voice-disjoint from
    # training. The live catalog is the source of truth: an entry it no
    # longer offers means the tracked list has drifted from the engine, and
    # that is an error - the silent outcome is the exclusion ending up empty
    # and the corpus quietly training on a held-out voice.
    # No tracked file (a checkout predating it) is a no-op, and says so.
    # Runs BEFORE the corpus-mode decision because the exclusion is part of
    # what the reuse check validates: the cf9c065b reuse, 2026-09-22, was a
    # run whose effective voice set differed from the corpus on exactly this
    # line and nothing else.
    holdout = load_voice_holdout()
    holdout_kokoro = holdout.get("kokoro") or []
    if holdout_kokoro:
        n_before = len(kokoro_voices)
        kokoro_voices, holdout_missing = exclude_voice_holdout(
            "kokoro", kokoro_voices, holdout)
        if holdout_missing:
            sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) names "
                     f"Kokoro voice(s) the live catalog does not offer: "
                     f"{holdout_missing}. The catalog is the source of "
                     f"truth - update or delete the stale entries in the "
                     f"tracked list rather than rebuilding a corpus whose "
                     f"holdout cannot be enforced.")
        print(f"  Excluding {n_before - len(kokoro_voices)} voice-holdout "
              f"voice(s) reserved for the synthetic ranking set: "
              f"{', '.join(holdout_kokoro)}")
    else:
        print(f"  NOTE: no voice holdout at {voice_holdout_path()} - the "
              f"synthetic ranking set has no reserved voices")

    # The Piper half, resolved here for the same reason (and only when it is
    # in play - piper_fraction 0 renders no Piper clips, so the request and
    # the manifest both say `piper: []`, which is a match, not a hole).
    piper_voices = []
    if args.piper_fraction > 0:
        piper_voices = select_piper_voices(
            args.piper_url, wake_word,
            languages=tuple(args.piper_languages.split(",")),
            max_speakers=args.piper_speakers)
        # The Piper half of the voice holdout: excluded from the audited
        # selection, same fail-loudly rule as the Kokoro side above. "In the
        # catalog" here means in the AUDITED selection - a holdout pair the
        # server offers but the audit tables drop (mispronouncing, unaudited)
        # fails the check too, because that pair cannot serve as an eval
        # voice either, so the tracked list must move, not the audit.
        if holdout.get("piper"):
            piper_voices, holdout_missing = exclude_voice_holdout(
                "piper", piper_voices, holdout)
            if holdout_missing:
                sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) "
                         f"names Piper (voice, speaker) pair(s) the live "
                         f"audited selection does not carry: {holdout_missing}. "
                         f"Update the tracked list to match the catalog this "
                         f"corpus is built from.")

    # === CORPUS MODE: reuse the frozen corpus or rebuild it =====================
    # P0.3: during a sweep the corpus is a HELD-FIXED INDEPENDENT VARIABLE. The
    # default (auto) reuses it whenever a corpus.json manifest exists and matches
    # the requested shaping AND the resolved voice set above, so a tuning loop
    # does not redraw the TTS noise on every point; rebuilding is explicit
    # (--corpus rebuild), and so is reuse (which refuses rather than falling
    # back to a rebuild on a mismatch).
    # --skip-corpus is kept as the legacy spelling of reuse-for-resume.
    corpus_dir = WORK_DIR / "data" / "corpus" / safe_name / "oww"
    corpus_shaping = {
        "samples_per_voice": args.samples_per_voice,
        "runon_fraction": args.runon_fraction,
        "child_fraction": args.child_fraction,
        "piper_fraction": args.piper_fraction,
        "real_copies": args.real_copies,
        "piper_speakers": args.piper_speakers,
        "piper_languages": args.piper_languages,
        "negatives_file": args.negatives_file,
        "exclude_voices": args.exclude_voices,
        "include_legacy_voices": args.include_legacy_voices,
        "no_trim": args.no_trim,
    }
    # The voice set is a TOP-LEVEL axis of the request (manifest["voices"]),
    # not a shaping flag: it is the live catalog minus every exclusion, as
    # resolved above - the same value the manifest records when the build
    # below completes, so the check and the build cannot drift apart.
    requested = dict(corpus_shaping)
    requested["voices"] = {"kokoro": kokoro_voices, "piper": piper_voices}
    if args.skip_corpus:
        args.corpus = "reuse"
    if args.corpus == "rebuild":
        args.skip_corpus = False
        print("Corpus mode: REBUILD (--corpus rebuild - the manifest, if any, is ignored)")
    elif args.corpus == "reuse":
        # check_reuse exits with a diff when the shaping OR the voice set does
        # not match; a missing manifest is an error in EXPLICIT mode, because
        # silently rebuilding is the fallback this flag exists to make
        # impossible.
        corpus_manifest.check_reuse(corpus_dir, requested)
        args.skip_corpus = True
    else:  # auto
        manifest = corpus_manifest.load_manifest(corpus_dir)
        if manifest is not None:
            diffs = corpus_manifest.matches_requested(manifest, requested)
            if not diffs:
                print("Corpus mode: REUSE (manifest matches the requested shaping "
                      "and voice set) - the corpus is a held-fixed variable for "
                      "this run")
                args.skip_corpus = True
            else:
                for line in diffs:
                    print(f"  corpus manifest: {line}")
                print("Corpus mode: REBUILD (the manifest does not match the "
                      "requested shaping or voice set - reusing a differently-"
                      "shaped corpus or one built from a different voice set "
                      "silently would be the measurement this check prevents; "
                      "the cf9c065b reuse, 2026-09-22, did exactly that on the "
                      "voice axis)")
        else:
            print("Corpus mode: REBUILD (no corpus.json manifest - the corpus will "
                  "be built and the manifest written)")

    print("=" * 60)
    print("OpenWakeWord Training")
    print("=" * 60)
    print(f"Wake word: {wake_word}")
    print(f"Samples per voice: {args.samples_per_voice}")
    print(f"Training steps: {args.training_steps}")
    print(f"Layer size: {args.layer_size}")
    if args.seed:
        print(f"Seed: {args.seed}")
    else:
        print("Seed: 0 (unseeded - this run is not reproducible bit for bit, and")
        print("      sweep points should be, so pass --seed for tuning runs)")
    print()

    print("[Compute]")
    report_onnx_providers()

    # NOTE: the voice set (kokoro_voices, piper_voices, pool) was resolved
    # BEFORE the corpus-mode decision above - the reuse check validated
    # against that same computation, and so does the manifest write at the
    # end of this stage: one computation, no drift between check and build.
    # The [Kokoro servers] header printed above is why nothing here probes
    # again.

    # Setup directories
    base_dir = setup_training_dirs(wake_word, args.skip_corpus)
    pos_train = base_dir / "positive_train"
    pos_test = base_dir / "positive_test"
    neg_train = base_dir / "negative_train"
    neg_test = base_dir / "negative_test"

    # THE WHOLE CORPUS STAGE. Skipped wholesale rather than per-call, so a
    # --skip-corpus run cannot half-generate into a corpus it did not build.
    corpus_start = time.time()
    if not args.skip_corpus:
        # Text variations for positive samples.
        #
        # NO UPPERCASE. `wake_word.upper()` was in this list for the first twelve runs
        # and it renders the invented word as SPELLED-OUT LETTERS - "hey S-E-E-R-E-E" -
        # labelled as the wake word. A sixth of the plain positives were mislabelled
        # that whole time. Caught by ear; the measurements that were supposed to catch
        # it both failed, and how they failed is the point:
        #
        #   * duration: 1083 ms against 965 ms, only +12%. Too weak to conclude
        #     anything from; read as "emphatic delivery" instead.
        #   * embedding distance: 0.535 from plain, about a DIFFERENT VOICE (0.70).
        #     Read as "lots of diversity" when it was "this is not the same phrase".
        #
        # A large distance from plain cannot distinguish useful variety from a
        # different utterance. Anything added here must be LISTENED to.
        #
        # It is uppercase on the invented word specifically: "HEY seeree" measures
        # 0.031 from plain (nothing happens), while "hey SEEREE" measures 0.030 from
        # "HEY SEEREE" (both spell it). A real word in caps is fine; the wake word is
        # not a real word, which is the whole reason it makes a good wake word.
        #
        # What is left is punctuation, which changes prosody without touching
        # pronunciation. Distances from plain (af_bella / am_adam):
        #
        #   hey seeree,    0.437 / 0.344
        #   hey seeree!    0.291 / 0.201
        #   hey seeree...  0.264 / 0.523
        #   hey seeree!!   0.158 / 0.232
        #   hey seeree?    0.105 / 0.304
        #   hey seeree.    0.086 / 0.294
        #   Hey Seeree     0.027 / 0.053   <- dropped, indistinguishable from plain
        #
        # `.lower()` is also gone: a literal duplicate of a lowercase wake word.
        #
        # The phrase-alone texts and tuned speed grid live in corpus/positives.py so
        # the microWakeWord corpus renders the same thing.
        positive_texts = plain_positive_texts(wake_word)

        # Negative phrases - see build_negative_phrases for why the confusable ones
        # (near-misses of the wake word) are the important half of this list.
        print("\n[Negative wordlist]")
        negative_phrases = build_negative_phrases(wake_word, args.negatives_file,
                                                 with_commands=args.runon_fraction > 0)
        print(f"  Total negative phrases: {len(negative_phrases)}")

        # === POSITIVE SAMPLES ===
        print("\n" + "=" * 60)
        print("Generating POSITIVE samples...")
        print("=" * 60)

        # Split the positive budget between the phrase alone and the phrase running into
        # a command. The total is unchanged, so the balance against the negatives is too.
        runon_train = int(args.samples_per_voice * args.runon_fraction)
        plain_train = args.samples_per_voice - runon_train
        runon_test = int(args.samples_per_voice // 10 * args.runon_fraction)
        plain_test = args.samples_per_voice // 10 - runon_test

        # piper_voices was resolved before the corpus-mode decision (above),
        # alongside the Kokoro set: the reuse check and the manifest write
        # validate against that same computation, so it is not re-derived
        # here. Piper SUBSTITUTES for part of the phrase-alone budget rather
        # than adding to it.
        #
        # Adding would move three things at once: engine diversity, total corpus size,
        # and - because real clips are a FRACTION of the positive set - real-clip
        # density, which run 10 measured as the largest single lever here (run-on
        # 53% -> 77%). A naive "also generate Piper" over all 84 voices would have
        # taken real clips from ~17% of positives to ~6%, measuring dilution rather
        # than diversity.
        #
        # Substituting holds the total, the plain/run-on split, and real-clip density
        # fixed, leaving one variable: where a share of the phrase-alone clips came
        # from. Run-ons stay entirely Kokoro - see the --piper-fraction help for why.
        kokoro_plain_train, kokoro_plain_test = plain_train, plain_test
        if piper_voices:
            # The Piper half of the voice holdout was already excluded in the
            # pre-check resolution; the budget arithmetic is all that remains
            # at build time.
            kokoro_plain_train = int(round(plain_train * (1 - args.piper_fraction)))
            kokoro_plain_test = int(round(plain_test * (1 - args.piper_fraction)))
            # Budget in TOTAL clips, then spread over however many Piper voices
            # there are - the two engines do not have the same voice count, so a
            # per-voice figure would not substitute one-for-one.
            piper_total_train = (plain_train - kokoro_plain_train) * len(kokoro_voices)
            piper_total_test = (plain_test - kokoro_plain_test) * len(kokoro_voices)
            piper_per_voice_train = max(1, piper_total_train // len(piper_voices))
            piper_per_voice_test = max(1, piper_total_test // len(piper_voices))

        print("\n[Kokoro TTS]")
        print(f"  Per voice: {kokoro_plain_train} phrase-alone, {runon_train} run-on "
              f"({args.runon_fraction:.0%})")
        # workers/batch by KEYWORD: this signature has a `speeds` parameter between
        # desc and workers, and positional args once fell into it (workers-as-speeds
        # died with "'int' object is not iterable" on the first corpus run, 2026-09-09).
        generate_kokoro_samples(pool, kokoro_voices, pos_train,
                                kokoro_plain_train, positive_texts, "Kokoro positive train",
                                workers=args.tts_workers, batch=args.tts_batch)
        generate_kokoro_samples(pool, kokoro_voices, pos_test,
                                kokoro_plain_test, positive_texts, "Kokoro positive test",
                                workers=args.tts_workers, batch=args.tts_batch)

        if piper_voices:
            print(f"\n[Piper TTS]  {len(piper_voices)} voices, "
                  f"{piper_per_voice_train} phrase-alone each "
                  f"(~{args.piper_fraction:.0%} of the phrase-alone budget)")
            generate_piper_samples(args.piper_url, piper_voices, pos_train,
                                   piper_per_voice_train, positive_texts,
                                   PLAIN_SPEED_GRID, "Piper positive train")
            generate_piper_samples(args.piper_url, piper_voices, pos_test,
                                   piper_per_voice_test, positive_texts,
                                   PLAIN_SPEED_GRID, "Piper positive test")

        if runon_train:
            # ONE POOL FOR EVERYTHING. Run-ons briefly had their own server: on
            # Kokoro-FastAPI they were the slow half (229 ms/clip batched on CPU
            # against 88 for plain), and Metal was faster for them (143 ms) while
            # slower for plain. MLX removed the asymmetry - 89 ms single, 63 ms
            # batched - so there is no slow half left to route elsewhere.
            #
            # One reference cache across both sets: the phrase-alone lengths are the
            # same, and rebuilding it would cost a few hundred needless TTS calls.
            reference = {}
            generate_runon_samples(pool, kokoro_voices, pos_train,
                                   runon_train, wake_word, "Kokoro run-on train",
                                   reference, args.tts_workers, args.tts_batch)
            generate_runon_samples(pool, kokoro_voices, pos_test,
                                   runon_test, wake_word, "Kokoro run-on test",
                                   reference, args.tts_workers, args.tts_batch)

        # Before the real clips are copied in, so the shift only ever sees Kokoro
        # output - and before trimming, so the shifted copies are trimmed like the rest.
        if args.child_fraction > 0:
            print("\n[Child-range copies]")
            print(f"  Shifting {args.child_fraction:.0%} of Kokoro clips: "
                  f"female {CHILD_STRETCH['f'][0]}-{CHILD_STRETCH['f'][1]}x, "
                  f"male {CHILD_STRETCH['m'][0]}-{CHILD_STRETCH['m'][1]}x")
            add_child_range_copies(pos_train, "VTLP positive train", args.child_fraction)
            add_child_range_copies(pos_test, "VTLP positive test", args.child_fraction)

        print("\n[Real Voice]")
        # The training half of the recordings. data/recordings/holdout/ is a SIBLING
        # and is never read here - copy_real_samples globs this tree recursively, so
        # a holdout nested inside it would be trained on and every eval number after
        # would measure memorisation. eval/src/paths.py enforces the pair.
        real_samples_dir = WORK_DIR / "data" / "recordings" / "samples"
        real_count = copy_real_samples(real_samples_dir, pos_train, args.real_copies)
        if real_count > 5:
            copy_real_samples(real_samples_dir, pos_test, args.real_copies)

        # === NEGATIVE SAMPLES ===
        print("\n" + "=" * 60)
        print("Generating NEGATIVE samples...")
        print("=" * 60)

        print("\n[Kokoro TTS]")
        generate_kokoro_samples(pool, kokoro_voices, neg_train,
                                args.samples_per_voice, negative_phrases,
                                "Kokoro negative train", workers=args.tts_workers, batch=args.tts_batch)
        generate_kokoro_samples(pool, kokoro_voices, neg_test,
                                args.samples_per_voice // 10, negative_phrases,
                                "Kokoro negative test", workers=args.tts_workers, batch=args.tts_batch)

    # === COUNT SAMPLES ===
    n_pos_train = len(list(pos_train.glob("*.wav")))
    n_pos_test = len(list(pos_test.glob("*.wav")))
    n_neg_train = len(list(neg_train.glob("*.wav")))
    n_neg_test = len(list(neg_test.glob("*.wav")))

    print("\n" + "=" * 60)
    print("Sample counts:")
    print(f"  Positive: {n_pos_train} train, {n_pos_test} test")
    print(f"  Negative: {n_neg_train} train, {n_neg_test} test")
    print("=" * 60)

    # === TRIM SILENCE ===
    # Must run before augmentation: OpenWakeWord places the end of each array at the
    # end of the detection window, so silence on the clip displaces the speech.
    # Negatives are trimmed too - treating both classes identically keeps clip length
    # from becoming a cue the model can learn instead of the phrase itself.
    # `and not args.skip_corpus` because trimming EDITS THE CLIPS IN PLACE. A resumed
    # run would trim already-trimmed audio, and the second pass does not stop at the
    # first pass's boundary - it eats into the speech. Cheap to re-run, not safe to.
    if not args.no_trim and not args.skip_corpus:
        print("\n" + "=" * 60)
        print("Trimming silence (aligns speech with the detection window)...")
        print("=" * 60)
        for directory, label in [(pos_train, "positive train"), (pos_test, "positive test"),
                                 (neg_train, "negative train"), (neg_test, "negative test")]:
            n_trimmed, mean_ms = trim_directory(directory, f"Trim {label}")
            print(f"  {label}: trimmed {n_trimmed} clips (mean {mean_ms:.0f}ms removed)")

    # === FREEZE THE CORPUS ===
    # The manifest is written only on a REBUILD, after trimming (its digest must
    # cover the final tree). It is what --corpus reuse validates against on the
    # next run, and what provenance hashes into the run tag's corpus half.
    if not args.skip_corpus:
        from wordlists import path_for  # recorded, not gated: the training confusables
        # The manifest's shaping is the REQUESTED shaping (corpus_shaping - what
        # matches_requested diffs on reuse, unchanged) plus the holdout list, so a
        # reader can see the exclusion without re-deriving it: the corpus the
        # manifest names is what training consumed, and which voices it does not
        # contain is part of that name. The top-level `voices` field records the
        # SAME post-exclusion set (write_manifest's voices argument, below), and
        # matches_requested now diffs THAT - the axis the cf9c065b reuse,
        # 2026-09-22, was blind on: recorded, never compared, and a
        # post-reservation request matched on every shaping flag because of it.
        manifest_shaping = dict(corpus_shaping)
        manifest_shaping["voice_holdout"] = holdout
        corpus_manifest.write_manifest(
            corpus_dir, wake_word, "oww",
            seed=args.seed,
            shaping=manifest_shaping,
            engines={
                "kokoro": {"url": args.kokoro_url, "version": None},
                **({"piper": {"url": args.piper_url, "version": None}}
                   if piper_voices else {}),
            },
            voices={"kokoro": kokoro_voices, "piper": piper_voices},
            per_voice_counts={"positive_train": n_pos_train,
                              "positive_test": n_pos_test,
                              "negative_train": n_neg_train,
                              "negative_test": n_neg_test},
            wordlist_path=path_for(wake_word),
            wall_time_s=time.time() - corpus_start)

    # Create config and run training. The training-stage hyperparameters that
    # are not also corpus shaping arrive as keywords; --set (args.set) is the
    # last word, applied after everything else inside create_config.
    create_config(wake_word, n_pos_train, args.training_steps, args.layer_size,
                  args.data_dir, args.augmentation_rounds, args.max_negative_weight,
                  seed=args.seed, lr=args.lr,
                  batch_n_per_class=args.batch_n_per_class,
                  target_fp_per_hour=args.target_fp_per_hour,
                  target_accuracy=args.target_accuracy,
                  target_recall=args.target_recall,
                  n_samples_val=args.n_samples_val,
                  augmentation_batch_size=args.augmentation_batch_size,
                  output_dir=smoke_dir,
                  overrides=args.set)

    # === FEATURE CACHE GUARD ===
    # The .npy arrays are keyed on (corpus digest, augmentation_rounds, seed) in a
    # features.json sidecar. Upstream reuses a positive_features_train.npy whenever
    # it exists and knows nothing about why it was built - so changing
    # --augmentation-rounds on a --skip-corpus run was a no-op that looked like a
    # measurement. The sidecar makes the hit a decision: match = reuse, anything
    # else = recompute (and a missing sidecar is a recompute too - pre-manifest
    # caches have no identity to trust). The corpus digest is manifest-aware, so a
    # frozen corpus keys cheaply without re-hashing the tree.
    _short, _mp, corpus_hex, _n, _b = provenance.corpus_tag(wake_word, "oww")
    sidecar_path = corpus_dir / "features.json"
    sidecar = {"corpus": corpus_hex or _short,
               "augmentation_rounds": args.augmentation_rounds,
               "seed": args.seed}
    overwrite = args.rebuild_features
    feature_cache = base_dir / "positive_features_train.npy"
    if feature_cache.exists() and not overwrite:
        old = (json.loads(sidecar_path.read_text())
               if sidecar_path.exists() else None)
        if old and all(old.get(k) == v for k, v in sidecar.items()):
            print(f"Reusing cached feature arrays (features.json matches: "
                  f"corpus {sidecar['corpus'][:7]}, "
                  f"rounds {sidecar['augmentation_rounds']}, "
                  f"seed {sidecar['seed']})")
        else:
            reason = ("no features.json sidecar - a cache with no identity is a "
                      "recompute, not a hit)" if old is None else
                     f"features.json disagrees with this run "
                     f"(corpus/rounds/seed moved: {old})")
            print(f"Cached feature arrays are STALE ({reason}) - recomputing. "
                  f"Pass --rebuild-features to say this out loud.")
            overwrite = True
    # --smoke forces the recompute even on a sidecar match: a cache hit here
    # would make the feature stage - the one this mode exists to run - not run.
    run_augmentation(overwrite=overwrite or args.smoke)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    # Note the existing model before training. setup_training_dirs clears the corpus
    # but NOT the exported model, so a previous run's model survives here - and if
    # training fails, an unchanged file would be reported as this run's output. That
    # happened: a CUDA OOM at 75% killed training, the script still printed
    # "TRAINING COMPLETE!", and the stale model was copied off the box and evaluated
    # twice before the identical checksums gave it away. Now that the two trees are
    # separate the model ALWAYS survives a run, so this check matters more, not less.
    # In smoke mode this is the smoke directory, not the canonical one: the
    # model is written beside where the archive would never look.
    model_path = ((smoke_dir if smoke_dir is not None
                   else WORK_DIR / "output" / safe_name / "oww")
                 / f"{safe_name}.onnx")
    before = model_path.stat().st_mtime if model_path.exists() else None

    returncode, audit = run_training()

    # Whether the model was WRITTEN is the real signal, not the exit code.
    # openwakeword saves the .onnx and then tries to convert it to tflite, which fails
    # on this image and exits 1: its conversion goes through onnx_tf, which is
    # deliberately not installed (abandoned; never worked here - tensorflow-cpu 2.8.1
    # against protobuf >= 3.20), replaced by onnx2tf + onnx2tflite.py. Treating that
    # exit as failure would discard a perfectly good run.
    fresh = model_path.exists() and (before is None
                                     or model_path.stat().st_mtime != before)

    print("\n" + "=" * 60)
    if not fresh:
        print("TRAINING FAILED")
        print("=" * 60)
        if not model_path.exists():
            print(f"{model_path} does not exist.")
        else:
            print(f"{model_path} was not rewritten - it is still the PREVIOUS run's")
            print("model. Do not evaluate or deploy it as though it were this run's.")
        if returncode != 0:
            print(f"\nopenwakeword's train.py exited {returncode}; traceback above.")
        sys.exit(returncode or 1)

    if returncode != 0:
        print(f"NOTE: openwakeword's train.py exited {returncode}, but the model was")
        print("written. Its own tflite conversion cannot run in this image - see")
        print("convert_to_tflite() below, which does it properly straight after.")
        print("=" * 60)

    print("TRAINING COMPLETE!")
    print("=" * 60)

    # === RUN TAG + RESOLVED CONFIG ===
    # The corpus half of the tag reads the manifest written above (cheap, and it
    # names the audio this run consumed); the config half hashes the resolved
    # hyperparameters + seed, so two sweep points at the same commit and corpus
    # get different names and can both be filed - today they share a tag and the
    # second one overwrites the first. The FULL resolved config (paths included)
    # is what the ledger reads, and it is filed under the tag that identifies it.
    # The audit keys are appended to the filed copy AFTER the tag is computed:
    # they are OUTCOMES of the run (which weights actually ran, which
    # checkpoints the merge chose), not inputs, and the merge gate is
    # thresholded on the run's own validation metrics, so they can differ
    # between two runs at an identical config. A tag that moved with them
    # would rename comparable runs and break the ledger's one-tag-one-run
    # contract.
    resolved = yaml.safe_load((WORK_DIR / "training_config.yaml").read_text())
    tag = provenance.run_tag(wake_word, target="oww",
                             config=_tag_input(resolved),
                             fallback=time.strftime("%Y%m%d-%H%M%S"))
    weight_audit, merged_checkpoints, merge_seen = _parse_audit_lines(audit)
    if not weight_audit or not merge_seen:
        print("WARNING: audit lines were missing from the training output (the "
              "clone is probably missing patches/log-weight-and-merge.py); the "
              "filed config records null for them.")
    resolved["effective_max_negative_weight"] = weight_audit or None
    # None = the audit lines never appeared (patch missing) - the WARNING above
    # is why. [] = the audit ran and nothing cleared the gate: the EXPECTED
    # value on this corpus - the gate requires accuracy, recall and fp to hold
    # simultaneously and no checkpoint has ever cleared it (improvement.md
    # P1.4, bug.md B2). A reader finding [] should not chase it as a bug.
    resolved["merged_checkpoints"] = None if not merge_seen else merged_checkpoints
    config_json = model_path.parent / f"{tag}.config.json"
    config_json.write_text(json.dumps(resolved, indent=2, default=str) + "\n")
    if args.smoke:
        # NO .last_run_tag on a smoke run. run-oww-training.sh archives the model
        # by READING that file back, and scripts/sweep.py dies without it - both
        # assume it names the last REAL run. A smoke writing it would put a 200-
        # step model where a deployable one's name goes, or break a sweep that
        # happened to interleave. The smoke lives in its own directory, is
        # cleaned up by hand, and is not a run the archive can be about.
        print(f"  (smoke: no .last_run_tag written - the archive keeps the last "
              f"REAL run)")
    else:
        # The wrapper scripts name the archived model after this same tag; reading
        # it from here (rather than recomputing it in the shell) keeps the two from
        # drifting apart, which would leave the archive and its config named apart.
        (model_path.parent / ".last_run_tag").write_text(tag + "\n")
    print(f"Run tag: {tag}")
    print(f"Resolved config: {config_json}")

    tflite_path = convert_to_tflite(model_path)
    ownership.hand_back(WORK_DIR / "output", work_dir=WORK_DIR)

    size_kb = model_path.stat().st_size / 1024
    print(f"Model: {model_path} ({size_kb:.0f}KB)")
    if tflite_path:
        print(f"       {tflite_path} ({tflite_path.stat().st_size / 1024:.0f}KB)")
    print(f"\nTest with: python test_model.py --model {model_path}")


if __name__ == "__main__":
    main()
