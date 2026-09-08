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
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
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
from train.corpus import kokoro_mlx  # noqa: E402
from train import ownership  # noqa: E402

warnings.filterwarnings("ignore", message="Reached EOF prematurely")

WORK_DIR = REPO_ROOT
os.chdir(WORK_DIR)

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
        # Imported here, not at module scope: it pulls in tensorflow, which costs
        # seconds and is needed by nothing else in this file.
        from train.oww.onnx2tflite import convert
        diff = convert(model_path, tflite_path)
        print(f"  verified against the source ONNX, max diff {diff:.2e}")
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
                  max_negative_weight: int = 2000):
    """Create training configuration."""
    safe_name = wake_word.replace(" ", "_").lower()

    # Load default config from OpenWakeWord
    default_path = WORK_DIR / "openwakeword/examples/custom_model.yml"
    with open(default_path, 'r') as f:
        config = yaml.load(f.read(), yaml.Loader)

    config["target_phrase"] = [safe_name]
    config["model_name"] = safe_name
    config["n_samples"] = n_samples
    config["n_samples_val"] = max(1000, n_samples // 10)
    config["steps"] = training_steps
    config["layer_size"] = layer_size
    config["target_accuracy"] = 0.7
    config["target_recall"] = 0.5
    config["target_false_positives_per_hour"] = 0.1
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
    config["output_dir"] = str(WORK_DIR / "output" / safe_name / "oww")
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

    config_path = WORK_DIR / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Config saved: {config_path}")
    return config


def run_augmentation():
    """Run OpenWakeWord augmentation pipeline."""
    print("\n" + "=" * 60)
    print("Running augmentation pipeline...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--augment_clips"
    ], check=True)


def wait_for_kokoro_shutdown(timeout: int = 120):
    """Block until the Kokoro servers have released their VRAM.
    """
    import socket

    import torch
    if not torch.cuda.is_available():
        print("  no CUDA - not waiting for Kokoro (nothing holds VRAM here)")
        return

    servers = [("kokoro", 8880), ("kokoro2", 8881)]
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


def run_training():
    """Run OpenWakeWord model training."""
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)

    # Both BEFORE the subprocess: it allocates the feature array on its first
    # breath, so anything reported afterwards describes a decision already made.
    wait_for_kokoro_shutdown()
    report_free_vram()

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    result = subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--train_model"
    ])
    return result.returncode


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
    parser.add_argument("--layer-size", type=int, default=64, choices=[32, 64, 128], help="Network layer size")
    parser.add_argument("--kokoro-url", default=os.environ.get("KOKORO_URL", "http://localhost:8880"),
                        help="Kokoro TTS URL. Comma-separate several to split the work "
                             "across them: one Kokoro process is single-threaded and "
                             "saturates one core, so more PROCESSES scale where more "
                             "client threads do not. (Not on Metal, where two "
                             "instances share one GPU and measured no faster.)\n"
                             "mlx:// renders in-process instead - see "
                             "tts-service/README.md.")
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
                             "TTS, real-clip copying and trimming; still re-runs "
                             "augmentation and training, so --training-steps and the "
                             "model geometry are still yours to change. Sample-shaping "
                             "flags (--samples-per-voice, --piper-fraction, "
                             "--child-fraction, --runon-fraction) are IGNORED: the clips "
                             "already exist and this does not rebuild them.")
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
                             "threshold, for free. Measured in tuning run 8. Recommend to leave at default.")
    parser.add_argument("--augmentation-rounds", type=int, default=3,
                        help="How many differently-augmented copies of each clip to "
                             "compute features for (default: %(default)s). Multiplies "
                             "training data at no TTS cost.")
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
                        default=os.environ.get("PIPER_URL", "piper:10200"),
                        help="Wyoming TTS host:port for Piper (default: %(default)s)")
    parser.add_argument("--piper-speakers", type=int, default=12,
                        help="Speakers to sample per multi-speaker Piper voice, "
                             "evenly spaced (default: %(default)s). "
                             "en_US-libritts_r-medium alone carries 904, which "
                             "would swamp the corpus with one model's g2p.")
    parser.add_argument("--piper-languages", default="en_US,en_GB",
                        help="Language prefixes to include (default: %(default)s)")
    args = parser.parse_args()

    wake_word = args.wake_word
    safe_name = wake_word.replace(" ", "_").lower()

    print("=" * 60)
    print("OpenWakeWord Training")
    print("=" * 60)
    print(f"Wake word: {wake_word}")
    print(f"Samples per voice: {args.samples_per_voice}")
    print(f"Training steps: {args.training_steps}")
    print(f"Layer size: {args.layer_size}")
    print()

    print("[Compute]")
    report_onnx_providers()

    # ONLY WHEN GENERATING. --skip-corpus needs no TTS, and probing here would fail
    # a resumed run for want of a server it never calls - while also holding
    # ~8 GiB of VRAM that training is about to want. See wait_for_kokoro_shutdown.
    if not args.skip_corpus:
        # Get Kokoro voices
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

    # Setup directories
    base_dir = setup_training_dirs(wake_word, args.skip_corpus)
    pos_train = base_dir / "positive_train"
    pos_test = base_dir / "positive_test"
    neg_train = base_dir / "negative_train"
    neg_test = base_dir / "negative_test"

    # THE WHOLE CORPUS STAGE. Skipped wholesale rather than per-call, so a
    # --skip-corpus run cannot half-generate into a corpus it did not build.
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

        # Piper SUBSTITUTES for part of the phrase-alone budget rather than adding to it.
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
        piper_voices = []
        kokoro_plain_train, kokoro_plain_test = plain_train, plain_test
        if args.piper_fraction > 0:
            host, _, port = args.piper_url.rpartition(":")
            piper_voices = select_piper_voices(
                host, port, wake_word,
                languages=tuple(args.piper_languages.split(",")),
                max_speakers=args.piper_speakers)
            if piper_voices:
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
        generate_kokoro_samples(pool, kokoro_voices, pos_train,
                                kokoro_plain_train, positive_texts, "Kokoro positive train",
                                args.tts_workers, args.tts_batch)
        generate_kokoro_samples(pool, kokoro_voices, pos_test,
                                kokoro_plain_test, positive_texts, "Kokoro positive test",
                                args.tts_workers, args.tts_batch)

        if piper_voices:
            host, _, port = args.piper_url.rpartition(":")
            print(f"\n[Piper TTS]  {len(piper_voices)} voices, "
                  f"{piper_per_voice_train} phrase-alone each "
                  f"(~{args.piper_fraction:.0%} of the phrase-alone budget)")
            generate_piper_samples(host, int(port), piper_voices, pos_train,
                                   piper_per_voice_train, positive_texts,
                                   PLAIN_SPEED_GRID, "Piper positive train")
            generate_piper_samples(host, int(port), piper_voices, pos_test,
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
                                "Kokoro negative train", args.tts_workers, args.tts_batch)
        generate_kokoro_samples(pool, kokoro_voices, neg_test,
                                args.samples_per_voice // 10, negative_phrases,
                                "Kokoro negative test", args.tts_workers, args.tts_batch)

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

    # Create config and run training
    create_config(wake_word, n_pos_train, args.training_steps, args.layer_size,
                  args.data_dir, args.augmentation_rounds, args.max_negative_weight)
    run_augmentation()

    # Note the existing model before training. setup_training_dirs clears the corpus
    # but NOT the exported model, so a previous run's model survives here - and if
    # training fails, an unchanged file would be reported as this run's output. That
    # happened: a CUDA OOM at 75% killed training, the script still printed
    # "TRAINING COMPLETE!", and the stale model was copied off the box and evaluated
    # twice before the identical checksums gave it away. Now that the two trees are
    # separate the model ALWAYS survives a run, so this check matters more, not less.
    model_path = WORK_DIR / "output" / safe_name / "oww" / f"{safe_name}.onnx"
    before = model_path.stat().st_mtime if model_path.exists() else None

    returncode = run_training()

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

    tflite_path = convert_to_tflite(model_path)
    ownership.hand_back(WORK_DIR / "output", work_dir=WORK_DIR)

    size_kb = model_path.stat().st_size / 1024
    print(f"Model: {model_path} ({size_kb:.0f}KB)")
    if tflite_path:
        print(f"       {tflite_path} ({tflite_path.stat().st_size / 1024:.0f}KB)")
    print(f"\nTest with: python test_model.py --model {model_path}")


if __name__ == "__main__":
    main()
