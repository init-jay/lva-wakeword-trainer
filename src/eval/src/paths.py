#!/usr/bin/env python3
"""Where the evaluation tools look for held-out recordings, corpora and models.

One module rather than a default string per tool, because the samples/holdout
split is a SAFETY PROPERTY and not a naming convention. `train/corpus/real.py`
globs the samples tree RECURSIVELY for positives, so a holdout nested anywhere
inside it is trained on, and every number this harness reports silently becomes
training accuracy - it overstated detection by ~10 points during the tuning
written up in the tuning log, and nothing in the output looked wrong. `holdout/` is
therefore a SIBLING of `samples/`, never a child, and `warn_if_trained_on` says
so out loud when a run is pointed back inside the training set anyway.

    data/recordings/samples/<speaker>/          trained on
    data/recordings/holdout/<speaker>/          evaluated against, never trained on
    data/recordings/holdout/<speaker>_runon/    ditto, phrase running into a command

THE `_runon` SUFFIX IS LOAD-BEARING. Plain and run-on clips answer different
questions and are never pooled: `compare_models.py` reports them as separate rows,
and `eval_model.py` builds its own command-following case by concatenating a
command onto a plain clip, so a real run-on recording in its positives set would
be scored as if it were the phrase alone. Since the loaders recurse, splitting on
the directory suffix is what keeps a bare `--positives data/recordings/holdout`
from quietly mixing the two.

PATHS ARE ANCHORED ON THE REPO ROOT, not the working directory. These tools run
in the eval image with the repo mounted, and `python -m eval.x` from the root is
the documented invocation - so cwd-relative defaults have worked, but only by
coincidence of where the caller happened to be standing.
"""

from pathlib import Path

# The repo root is the nearest ancestor that has BOTH data/recordings/ and
# src/wordlists - not a fixed depth above this file. The image mounts this tree
# at /app/eval, one level shallower than the checkout's src/eval/src/, and a
# fixed offset would be right in exactly one of the two places. (wordlists/ moved
# under src/ in the restructure; data/recordings/ stayed at the git root, so the
# pair now straddles the boundary and still singles the root out.)
def _repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "recordings").is_dir() and (parent / "src" / "wordlists").is_dir():
            return parent
    # A fresh clone has no data/ at all: it is gitignored and produced by the record
    # and corpus steps. The root is still unambiguous from TRACKED markers, and
    # refusing to resolve it here would make this module - and so every tool and test
    # that imports it - unrunnable until somebody has been to a microphone. Same walk,
    # weaker anchor, and the strict one above still wins wherever both exist.
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "wordlists").is_dir() and (parent / "Makefile").is_file():
            return parent
    raise RuntimeError(
        f"no repo root above {__file__}: looked for an ancestor holding BOTH "
        "data/recordings/ and src/wordlists/, then for one holding the tracked pair "
        "(src/wordlists/ and Makefile). A fresh clone has no data/ at all - it is "
        "gitignored and produced by the record and corpus steps - so if the second "
        "walk also failed, this file is not inside a checkout of the repo.")

REPO_ROOT = _repo_root()

RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
SAMPLES_DIR = RECORDINGS_DIR / "samples"
HOLDOUT_DIR = RECORDINGS_DIR / "holdout"

# Trained models and their scorecards, one directory per wake word and per trainer:
#
#     output/<wake_word>/oww/<wake_word>_<commit>.onnx     openWakeWord, the ship candidate
#     output/<wake_word>/oww/<wake_word>_<commit>.tflite   its conversion
#     output/<wake_word>/mww/<wake_word>_<commit>.tflite   microWakeWord, for the ESP32
#     output/<wake_word>/mww/<wake_word>_<commit>.json     its ESPHome manifest
#     output/<wake_word>/mww/tflite_streaming_roc_<commit>.txt
#
# The commit tag is what a run is referred to, and it is why the manifest and the
# .tflite it names have to be moved as a pair - `"model"` in the JSON is a bare
# sibling filename, so a manifest separated from its model is a broken model.
#
# Both trainers write here now: train/oww/train.py exports to output/<w>/oww/, and
# train/mww/ trains into output/<w>/mww/<run-tag>/, one directory per run because
# microWakeWord refuses to train into an existing one. The commit-tagged files listed
# above are the collected form - run-oww-training.sh produces them for the .onnx.
#
# The generated corpora went the other way, to data/corpus/. That split is what keeps
# train.py's per-run rmtree away from anything in this tree.
OUTPUT_DIR = REPO_ROOT / "output"

# The TTS evaluation corpora, which are generated rather than recorded - hence
# data/corpus/ beside the trainer's, not data/recordings/.
EVAL_CORPUS_DIR = REPO_ROOT / "data" / "corpus" / "eval"
NEGATIVES_DIR = EVAL_CORPUS_DIR / "negatives_tts"
POSITIVES_DIR = EVAL_CORPUS_DIR / "positives_tts"

RUNON_SUFFIX = "_runon"


def holdout_dirs(runon=False, root=HOLDOUT_DIR):
    """Held-out speaker directories, split on the `_runon` suffix.

    Returns [] rather than raising when nothing is there, so a tool can report
    "no held-out positives" in its own words instead of dying in argparse.

    A holdout with no speaker subdirectories at all is one unnamed speaker, and
    the root itself is the plain set - the layout a single-speaker `--holdout`
    recording session produces before anyone passes `--speaker`.
    """
    if not root.is_dir():
        return []
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        return [] if runon else [root]
    return [d for d in subdirs if d.name.endswith(RUNON_SUFFIX) == runon]


def speaker_label(directory):
    """Short name for a speaker directory, for keying a per-speaker report on.

    Relative to the recordings tree when it sits inside one, so
    data/recordings/holdout/speaker1 reads as `speaker1` rather than a path - and a
    directory somewhere else entirely still gets its own basename rather than
    colliding with everything.
    """
    path = Path(directory).resolve()
    for base in (HOLDOUT_DIR, SAMPLES_DIR, RECORDINGS_DIR):
        try:
            return str(path.relative_to(base.resolve()))
        except (ValueError, OSError):
            continue
    return path.name


def warn_if_trained_on(directories):
    """Shout when an evaluation is pointed at clips the trainer also reads.

    A warning and not a hard error: measuring training accuracy on purpose is a
    legitimate thing to do - it is the baseline the held-out number is compared
    against - and only doing it BY ACCIDENT is the failure. Returns the offending
    directories so a caller can decide differently.
    """
    inside = []
    samples = SAMPLES_DIR.resolve()
    for directory in directories:
        try:
            Path(directory).resolve().relative_to(samples)
        except (ValueError, OSError):
            continue
        inside.append(str(directory))
    if inside:
        print("WARNING: these are TRAINING clips, so this reports training "
              "accuracy, not detection:")
        for directory in inside:
            print(f"    {directory}")
        print(f"         Held-out recordings live in {HOLDOUT_DIR}, outside the "
              f"tree the trainer globs.")
    return inside


def describe(directories):
    """`a, b, c` with the repo root stripped, for headers that name their inputs."""
    out = []
    for directory in directories:
        path = Path(directory)
        try:
            out.append(str(path.resolve().relative_to(REPO_ROOT)))
        except (ValueError, OSError):
            out.append(str(path))
    return ", ".join(out) if out else "(none)"
