#!/usr/bin/env python3
"""Where the evaluation tools look for held-out recordings, corpora and models.

One module rather than a default string per tool, because the samples/holdout
split is a SAFETY PROPERTY and not a naming convention. `src/train/corpus/real.py`
globs the samples tree RECURSIVELY for positives, so a holdout nested anywhere
inside it is trained on, and every number this harness reports silently becomes
training accuracy - it overstated detection by ~10 points during the tuning
written up in the tuning log, and nothing in the output looked wrong. `holdout/` is
therefore a SIBLING of `samples/`, never a child, and `warn_if_trained_on` says
so out loud when a run is pointed back inside the training set anyway.

    data/recordings/<wake_word>/samples/<speaker>/          trained on
    data/recordings/<wake_word>/holdout/<speaker>/          evaluated against, never trained on
    data/recordings/<wake_word>/holdout/<speaker>_runon/    ditto, phrase running into a command

Every child of data/recordings/ and of data/corpus/ is a wake word (the slug is
the same one recipes/, corpus/ and output/ use: spaces to underscores, lower).
word-agnostic inputs live elsewhere: data/external/, data/piper_voices/.

THE `_runon` SUFFIX IS LOAD-BEARING. Plain and run-on clips answer different
questions and are never pooled: `compare_models.py` reports them as separate rows,
and `eval_model.py` builds its own command-following case by concatenating a
command onto a plain clip, so a real run-on recording in its positives set would
be scored as if it were the phrase alone. Since the loaders recurse, splitting on
the directory suffix is what keeps a bare `--positives data/recordings/<wake_word>/holdout`
from quietly mixing the two.

PATHS ARE ANCHORED ON THE REPO ROOT, not the working directory. These tools run
in the eval image with the repo mounted, and `python -m eval.x` from the root is
the documented invocation - so cwd-relative defaults have worked, but only by
coincidence of where the caller happened to be standing.
"""

from pathlib import Path

# The repo root is the nearest ancestor that has BOTH data/recordings/ and
# src/recipe - not a fixed depth above this file. The image mounts this tree
# at /app/eval, one level shallower than the checkout's src/eval/src/, and a
# fixed offset would be right in exactly one of the two places. (recipe/ moved
# under src/ in the restructure; data/recordings/ stayed at the git root, so the
# pair now straddles the boundary and still singles the root out.)
def _repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "recordings").is_dir() and (parent / "src" / "recipe").is_dir():
            return parent
    # A fresh clone has no data/ at all: it is gitignored and produced by the record
    # and corpus steps. The root is still unambiguous from TRACKED markers, and
    # refusing to resolve it here would make this module - and so every tool and test
    # that imports it - unrunnable until somebody has been to a microphone. Same walk,
    # weaker anchor, and the strict one above still wins wherever both exist.
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "recipe").is_dir() and (parent / "Makefile").is_file():
            return parent
    raise RuntimeError(
        f"no repo root above {__file__}: looked for an ancestor holding BOTH "
        "data/recordings/ and src/recipe/, then for one holding the tracked pair "
        "(src/recipe/ and Makefile). A fresh clone has no data/ at all - it is "
        "gitignored and produced by the record and corpus steps - so if the second "
        "walk also failed, this file is not inside a checkout of the repo.")

REPO_ROOT = _repo_root()

RECORDINGS_ROOT = REPO_ROOT / "data" / "recordings"
CORPUS_ROOT = REPO_ROOT / "data" / "corpus"


def word_name(wake_word):
    """The directory slug a wake word uses under recipes/, corpus/ and output/:
    spaces to underscores, lowercased - the same rule src/recipe.path_for and
    src/train/provenance.py apply. One rule for the whole data/ tree."""
    return wake_word.replace(" ", "_").lower()


def recordings_dir(wake_word, root=None):
    """data/recordings/<wake_word>/ - every child of data/recordings/ is a wake
    word; `root` is overridable so tests can point at a tree of their own."""
    return (RECORDINGS_ROOT if root is None else Path(root)) / word_name(wake_word)


def samples_dir(wake_word, root=None):
    return recordings_dir(wake_word, root) / "samples"


def holdout_dir(wake_word, root=None):
    return recordings_dir(wake_word, root) / "holdout"


def eval_corpus_dir(wake_word, root=None):
    """The generated TTS evaluation corpora for one word - beside the trainer's
    data/corpus/<wake_word>/{oww,mww} trees, never inside them (the trainer
    rmtree's those)."""
    root = CORPUS_ROOT if root is None else Path(root)
    return root / word_name(wake_word) / "eval"


NEGATIVES_SUBDIR = "negatives_tts"
POSITIVES_SUBDIR = "positives_tts"


def negatives_dir(wake_word):
    return eval_corpus_dir(wake_word) / NEGATIVES_SUBDIR


def positives_dir(wake_word):
    return eval_corpus_dir(wake_word) / POSITIVES_SUBDIR


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
# Both trainers write here now: src/train/oww/train.py exports to output/<w>/oww/, and
# src/train/mww/ trains into output/<w>/mww/<run-tag>/, one directory per run because
# microWakeWord refuses to train into an existing one. The commit-tagged files listed
# above are the collected form - run-oww-training.sh produces them for the .onnx.
#
# The generated corpora went the other way, to data/corpus/. That split is what keeps
# train.py's per-run rmtree away from anything in this tree.
OUTPUT_DIR = REPO_ROOT / "output"

RUNON_SUFFIX = "_runon"


def holdout_dirs(runon=False, wake_word=None, root=None):
    """Held-out speaker directories for one word, split on the `_runon` suffix.

    Returns [] rather than raising when nothing is there, so a tool can report
    "no held-out positives" in its own words instead of dying in argparse.

    A holdout with no speaker subdirectories at all is one unnamed speaker, and
    the root itself is the plain set - the layout a single-speaker `--holdout`
    recording session produces before anyone passes `--speaker`.
    """
    if root is None:
        if wake_word is None:
            raise ValueError("holdout_dirs needs wake_word (or an explicit root)")
        root = holdout_dir(wake_word)
    if not root.is_dir():
        return []
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        return [] if runon else [root]
    return [d for d in subdirs if d.name.endswith(RUNON_SUFFIX) == runon]


def speaker_label(directory, wake_word=None):
    """Short name for a speaker directory, for keying a per-speaker report on.

    Relative to the word's recordings tree when it sits inside one, so
    data/recordings/<word>/holdout/speaker1 reads as `holdout/speaker1` rather than
    a path - and a directory somewhere else entirely still gets its own basename
    rather than colliding with everything.
    """
    path = Path(directory).resolve()
    bases = []
    if wake_word is not None:
        bases = [holdout_dir(wake_word), samples_dir(wake_word), recordings_dir(wake_word)]
    for base in bases:
        try:
            return str(path.relative_to(base.resolve()))
        except (ValueError, OSError):
            continue
    return path.name


def warn_if_trained_on(directories, wake_word, root=None):
    """Shout when an evaluation is pointed at clips the trainer also reads.

    A warning and not a hard error: measuring training accuracy on purpose is a
    legitimate thing to do - it is the baseline the held-out number is compared
    against - and only doing it BY ACCIDENT is the failure. Returns the offending
    directories so a caller can decide differently.
    """
    inside = []
    samples = samples_dir(wake_word, root).resolve()
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
        print(f"         Held-out recordings live in {holdout_dir(wake_word)}, "
              f"outside the tree the trainer globs.")
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
