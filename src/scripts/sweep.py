#!/usr/bin/env python3
"""A sweep runner for the tuning loop: train -> eval -> (not better) -> retrain.

    src/train/train-mww-applesilicon/.venv/bin/python src/scripts/sweep.py sweep.yaml [--dry-run]
    (the venv is now under src/train/: src/train/train-mww-applesilicon/.venv)

improvement.md P0.5. The loop this repo actually runs had no machine-readable
edge: the results of "try a setting, is it better?" lived in 40+
logs/training-*.log files and in scrollback, so the answer to "what did we
already try, and what did it do" had to be reconstructed by re-reading them.
This script runs a YAML-described grid of trainer options against ONE frozen
corpus, each point at a deterministic sequence of seeds, and appends one
record per completed run to output/<wake_word>/runs.jsonl (src/train/ledger.py),
so the loop ends in a number a program can read and the next session starts
from what was actually measured instead of from memory.

FROZEN CORPUS, ONE RULE. During a sweep the corpus is a HELD-FIXED
INDEPENDENT VARIABLE (improvement.md P0.3). The first grid point builds it
(mww: train.mww.corpus then train.mww.features; oww: the first train run
builds it in-place); every later point only VERIFIES it - mww corpus
--skip, which checks the corpus.json manifest against the shaping this
invocation would use and exits with a diff on any mismatch; oww
--corpus reuse, the same check run inside the trainer. If no manifest
exists after point 1 this script fails loudly: a corpus with no identity
cannot be named in a run tag, and a sweep whose points trained on different
audio is a comparison that answers nothing. This is also why a grid key
that would reshape the corpus is refused up front: the point-1 shape is
frozen into the manifest, so varying it would make every later point's
reuse check fail - and an unrefused version of it would quietly mean a
different corpus at different points.

REPEATS >= 2, AND THE SPREAD IS THE POINT. Two runs of an IDENTICAL
configuration have measured 77% and 67% on the same held-out clips (10
points of seed/render noise in this repo), so a single repeat of a setting
cannot separate the effect of the setting from that noise: with N < 2 there
is no spread to compare the effect against, which is exactly the
"coin-flip dressed up as a ranking" the eval gates exist to refuse. N < 2
is therefore refused, and the ledger prints min/max beside the mean per
configuration so a difference inside the repeat-to-repeat band reads as
what it is.

RESUMABLE. A point whose (target, tag) is already in the ledger is skipped
and the skip is printed. The mww tag is knowable before training (computed
with --print-tag, the same way run-mww-training-applesilicon.sh does it);
the oww tag is only knowable after the run (.last_run_tag), so an oww point
already in the ledger is recognised after training, and the record is
simply not appended again (ledger.record would refuse it anyway - the
ledger is history, not a cache).

THE ONE THING THIS SCRIPT MUST NEVER DO: rewrite the ledger. Appending a
line is the only write it makes to runs.jsonl; editing, compacting or
deleting a "not better" result would make the file a cache with a memory of
its own, and a history you can edit is not a history.

PREREQUISITES: this is a HOST-SIDE loop for the Apple Silicon route and it
starts NOTHING. The TTS engines the corpus stage speaks to (Piper on 8898,
Kokoro on 8900 - src/tts-service/README.md) must already be running when the
corpus is built, and no Docker is involved anywhere. The trainer
subprocesses default to the Apple Silicon venvs (train-mww-applesilicon /
train-applesilicon; override with `python:`). The eval subprocess defaults
to THIS interpreter, because the eval stack (onnxruntime, TFLite,
pymicro-features) lives in its own environment - on the Mac that is the
Docker `eval` image, whose compose run cannot be invoked from this script
without starting Docker, which it does not; set `eval.python` to an
interpreter that imports the eval stack, or run this script with one.

PyYAML: both trainer venvs carry it (6.0.3, verified 2026-09-10); the
system python3 (3.14.6) does not - run this script with a venv python.

YAML SHAPE

    wake_word: "hey seeree"
    target: mww                # or oww
    python: <path>             # optional; the trainer venv python
    base_seed: 42              # optional; seeds are base + 1000*point + repeat
    max_faph: 0.2              # optional; mww manifest budget (run-script default)
    base:                      # trainer CLI options, applied to every point
      training-steps: 10000
    grid:                      # Cartesian product; keys are trainer CLI options
      piper-fraction: [0.0, 0.3]   # (oww) / batch-size: [8, 16] (mww)
    repeats: 2
    corpus:                    # mww only; the corpus-stage flags. Must mirror
      piper-url: tcp://127.0.0.1:8898   the flags the frozen corpus was built
      piper-speakers: 12           with, or the --skip check prints the diff.
    ambient: []                # optional; defaults to data/external/mww_ambient/*
    eval:
      enabled: true
      python: <path>           # optional; the interpreter with the eval stack
      compare_against: <tag>   # optional; a previously filed run tag
      voice-holdout-set: <path>  # optional; the synthetic voice-holdout set
                                # (src/wordlists/voice_holdout.yaml); passed to eval
                                # as --voice-holdout-set so the filed eval block
                                # carries the arm-ranking number
"""

import argparse
import itertools
import json
import subprocess
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The import root src/: `train`, `wordlists` and the eval modules import by
# their pre-reorg names, and the trainer venvs do not install them, so every
# `-m train.*` / eval subprocess below gets PYTHONPATH pointing here.
sys.path.insert(0, str(REPO_ROOT / "src"))
PYTHONPATH = str(REPO_ROOT / "src")

try:
    import yaml
except ModuleNotFoundError:
    sys.exit("PyYAML is required: run this script with one of the trainer venvs\n"
             "  (both carry it: train-mww-applesilicon/.venv, train-applesilicon/.venv).\n"
             "  The system python3 does not.")

from train import ledger  # noqa: E402
from train.corpus import manifest as corpus_manifest  # noqa: E402

TARGETS = ("mww", "oww")

# The Apple Silicon host venvs - the route this script exists for. Both carry
# PyYAML (module docstring); `python:` in the YAML overrides.
TRAINER_PYTHON = {
    "mww": REPO_ROOT / "src" / "train" / "train-mww-applesilicon" / ".venv" / "bin" / "python",
    "oww": REPO_ROOT / "src" / "train" / "train-applesilicon" / ".venv" / "bin" / "python",
}

# Keys that SHAPE THE CORPUS, per trainer. A grid key on one of these is
# refused (module docstring): src/train/mww/corpus.py and src/train/oww/train.py list
# the shaping flags the reuse check compares against, and these are those
# flags in CLI spelling.
MWW_CORPUS_KEYS = {"samples-per-voice", "negatives-per-voice", "kokoro-fraction",
                   "child-fraction", "real-copies", "real-vtlp",
                   "balance-real-copies", "balance-max-multiplier",
                   "piper-speakers",
                   "piper-languages", "negatives-file", "no-trim"}
OWW_CORPUS_KEYS = {"samples-per-voice", "runon-fraction", "child-fraction",
                   "piper-fraction", "real-copies", "real-copies-override",
                   "real-vtlp", "balance-real-copies", "balance-max-multiplier",
                   "piper-speakers", "piper-languages", "negatives-file",
                   "exclude-voices", "include-legacy-voices", "no-trim"}

MWW_QUANT_DIR = "tflite_stream_state_internal_quant"
MWW_MODEL_FILE = "stream_state_internal_quant.tflite"


def die(message, code=1):
    print(f"\nSWEEP FAILED: {message}", file=sys.stderr)
    sys.exit(code)


def build_jobs(grid, grid_keys, repeats, base_seed):
    """The point x repeat expansion: (point_index, grid_point, repeat, seed).
    Seeds are base + 1000*point + repeat - the 1000 stride keeps a repeat of
    one point from colliding with the seed of the next point. Named so
    tests/test_sweep.py can build the job list and assert what each job
    runs; the dry run and the real loop both call it."""
    points = [dict(zip(grid_keys, combo)) for combo in
              itertools.product(*(grid[k] for k in grid_keys))] or [{}]
    return [(pi, gp, r, base_seed + 1000 * pi + r)
            for pi, gp in enumerate(points) for r in range(repeats)]


def is_first_job(pi, repeat):
    """Whether (point, repeat) is the first JOB of the sweep: point 0, repeat 0.

    Distinct from the first POINT (pi == 0): a repeat of point 0 is NOT the
    first job, and must not run in auto corpus mode - auto reuses a matching
    manifest but SILENTLY REBUILDS a mismatched one (src/train/oww/train.py's
    --corpus auto), so a TTS catalog change between two repeats of point 0
    would redraw the frozen corpus mid-sweep with no error, and every later
    point would then verify against the NEW manifest and pass (bug.md C5,
    2026-09-22 - the mirror of the cf9c065b silent reuse). Only the first
    job may build; every later job verifies.
    """
    return pi == 0 and repeat == 0


def job_corpus_reuse(first_job, repeat, corpus_axes):
    """Whether an oww job runs with --corpus reuse (True) or auto (False).

    Default is C5's rule: only the first job is auto, every later job
    reuses-and-refuses. The exception is a sweep whose GRID varies a corpus
    axis, declared explicitly in `corpus_axes:` (e.g. real-vtlp): then
    the corpus is frozen per POINT, not per sweep - each point's repeat 0
    runs auto (verify, or rebuild because the grid itself changed the
    shaping), and repeats inside a point still refuse a mismatch. The
    rebuild is the experiment's intent, not drift; the difference is that
    the YAML SAYS SO. Cross-point rows therefore carry different corpus ids
    in the ledger, and TTS noise rides between arms - the arms' own repeats
    are what separate the lever from the redraw.
    """
    return not (first_job or (repeat == 0 and corpus_axes))


def options_to_args(options):
    """{key: value} -> CLI arguments: True is a bare flag, False/None dropped,
    everything else `--key value` (the trainers' argparse spelling)."""
    out = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(f"--{key}")
        else:
            out.extend([f"--{key}", str(value)])
    return out


def ambient_dirs(override):
    """mww ambient sets: data/external/mww_ambient/* auto-discovered, sorted -
    the same discovery run-mww-training-applesilicon.sh does (it passes every
    subdirectory present; mWW treats them as separate ambient sets). A YAML
    `ambient:` list wins, verbatim."""
    if override:
        return [str(p) for p in override]
    root = REPO_ROOT / "data" / "external" / "mww_ambient"
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.iterdir() if p.is_dir())


def run_stage(name, cmd, stage_times):
    """Run one subprocess, streaming its output to the terminal, and time it.

    The stage wall times land in the ledger record: for a sweep, how long a
    setting took is part of what it did (the mww host run was measured at
    14m14s against 26m06s in the container, and the corpus is most of it).
    """
    print(f"\n=== {datetime.now():%H:%M:%S}  {name}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    start = time.monotonic()
    # The import root src/ on the child's PYTHONPATH: the trainer venvs do not
    # install the repo modules, so `train.*` resolves only through it.
    result = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONPATH": PYTHONPATH})
    stage_times[name] = round(time.monotonic() - start, 1)
    return result


def load_spec(path):
    """The sweep YAML. Unknown top-level keys are a typo waiting to be
    silently ignored, which in a tuning loop means a comparison run that
    believes it varied a knob it never varied - refused here, now.

    `corpus_axes:` - optional; grid keys that change the CORPUS identity
    (oww only, e.g. real-vtlp). The corpus is then frozen PER POINT: each
    arm's repeat 0 runs --corpus auto (verify, or rebuild because the grid
    itself changed the shaping - the rebuild is the experiment's intent,
    not drift), and repeats inside a point still refuse a mismatch
    (job_corpus_reuse). Cross-arm rows therefore carry different corpus ids
    and TTS redraw noise; the within-arm repeats separate the lever from
    the redraw."""
    spec = yaml.safe_load(Path(path).read_text())
    unknown = set(spec) - {"wake_word", "target", "python", "eval", "corpus",
                           "base_seed", "max_faph", "ambient", "base", "grid",
                           "repeats", "corpus_axes"}
    if unknown:
        die(f"unknown keys in {path}: {sorted(unknown)}")
    if not spec.get("wake_word"):
        die("wake_word is required")
    spec["target"] = spec.get("target", "mww")
    if spec["target"] not in TARGETS:
        die(f"target must be one of {TARGETS}, got {spec['target']!r}")
    spec["repeats"] = int(spec.get("repeats", 2))
    if spec["repeats"] < 2:
        # The refusal the module docstring carries: with one repeat there is
        # no spread to compare the effect against, and this repo's measured
        # repeat-to-repeat noise at an identical config is 10 points.
        die("repeats must be >= 2. With a single repeat a measured difference "
            f"cannot be separated from seed noise - this repo has measured 10 points "
            f"between two runs of an IDENTICAL configuration - and the ledger's whole "
            f"job is to print that spread beside the mean.")
    spec["base_seed"] = int(spec.get("base_seed", 42))
    if not spec["base_seed"]:
        die("base_seed must be nonzero: both trainers read --seed 0 as 'unseeded', "
            "and an unseeded sweep point is a point you cannot re-run.")
    if "seed" in (spec.get("base") or {}):
        die("the sweep owns --seed (deterministic per point, module docstring); "
            "remove it from `base`")
    grid = spec.get("grid") or {}
    corpus_keys = MWW_CORPUS_KEYS if spec["target"] == "mww" else OWW_CORPUS_KEYS
    axes = spec.get("corpus_axes") or []
    if axes:
        if spec["target"] == "mww":
            die("corpus_axes is oww-only: the mww corpus stage is owned by "
                "the sweep runner (build-once, --skip-verify), not the trainer")
        for key in axes:
            if key not in grid:
                die(f"corpus_axes names {key!r}, but the grid does not vary it")
            if key not in corpus_keys:
                die(f"corpus_axes names {key!r}, which is not a corpus-shaping "
                    f"flag - a rebuild would change nothing and the point-"
                    f"frozen mode would only add TTS noise")
    for key in grid:
        if key == "seed":
            die("the sweep owns --seed (deterministic per point); it cannot be a grid key")
        if key in corpus_keys and key not in axes:
            die(f"{key!r} shapes the CORPUS, which is frozen once for the whole "
                f"sweep: it cannot be a grid key. Put it in `base` (it then applies "
                f"identically to every point) or in `corpus`; or, if the point is "
                f"the experiment, declare it under `corpus_axes:` - the corpus is "
                f"then frozen PER POINT and each arm rebuilds it.")
    if "tag" in (spec.get("base") or {}):
        die("the sweep names its own runs by run tag; remove --tag from `base`")
    eval_section = spec.get("eval")
    if eval_section is not None and not isinstance(eval_section, bool):
        # Same typo refusal as the top level, one level down: an unknown key
        # here means a flag the operator believes they passed never reaches
        # eval_model.py. (bool: `eval: true` is the shorthand main() expands.)
        unknown_eval = set(eval_section) - {"enabled", "python",
                                            "compare_against",
                                            "voice-holdout-set"}
        if unknown_eval:
            die(f"unknown keys in the eval section of {path}: "
                f"{sorted(unknown_eval)}")
        vhs = eval_section.get("voice-holdout-set")
        if vhs and not (REPO_ROOT / vhs).is_dir():
            # A typo'd path here would not fail at eval time loudly enough:
            # the eval subprocess would error, the run would still be filed,
            # and the ledger record would simply lack the ranking number -
            # the block silently dropped, exactly what a pre-flight check
            # exists to stop.
            die(f"eval voice-holdout-set {vhs!r} does not exist (looked for "
                f"{REPO_ROOT / vhs}) - fix the path or remove the key; a "
                f"missing holdout set must not silently drop the ranking arm")
    if spec["target"] == "mww":
        # The mww TRAIN stage does not take the corpus-shaping flags (the run
        # script consumes them itself); in this YAML they live in `corpus:`.
        for key in MWW_CORPUS_KEYS & set(spec.get("base") or {}):
            die(f"{key!r} is a corpus flag: put it in the `corpus:` section, not "
                f"`base` - the mww train stage would reject it")
    return spec


def mww_tag(python, wake_word, base_args, seed):
    """The run tag via --print-tag, the same way the run script computes it:
    through train.mww.train, not through train.provenance - the config half
    is only knowable where the resolved config is built, and computing it in
    two places is how the archive and the run directory would drift apart.
    The tag does NOT include --ambient (tag_input excludes the feature-set
    layout), so the print-tag call omits it; the real train run below gets
    it, and nothing changes. `tail -1` for the same reason the run script
    does: the tag is the last thing the process prints."""
    out = subprocess.run([str(python), "-m", "train.mww.train",
                          "--wake-word", wake_word, "--print-tag",
                          *base_args, "--seed", str(seed)],
                         cwd=REPO_ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": PYTHONPATH})
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        die(f"train.mww.train --print-tag exited {out.returncode}")
    return out.stdout.strip().splitlines()[-1]


def trainer_cmd(target, python, wake_word, base_args, point_args, seed,
                tag=None, ambient=(), corpus_reuse=False):
    """The resolved trainer command for one point: base, then grid, then seed.
    A grid key after base deliberately overrides a base default rather than
    depending on dict order. THE single source of truth for what a point
    runs - the dry run prints it, the real loop runs it, and
    tests/test_sweep.py asserts the grid values reach it. (2026-09-22, B1:
    the grid was computed per point but never applied - every point ran the
    base config and was filed under a label the run did not use. A command
    built inline in the loop is exactly where that kind of omission hides;
    a named function with a test is not.)"""
    args = [*base_args, *point_args, "--seed", str(seed)]
    if target == "mww":
        # `--ambient` IS the flag, not a positional: src/train/mww/train.py declares it
        # with nargs="*", so a bare list of directories parses as "zero ambient sets"
        # and the run dies at its own preflight ("no validation_ambient or
        # testing_ambient data in any feature set") with exit 1, not exit 2.
        # Every point of an early sweep failed this way. The run script gets it
        # right (src/scripts/run-mww-training-applesilicon.sh:466), which is how a
        # divergence between the two survived - and tests/test_sweep.py now
        # asserts this shape.
        ambient_args = ["--ambient", *[str(a) for a in ambient]] if ambient else []
        return [str(python), "-m", "train.mww.train",
                "--wake-word", wake_word, "--tag", tag, *ambient_args, *args]
    cmd = [str(python), "-m", "train.oww.train",
           "--wake-word", wake_word, *args]
    if corpus_reuse:
        cmd += ["--corpus", "reuse"]
    return cmd


def tag_names_no_corpus(tag):
    """True when a run tag's corpus half is provenance's 'absent' placeholder.

    src/train/provenance.py:167 renders the corpus digest as the literal "absent" when the
    corpus dir holds no manifest, so such a tag names no audio. Named here (rather than
    an inline `in` test) so tests/test_sweep.py can pin the refusal without running a
    sweep.
    """
    return "-cabsent" in f"-{tag}-"


def corpus_dirs(wake_word, target):
    safe = ledger.safe_name(wake_word)
    corpus = REPO_ROOT / "data" / "corpus" / safe / target
    return corpus, corpus / "features", REPO_ROOT / "output" / safe / target


def corpus_exists(corpus):
    """Any WAV in the corpus, under either layout: oww lays out four subdirs
    (positive/negative x train/test), mww two (positives, negatives). Listing
    both covers either, the same reason the manifest digest does
    (src/train/corpus/manifest.py)."""
    subdirs = ("positives", "negatives",
               "positive/train", "positive/test",
               "negative/train", "negative/test")
    return any((corpus / d).glob("*.wav") for d in subdirs if (corpus / d).is_dir())


def corpus_action(wake_word, target, corpus, features, first_point, dry_run,
                  python, corpus_args, stage_times, first_job=False,
                  auto_point=False):
    """The frozen-corpus rule (module docstring), as a stage or two.

    mww point 1: build (train.mww.corpus, then train.mww.features) - or, if
    a manifest is already on disk, verify with --skip and build features if
    missing: freezing means not re-rendering, and a sweep that re-renders
    TTS audio per point measures the engines, not the knob. Every later
    mww point: --skip verification only. The skip path probes the catalogs
    first (cheap `voices` fetches, no rendering) because it diffs the
    effective voice set against the manifest - the axis the cf9c065b reuse,
    2026-09-22, was blind on - so the engines must be REACHABLE for a reuse,
    which a sweep fleet is. oww: the train run itself owns the corpus
    stage - the first job in auto mode (build + write manifest); every later
    job, point 1's repeats included, with --corpus reuse (the trainer's own
    check_reuse call). It is reuse, not auto, on purpose: auto reuses a
    matching manifest but silently REBUILDS a mismatched one, which is how a
    catalog change between two repeats of point 1 would redraw the frozen
    corpus mid-sweep without an error (bug.md C5, 2026-09-22 - the mirror of
    the cf9c065b silent reuse).

    `first_job` is the first (point, repeat) of the sweep; `first_point` is
    that point. The dry-run labels build only for the former, because the
    later jobs of point 1's own corpus find the manifest the first job wrote.

    Returns (human action string, stage_times); the times the build spent
    land in `stage_times`.
    """
    if dry_run:
        if first_job:
            return (("build" if target == "mww"
                     else "build (by the first train run, corpus auto mode)"),
                    stage_times)
        if auto_point:
            return ("auto per point (corpus_axes: verify, or rebuild - the "
                    "grid itself changed the corpus shaping)", stage_times)
        return "reuse (manifest verified per point)", stage_times

    manifest = corpus / "corpus.json"
    if not first_point and not manifest.is_file():
        die(f"no corpus.json manifest at {corpus} for a later point. A sweep "
            f"cannot reuse a corpus it cannot name: rebuild it (drop the reuse "
            f"and re-run point 1).")

    if target == "oww":
        # The oww train run is the only actor: the first job stays in auto
        # mode (build, write the manifest); every later job - point 1's
        # repeats included - passes --corpus reuse, which makes the trainer
        # call manifest.check_reuse and exit with the diff before any
        # training is spent. Not auto on the repeats: auto would silently
        # rebuild a mismatched manifest (bug.md C5, 2026-09-22).
        return "reuse (verified inside the train run via --corpus reuse)"

    if first_point and not manifest.is_file():
        # BUILD. Into an existing tree the corpus stage refuses to append
        # (it would merge two runs - src/train/mww/corpus.py), so --clean.
        cmd = [python, "-m", "train.mww.corpus", "--wake-word", wake_word,
               *corpus_args]
        if corpus_exists(corpus):
            cmd.append("--clean")
            print(f"\n=== corpus WAVs already at {corpus} without a manifest - "
                  f"rebuilding (--clean)")
        result = run_stage("corpus (build - frozen for this whole sweep)",
                           cmd, stage_times)
        if result.returncode != 0:
            die(f"the corpus build exited {result.returncode}. There is no frozen "
                f"corpus to train on; a sweep that trained anyway would measure a "
                f"corpus it did not ask for.")
        if not manifest.is_file():
            die(f"corpus built but no {manifest} written - the corpus stage "
                f"predates the manifest stage. Re-run the current version.")
        # FEATURES, once: derived from the frozen corpus by tracked code, so
        # one build serves every point and repeat (src/train/mww/features.py).
        if not features.is_dir():
            cmd = [python, "-m", "train.mww.features", "--wake-word", wake_word]
        else:
            # --clean whenever the tree exists: the corpus just changed under
            # it (we just rebuilt it), and features that survive a corpus
            # rebuild are the stale-spectrogram failure features.py documents.
            cmd = [python, "-m", "train.mww.features",
                   "--wake-word", wake_word, "--clean"]
        result = run_stage("features (built once for the whole sweep)", cmd, stage_times)
        if result.returncode != 0:
            die(f"the features stage exited {result.returncode}; every point of this "
                f"sweep would train on the features that were on disk before it.")
        return "build (+ features, built once for the whole sweep)"

    # REUSE (first point with an existing manifest, or any later point):
    # --skip checks the manifest against the shaping AND the effective voice
    # set THIS invocation would use and exits with the diff on any mismatch;
    # it probes the catalogs first (no rendering) because the voice set is
    # resolved from them.
    result = run_stage("corpus (verify frozen - --skip)",
                       [python, "-m", "train.mww.corpus",
                        "--wake-word", wake_word, "--skip", *corpus_args],
                       stage_times)
    # The refusal is fatal, not advisory. run_stage returns the CompletedProcess and
    # the training stages below deliberately carry on when one fails (a failed point is
    # skipped and a re-run retries it); that is the WRONG rule here. --skip exits 1 when
    # the manifest disagrees with this invocation's shaping, and the return code was
    # dropped: a sweep whose grid asked for a different balancing printed
    # "refusing to reuse a corpus shaped differently" for every point and then
    # trained every arm against the previous run's corpus, so the balancing
    # arm measured no balancing at all. Corpus verification is the
    # frozen-independent-variable rule (module docstring), so disagreement means
    # the operator must rebuild - not that the sweep should proceed on the old
    # audio.
    if result.returncode != 0:
        die(f"the corpus on disk is not the corpus this sweep asked for "
            f"(verify --skip exited {result.returncode}, above). The mww corpus is "
            f"frozen for a whole sweep and the sweep never rebuilds a manifest that "
            f"exists, so rebuild it first: "
            f"src/scripts/run-mww-training-applesilicon.sh (or move the corpus dir aside), "
            f"then re-run this sweep.")
    if first_point and not features.is_dir():
        result = run_stage("features (none on disk yet)",
                           [python, "-m", "train.mww.features", "--wake-word", wake_word],
                           stage_times)
        if result.returncode != 0:
            die("the features stage exited non-zero on a corpus with no features on "
                "disk; there is nothing to train on.")
    return "reuse (manifest verified, --skip)"


def artifact_for(target, wake_word, out_dir, tag):
    """The file eval scores, per the run scripts' collection steps: mww the
    ESPHome .json manifest (it carries the cutoff and sliding window, so
    scoring it puts the manifest under test too - the eval-models skill says
    the same), else its .tflite; oww the tagged .onnx."""
    safe = ledger.safe_name(wake_word)
    if target == "mww":
        run_dir = out_dir / tag / MWW_QUANT_DIR
        manifest = run_dir / f"{safe}.json"
        if manifest.is_file():
            return manifest
        return run_dir / MWW_MODEL_FILE
    return out_dir / f"{safe}_{tag}.onnx"


def eval_cmd(eval_python, artifact, json_path, voice_holdout_set=None):
    """The eval subprocess command for one filed run.

    Plain-path invocation, not `python -m eval.eval_model`: the module form
    exists only in the Docker image, where the mount makes the package name
    `eval` with eval_model.py at its top level. On the host the sources live
    in src/eval/src/ (a namespace package at src/eval/), so the module path does not
    resolve - the same lesson as generate_negatives.py (CLAUDE.md "Verify
    before asserting"). One builder for both targets: the oww and mww loops
    share this single call site, so a flag added here reaches both. With
    `voice_holdout_set` set, --voice-holdout-set is passed through and
    eval_model.py writes the voice_holdout_set block into the JSON, which the
    ledger stores verbatim (the synthetic arm-ranking number, low-variance
    versus the real holdout - src/wordlists/voice_holdout.yaml).
    """
    cmd = [str(eval_python), str(REPO_ROOT / "src" / "eval" / "src" / "eval_model.py"),
           "--model", str(artifact), "--json", str(json_path)]
    if voice_holdout_set:
        cmd += ["--voice-holdout-set", str(voice_holdout_set)]
    return cmd


def _fmt_grid_value(value):
    return "on" if value is True else "off" if value is False else value


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="sweep YAML: wake_word, target, base, grid, "
                   "repeats, eval (shape in the module docstring)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the point list, seeds and (mww) tags without "
                        "training, building or writing anything")
    args = p.parse_args()

    spec = load_spec(args.config)
    wake_word, target = spec["wake_word"], spec["target"]
    safe = ledger.safe_name(wake_word)
    python = Path(spec.get("python") or TRAINER_PYTHON[target])
    if not python.is_file():
        die(f"trainer python {python} not found - point `python:` at the right venv")
    eval_spec = spec.get("eval") or {}
    if isinstance(eval_spec, bool):
        eval_spec = {"enabled": eval_spec}
    eval_enabled = bool(eval_spec.get("enabled", True))
    eval_python = Path(eval_spec.get("python") or sys.executable)
    compare_against = eval_spec.get("compare_against")
    voice_holdout_set = eval_spec.get("voice-holdout-set")
    max_faph = float(spec.get("max_faph", 0.2))

    base_args = options_to_args(spec.get("base") or {})
    corpus_section = dict(spec.get("corpus") or {})
    # The run script's defaults for the host route: Piper on 8898, 12
    # speakers per voice. Everything else (kokoro-url, fractions) defaults
    # inside src/train/mww/corpus.py and inherits PIPER_URL/KOKORO_URL.
    corpus_section.setdefault("piper-url", "tcp://127.0.0.1:8898")
    corpus_section.setdefault("piper-speakers", 12)
    corpus_args = options_to_args(corpus_section) if target == "mww" else []
    ambient = ambient_dirs(spec.get("ambient")) if target == "mww" else []

    # Deterministic seed per (point, repeat): base + 1000*point + repeat.
    # The 1000 spacing keeps point blocks apart even if repeats grows, and
    # the whole formula is a function of the YAML alone - that is what makes
    # a sweep re-runnable: a re-run produces the same seeds for the same
    # points, so a point interrupted mid-sweep resumes onto the same tag and
    # the ledger skip (below) catches it.
    grid = spec.get("grid") or {}
    grid_keys = sorted(grid)
    jobs = build_jobs(grid, grid_keys, spec["repeats"], spec["base_seed"])

    corpus, features, out_dir = corpus_dirs(wake_word, target)
    ledger_index = ledger.by_tag(wake_word)
    this_target = ledger_index.get(target, {})

    print(f"sweep: {wake_word!r}  target={target}  python={python}")
    print(f"  grid: {len(jobs) // spec['repeats']} point(s)  repeats: {spec['repeats']}  "
          f"base_seed: {spec['base_seed']}  -> {len(jobs)} run(s)")
    print(f"  corpus: {corpus}")
    print(f"  ledger: {ledger.ledger_path(wake_word)}  "
          f"({len(this_target)} record(s) for {target} so far)")

    if args.dry_run:
        import shlex
        for pi, gp, repeat, seed in jobs:
            combo = "  ".join(f"{k}={_fmt_grid_value(gp[k])}" for k in grid_keys) or "(base)"
            point_args = options_to_args(gp)
            if target == "mww":
                # The tag must see the grid too: it names the config the run
                # will use, and a tag computed without the grid names a config
                # that will not be run (B1).
                tag = mww_tag(python, wake_word, base_args + point_args, seed)
            else:
                tag = "(computed after the run: .last_run_tag)"
            skip = "  [SKIP - already in the ledger]" if tag in this_target else ""
            action, _ = corpus_action(wake_word, target, corpus, features,
                                      pi == 0, True, python, corpus_args, {},
                                      first_job=is_first_job(pi, repeat),
                                      auto_point=(repeat == 0
                                                  and bool(spec.get("corpus_axes"))))
            cmd = trainer_cmd(target, python, wake_word, base_args, point_args,
                              seed, tag=tag if target == "mww" else None,
                              corpus_reuse=(target == "oww"
                                            and job_corpus_reuse(
                                                is_first_job(pi, repeat), repeat,
                                                spec.get("corpus_axes"))))
            print(f"\n  point {pi}  {combo}")
            print(f"    repeat {repeat}  seed {seed}  corpus: {action}")
            print(f"    tag: {tag}{skip}")
            # The full resolved command: the operator check that a grid value
            # actually reaches the trainer (B1 was invisible because the
            # dry run showed labels, not commands).
            print(f"    cmd: {shlex.join(cmd)}")
        if compare_against:
            print(f"\n  compare_against: {compare_against}  "
                  f"(artifact: {artifact_for(target, wake_word, out_dir, compare_against)})")
        return

    failed = []
    trained_tags = []
    for pi, gp, repeat, seed in jobs:
        combo = "  ".join(f"{k}={_fmt_grid_value(gp[k])}" for k in grid_keys) or "(base)"
        point_args = options_to_args(gp)
        first_point = pi == 0
        first_job = is_first_job(pi, repeat)
        stage_times = {}
        print(f"\n{'#' * 72}\n# point {pi}  {combo}  (repeat {repeat} of "
              f"{spec['repeats']}, seed {seed})\n{'#' * 72}")

        if target == "mww":
            # first_job passed on the real path too: the dry-run label and
            # the real stage must agree on what counts as the build (C5's
            # label/command-disagreement class). The mww --skip path itself
            # already keys on the manifest existing, which a repeat of
            # point 0 has by then - no separate hole on this target.
            corpus_action(wake_word, target, corpus, features, first_point,
                          False, python, corpus_args, stage_times,
                          first_job=first_job)
            # The tag is computed AFTER the corpus stage, deliberately: on a sweep
            # that BUILDS its corpus, --print-tag run before the build sees no
            # manifest and files the point's corpus half as "absent" - so the
            # baseline point of a build-first sweep names no corpus, which the
            # ledger reads as different from its own arm's and refuses to
            # compare. --print-tag writes nothing, so running it after the stage
            # costs a second subprocess and changes no state; the corpus the tag
            # names is the one the run trains on.
            if not (corpus / "corpus.json").is_file():
                # The build path writes it and fails loudly if it did not;
                # the verify path dies in --skip. Reaching this means neither
                # ran: the corpus has no identity and the run tag cannot name
                # it, so nothing may be trained on it.
                die(f"no corpus.json manifest at {corpus} - a sweep cannot "
                    f"train on a corpus it cannot name.")
            # The grid must be in the tag too: the tag names the config the
            # run will use (B1 - it used to name one that would not be run).
            tag = mww_tag(python, wake_word, base_args + point_args, seed)
            if tag_names_no_corpus(tag):
                # Belt and braces with the manifest check above: the corpus half of
                # the tag is the digest of the audio under the corpus dir, and
                # provenance renders it "absent" when there is none. A point filed
                # under such a tag names no corpus at all, so the ledger's
                # comparability rule (two runs share a corpus id or are not
                # comparable) would separate it from its own arm.
                die(f"the point's run tag {tag!r} names no corpus (cabsent) - the "
                    f"corpus stage did not leave a manifest at {corpus / 'corpus.json'}")
            if tag in this_target:
                print(f"  SKIP: {tag} is already in the ledger - the run was filed before")
                continue
            cmd = trainer_cmd("mww", python, wake_word, base_args, point_args,
                              seed, tag=tag, ambient=ambient)
            if (out_dir / tag).exists():
                # A failed earlier run left a partial directory and
                # model_train_eval refuses to train into one; deleting it is
                # the only thing --force is for here (src/train/mww/train.py's
                # own text), and the ledger skip above guarantees it cannot
                # delete a completed run.
                print(f"  removing partial {out_dir / tag} from a failed earlier run")
                cmd.append("--force")
        else:
            corpus_action(wake_word, target, corpus, features, first_point,
                          False, python, corpus_args, stage_times,
                          first_job=first_job)
            # C5: NOT (not first_point) - that left every repeat of point 0
            # in auto mode, where a manifest mismatch rebuilds the frozen
            # corpus silently. The one exemption is a
            # declared corpus_axes grid, where the point-0 job of each arm
            # rebuilds on purpose (job_corpus_reuse).
            cmd = trainer_cmd("oww", python, wake_word, base_args, point_args,
                              seed, corpus_reuse=job_corpus_reuse(
                                  first_job, repeat, spec.get("corpus_axes")))

        if target == "oww":
            # The model was WRITTEN is the real signal, not the exit code -
            # openwakeword exits 1 on its own (broken) tflite conversion
            # after a good run, and a failed run leaves the PREVIOUS model
            # in place; both checks are src/train/oww/train.py's, the mtime
            # comparison is the same test run-oww-training.sh re-checks in
            # the shell.
            model_path = out_dir / f"{safe}.onnx"
            before = model_path.stat().st_mtime if model_path.exists() else None
            result = run_stage("training (oww)", cmd, stage_times)
            fresh = model_path.exists() and (before is None
                                             or model_path.stat().st_mtime != before)
            if result.returncode != 0 and not fresh:
                failed.append((pi, seed, result.returncode))
                print(f"  TRAINING FAILED (exit {result.returncode}) - the model was "
                      f"not written; this point is not filed and the sweep continues.")
                continue
            if first_point and not (corpus / "corpus.json").is_file():
                # This check lives AFTER the oww run, not before it, because
                # the oww train run is the stage that writes the manifest in
                # auto mode. Missing here means point 1 produced no corpus
                # identity, and every later point's reuse check compares
                # against one: the sweep stops rather than train on audio no
                # one can name.
                die(f"no corpus.json manifest at {corpus} after the first oww "
                    f"run - the corpus stage did not complete. A sweep cannot "
                    f"continue: every later point must name the SAME corpus, "
                    f"and this one cannot be named.")
            tag = (out_dir / ".last_run_tag").read_text().strip()
            if not tag:
                die(f"no .last_run_tag at {out_dir / '.last_run_tag'} after an oww run")
            if tag in this_target:
                print(f"  SKIP (post-run): {tag} is already in the ledger - not filed again")
                continue
            # The tagged .onnx: run-oww-training.sh's collection step, so the
            # eval artifact and the ledger record share one name.
            tagged = out_dir / f"{safe}_{tag}.onnx"
            tagged.write_bytes(model_path.read_bytes())
            artifact = tagged
        else:
            result = run_stage("training (mww)", cmd, stage_times)
            if result.returncode != 0:
                failed.append((pi, seed, result.returncode))
                print(f"  TRAINING FAILED (exit {result.returncode}) - the point is not "
                      f"filed and the sweep continues; re-running this sweep retries it.")
                continue
            # Non-fatal, exactly as in the run script: a manifest failure does
            # not fail the run, and eval then scores the bare .tflite instead.
            run_stage("manifest (ESPHome)",
                      [python, "-m", "train.mww.manifest",
                       "--wake-word", wake_word, "--run", tag,
                       "--max-faph", str(max_faph)],
                      stage_times)
            artifact = artifact_for(target, wake_word, out_dir, tag)

        eval_block = None
        if eval_enabled:
            json_path = out_dir / "eval" / f"{tag}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            result = run_stage("eval",
                               eval_cmd(eval_python, artifact, json_path,
                                        voice_holdout_set),
                               stage_times)
            if result.returncode != 0 or not json_path.is_file():
                print(f"  EVAL FAILED (exit {result.returncode}) - the run is filed "
                      f"without an eval block, and the ledger says so")
            else:
                # Verbatim: the ledger stores what eval printed, not a
                # re-derivation of it (src/train/ledger.py, module docstring).
                eval_block = json.loads(json_path.read_text())

        config = None
        config_path = out_dir / f"{tag}.config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
        else:
            print(f"  NOTE: no {config_path} - the record's config is null")
        rec = ledger.record(
            wake_word, target=target, tag=tag,
            corpus_id=corpus_manifest.corpus_identity(corpus),
            seed=seed, config=config, wall_time=stage_times,
            eval_block=eval_block, grid=gp)
        this_target[rec["tag"]] = rec
        trained_tags.append(tag)
        print(f"\n  filed: {ledger.ledger_path(wake_word)}  (wall: "
              f"{', '.join(f'{k} {v:.0f}s' for k, v in stage_times.items()) or 'n/a'})")

    print(f"\n{'=' * 72}")
    print(f"done: {len(trained_tags)} run(s) filed, {len(failed)} failed")
    if failed:
        for pi, seed, rc in failed:
            print(f"  point {pi} (seed {seed}) failed (exit {rc}) - not in the "
                  f"ledger, so a re-run of this sweep retries it")

    print()
    print(ledger.summarise(wake_word, grid_keys=grid_keys))

    if compare_against:
        models = []
        for tag in dict.fromkeys(trained_tags + [compare_against]):
            path = artifact_for(target, wake_word, out_dir, tag)
            if not path.is_file():
                die(f"comparison artifact {path} does not exist - there is nothing "
                    f"to compare against; train or locate that run first")
            models.append(path)
        json_path = out_dir / "eval" / f"compare_{datetime.now():%Y%m%d-%H%M%S}.json"
        print(f"\n=== compare: {len(models)} model(s), incl. compare_against "
              f"{compare_against}")
        # The verdict is STREAMED, not reworded: when the bootstrap CIs
        # overlap, compare_models prints NOT DISTINGUISHABLE, and softening
        # that here would be editing the measurement this whole script
        # exists to stop hand-reading.
        run_stage("compare (matched false accepts)",
                  [eval_python, "-m", "eval.compare_models",
                   "--models", *[str(m) for m in models],
                   "--json", str(json_path)], {})

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
