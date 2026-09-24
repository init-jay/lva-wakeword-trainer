"""Guards for scripts/sweep.py: the grid must reach the trainer command.

The bug these tests exist for (bug.md B1, 2026-09-22): the per-point grid
dict was used for labels and the ledger field but never translated into
CLI arguments, so every point ran the base configuration and was filed
under a label the run did not use - the first sweep on record is four
50k-step runs wearing 25k/50k labels. The seam is `build_jobs ->
options_to_args -> trainer_cmd`; a dry run that printed the resolved
command would have made it visible before the first real run, and these
tests pin what the rendered command must contain.

The corpus_axes coverage (e.g. the real-vtlp axis) pins the same seam one level
up: load_spec's gate on the new key, the per-point exemption in
job_corpus_reuse, and the dry-run transcript itself - where the printed
corpus label and the resolved --corpus flag of the command must not
disagree, because that disagreement is exactly the C5 failure class (B1's
lesson: labels lie, commands do not).

The voice-holdout-set coverage pins the eval seam: load_spec
must accept the new key and die on a path that does not exist (a typo'd
holdout path that only showed up as a silently-missing ledger block is the
same "the run believes it passed a knob it never passed" class as B1), and
eval_cmd must carry --voice-holdout-set when configured and omit it when
not.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location("sweep", REPO_ROOT / "scripts" / "sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

GRID = {"training-steps": [25000, 50000]}
GRID_KEYS = list(GRID)


def _jobs():
    return sweep.build_jobs(GRID, GRID_KEYS, repeats=2, base_seed=42)


def test_options_to_args_spelling():
    args = sweep.options_to_args({"training-steps": 25000, "flag": True, "off": False})
    assert "--training-steps" in args and "25000" in args
    assert "--flag" in args
    assert "off" not in " ".join(args)


def test_jobs_expand_grid_times_repeats_with_seed_stride():
    jobs = _jobs()
    assert len(jobs) == 4                       # 2 points x 2 repeats
    seeds = [j[3] for j in jobs]
    # base + 1000*point + repeat: the stride keeps point 1's repeats clear
    # of point 0's, which is what makes a re-run's tags differ by design.
    assert seeds == [42, 43, 1042, 1043]
    grids = [j[1] for j in jobs]
    assert grids.count({"training-steps": 25000}) == 2
    assert grids.count({"training-steps": 50000}) == 2


def test_grid_value_reaches_the_rendered_command_oww():
    # THE B1 assertion: for every job, the grid value is in the exact
    # command string the real loop would run.
    for pi, gp, repeat, seed in _jobs():
        cmd = sweep.trainer_cmd("oww", "/venv/bin/python", "hey seeree",
                                base_args=[], point_args=sweep.options_to_args(gp),
                                seed=seed)
        joined = " ".join(cmd)
        assert str(gp["training-steps"]) in joined, (
            f"grid value {gp['training-steps']} missing from {joined!r} - "
            "the point would run the base config under a grid label (B1)")
        assert "--seed" in joined and str(seed) in joined


def test_grid_value_reaches_the_rendered_command_mww():
    for pi, gp, repeat, seed in _jobs():
        cmd = sweep.trainer_cmd("mww", "/venv/bin/python", "hey seeree",
                                base_args=[], point_args=sweep.options_to_args(gp),
                                seed=seed, tag="some-tag")
        joined = " ".join(cmd)
        assert str(gp["training-steps"]) in joined, f"grid value missing from {joined!r}"
        assert "--tag" in joined and "some-tag" in joined


def test_grid_overrides_base_not_the_other_way_around():
    # A grid key after base overrides a base default: argparse takes the
    # last occurrence, and trainer_cmd puts point_args after base_args.
    base = sweep.options_to_args({"training-steps": 50000})
    point = sweep.options_to_args({"training-steps": 25000})
    cmd = sweep.trainer_cmd("oww", "python", "hey seeree", base, point, seed=1)
    i_base = cmd.index("50000")
    i_point = cmd.index("25000")
    assert i_point > i_base, "grid must come AFTER base to override it"


def test_repeated_grid_key_names_the_config_the_tag_would_name():
    # mww's tag is computed through --print-tag with the SAME argument list
    # the run uses (base + point). The assertion here is that the list the
    # tag sees and the list the run sees are built identically - the tag
    # naming a different config from the run was B1's mww half.
    for pi, gp, repeat, seed in _jobs():
        tag_args = [] + sweep.options_to_args(gp)
        cmd = sweep.trainer_cmd("mww", "python", "hey seeree", [],
                                sweep.options_to_args(gp), seed=seed, tag="t")
        for i, a in enumerate(tag_args):
            assert a in cmd, f"tag argument {a!r} not in the run command"


def test_first_job_is_point_zero_repeat_zero_only():
    assert sweep.is_first_job(0, 0)
    assert not sweep.is_first_job(0, 1)
    assert not sweep.is_first_job(1, 0)
    assert not sweep.is_first_job(1, 1)


def test_point_zero_repeats_run_corpus_reuse_not_auto():
    # C5 (bug.md, 2026-09-22): corpus_reuse=(not first_point) left every
    # repeat of point 0 in auto mode, and auto SILENTLY REBUILDS a
    # mismatched manifest - a TTS catalog change between two repeats of
    # point 0 would redraw the frozen corpus mid-sweep with no error, and
    # every later point would verify against the NEW manifest and pass.
    # Only the first job (point 0, repeat 0) may build; every later job -
    # point 0's repeats included - verifies with --corpus reuse.
    #
    # Routed through job_corpus_reuse with no axes: that is the expression
    # main() computes, and it must reduce to exactly this C5 rule when
    # corpus_axes is absent.
    for pi, gp, repeat, seed in _jobs():
        cmd = sweep.trainer_cmd("oww", "python", "hey seeree", [],
                                sweep.options_to_args(gp), seed=seed,
                                corpus_reuse=sweep.job_corpus_reuse(
                                    sweep.is_first_job(pi, repeat), repeat,
                                    None))
        if pi == 0 and repeat == 0:
            assert "--corpus" not in cmd, (
                f"only the first job builds (auto mode): {cmd!r}")
        else:
            assert cmd[-2:] == ["--corpus", "reuse"], (
                f"point {pi} repeat {repeat} must verify the frozen corpus "
                f"(--corpus reuse), got: {cmd!r}")


def test_mww_dry_run_build_label_only_for_first_job():
    # The mww side of C5's class: the dry-run LABEL and the command must
    # agree. Build is the first JOB only; point 0 repeat 1+ already had the
    # manifest and verifies it (--skip), so its label says reuse. The mww
    # train stage itself takes no corpus mode flag (the corpus stage owns
    # the check), so there is no one-word equivalent to fix here - this
    # test pins that the label was already right and stays that way.
    corpus = REPO_ROOT / "data" / "corpus" / "hey_seeree" / "mww"
    features = corpus / "features"
    for pi, gp, repeat, seed in _jobs():
        action, _ = sweep.corpus_action("hey seeree", "mww", corpus, features,
                                        pi == 0, True, "/python", [], {},
                                        first_job=sweep.is_first_job(pi, repeat))
        if pi == 0 and repeat == 0:
            assert action == "build"
        else:
            assert action == "reuse (manifest verified per point)"


# -- corpus_axes --------------------------------------------------------

# A minimal oww spec; main() needs `python:` to be an existing file, so it
# points at the interpreter running the test (the dry run launches nothing).
def _write_spec(overrides):
    spec = {"wake_word": "dry word", "target": "oww",
            "python": sys.executable,
            "grid": {"training-steps": [25000, 50000]}, "repeats": 2}
    spec.update(overrides)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f)
    f.close()
    return f.name


def _load(**overrides):
    path = _write_spec(overrides)
    try:
        return sweep.load_spec(path)
    finally:
        os.unlink(path)


def _expect_dies(overrides, fragments):
    """load_spec must exit, and its SWEEP FAILED line must carry every fragment."""
    if isinstance(fragments, str):
        fragments = [fragments]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            _load(**overrides)
        except SystemExit:
            pass
        else:
            raise AssertionError("load_spec accepted the spec; expected it to die")
    for frag in fragments:
        assert frag in err.getvalue(), f"{frag!r} not in {err.getvalue()!r}"


def _dry_run(overrides):
    """sweep.main() --dry-run against a temp YAML, stdout captured. The oww
    dry run starts NOTHING (no --print-tag, no subprocess), and the dummy
    wake word keeps the ledger read empty, so it is safe in-process."""
    path = _write_spec(overrides)
    saved = sys.argv
    sys.argv = [str(REPO_ROOT / "scripts" / "sweep.py"), path, "--dry-run"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            sweep.main()
    finally:
        sys.argv = saved
        os.unlink(path)
    return out.getvalue()


def _jobs_from_dry(out):
    """{(point, repeat): (corpus label, cmd text)} from a dry-run transcript:
    the label and the resolved command the operator sees, as one pair - the
    pair C5 says must not disagree."""
    jobs, cur = {}, None
    for line in out.splitlines():
        if line.startswith("  point "):
            cur = {"pi": int(line.split()[1])}
        elif cur is not None and line.startswith("    repeat "):
            parts = line.split("corpus: ", 1)
            cur["repeat"] = int(line.split()[1])
            cur["label"] = parts[1] if len(parts) == 2 else ""
        elif cur is not None and line.startswith("    cmd: "):
            cur["cmd"] = line[len("    cmd: "):]
            jobs[(cur["pi"], cur["repeat"])] = (cur["label"], cur["cmd"])
    return jobs


def test_job_corpus_reuse_c5_rule():
    # C5, as the function main() calls: with no corpus
    # axes, only the first job (point 0, repeat 0) runs --corpus auto; every
    # later job - point 0's own repeats included - runs --corpus reuse, which
    # refuses a mismatch instead of silently rebuilding the frozen corpus.
    table = [(True, 0, False), (False, 0, True),
             (False, 1, True), (False, 2, True)]
    for first_job, repeat, want in table:
        got = sweep.job_corpus_reuse(first_job, repeat, None)
        assert got is want, (
            f"first_job={first_job} repeat={repeat}: {got!r}, want {want!r}")


def test_job_corpus_reuse_axes_exemption_per_point_only():
    # The corpus_axes exemption: a declared corpus axis freezes the corpus
    # PER POINT - every point's repeat 0 runs auto (verify, or rebuild, the
    # grid's declared intent), repeats inside a point still refuse. The
    # first job's row is unchanged by the exemption.
    axes = ["real-vtlp"]
    table = [(True, 0, False), (False, 0, False),
             (False, 1, True), (False, 2, True)]
    for first_job, repeat, want in table:
        got = sweep.job_corpus_reuse(first_job, repeat, axes)
        assert got is want, (
            f"first_job={first_job} repeat={repeat} axes={axes}: "
            f"{got!r}, want {want!r}")


def test_load_spec_corpus_axes_accepted():
    spec = _load(grid={"real-vtlp": ["", "ryan=30"]},
                 corpus_axes=["real-vtlp"])
    assert spec["corpus_axes"] == ["real-vtlp"]


def test_load_spec_corpus_axes_refusals():
    # oww only (the mww corpus stage is owned by the sweep runner); the axis
    # must be a grid key; and it must be a corpus-shaping flag, else the
    # "rebuild" changes nothing and only adds TTS noise.
    _expect_dies({"target": "mww", "grid": {"batch-size": [8, 16]},
                  "corpus_axes": ["batch-size"]},
                 "corpus_axes is oww-only")
    _expect_dies({"corpus_axes": ["real-vtlp"]},
                 "grid does not vary it")
    _expect_dies({"corpus_axes": ["training-steps"]},
                 "not a corpus-shaping flag")


def test_load_spec_grid_corpus_key_without_axes_still_refused():
    # Regression guard on the frozen-corpus refusal (module docstring): the
    # corpus_axes escape hatch must not swallow it - a corpus-shaping grid
    # key that is NOT declared still dies, and the refusal now points at the
    # hatch.
    _expect_dies({"grid": {"real-vtlp": ["", "ryan=30"]}},
                 ["shapes the CORPUS", "declare it under `corpus_axes:`"])


def test_oww_axes_dry_run_label_matches_command():
    # THE C5 failure class is a label and a command disagreeing, so for a
    # per-point frozen sweep the assertion is table-driven over all four
    # jobs: the printed corpus label and the resolved --corpus flag must
    # match, and both must match the per-point expectation. Point 0 builds,
    # point 1's repeat 0 goes auto on purpose (the grid changed the corpus
    # shaping); every other job verifies with --corpus reuse.
    out = _dry_run({"grid": {"real-vtlp": ["", "ryan=30"]},
                    "corpus_axes": ["real-vtlp"]})
    jobs = _jobs_from_dry(out)
    assert set(jobs) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    expected = {
        (0, 0): ("build (by the first train run, corpus auto mode)", False),
        (0, 1): ("reuse (manifest verified per point)", True),
        (1, 0): ("auto per point (corpus_axes: verify, or rebuild - the "
                 "grid itself changed the corpus shaping)", False),
        (1, 1): ("reuse (manifest verified per point)", True),
    }
    for key, (want_label, want_reuse) in expected.items():
        label, cmd = jobs[key]
        assert label == want_label, f"point {key} label: {label!r}"
        has_reuse = "--corpus reuse" in cmd
        assert has_reuse is want_reuse, f"point {key} cmd: {cmd!r}"
        assert label.startswith("reuse") is has_reuse, (
            f"point {key}: label {label!r} and cmd {cmd!r} disagree (C5 class)")


def test_oww_non_axes_dry_run_still_frozen_per_sweep():
    # C5 unchanged where corpus_axes is absent, at the dry-run level: only
    # the first job is auto; every other job - point 1's own repeat 0
    # included - verifies, and its resolved command carries --corpus reuse
    # (the existing trainer_cmd test pins the same rule one level down).
    out = _dry_run({})
    jobs = _jobs_from_dry(out)
    assert set(jobs) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    expected = {
        (0, 0): ("build (by the first train run, corpus auto mode)", False),
        (0, 1): ("reuse (manifest verified per point)", True),
        (1, 0): ("reuse (manifest verified per point)", True),
        (1, 1): ("reuse (manifest verified per point)", True),
    }
    for key, (want_label, want_reuse) in expected.items():
        label, cmd = jobs[key]
        assert label == want_label, f"point {key} label: {label!r}"
        has_reuse = "--corpus reuse" in cmd
        assert has_reuse is want_reuse, f"point {key} cmd: {cmd!r}"
        assert label.startswith("reuse") is has_reuse, (
            f"point {key}: label {label!r} and cmd {cmd!r} disagree (C5 class)")


# -- voice-holdout-set --------------------------------------------------

def test_load_spec_eval_accepts_voice_holdout_set():
    # The behaviour pinned: load_spec accepts the key and passes it through
    # VERBATIM (eval_cmd takes whatever the spec said, unchanged). The
    # existence check it runs before accepting resolves the spec's value
    # against the repo root, so the path here has to exist without relying
    # on a corpus the repo's data tree would hold: an absolute temp dir is
    # what the check accepts as an absolute path, and the assertion is on
    # exactly that string, unmodified.
    holdout = tempfile.mkdtemp(prefix="sweep-voice-holdout-")
    try:
        spec = _load(eval={"enabled": True, "python": sys.executable,
                           "voice-holdout-set": holdout})
    finally:
        shutil.rmtree(holdout)
    assert spec["eval"]["voice-holdout-set"] == holdout


def test_load_spec_eval_dies_on_missing_voice_holdout_set():
    # A typo'd path must fail at spec load, before a single training hour is
    # spent: at eval time the failure would still file the run, and the
    # ledger block would lack the ranking number - the arm silently dropped.
    _expect_dies({"eval": {"voice-holdout-set": "no/such/holdout/dir"}},
                 ["voice-holdout-set", "does not exist"])


def test_load_spec_eval_refuses_unknown_key():
    # Same typo refusal as the top level, one level down: an unknown eval
    # key means a flag the operator believes they passed never reaches
    # eval_model.py.
    _expect_dies({"eval": {"enabled": True, "voice-holdout-set-": "x"}},
                 "unknown keys in the eval section")


def test_eval_cmd_carries_flag_when_configured():
    cmd = sweep.eval_cmd("/venv/bin/python", "model.onnx", "eval.json",
                         voice_holdout_set="data/corpus/eval/voice_holdout_tts")
    assert cmd[0] == "/venv/bin/python"
    assert "eval_model.py" in cmd[1]
    assert cmd[-2:] == ["--voice-holdout-set",
                        "data/corpus/eval/voice_holdout_tts"], f"{cmd!r}"


def test_eval_cmd_omits_flag_when_not_configured():
    cmd = sweep.eval_cmd("/venv/bin/python", "model.onnx", "eval.json")
    assert "--voice-holdout-set" not in cmd
    assert "--model" in cmd and "model.onnx" in cmd
    assert "--json" in cmd and "eval.json" in cmd


def test_mww_ambient_dirs_follow_the_ambient_flag():
    # The bug: trainer_cmd spread the auto-discovered ambient dirs straight after
    # --tag, positional. train/mww/train.py's --ambient is nargs="*", so the bare
    # directories parsed as "zero ambient sets" and the run died at its own preflight
    # ("no validation_ambient or testing_ambient data in any feature set", exit 1) -
    # and every point of an early sweep failed this way. argparse does NOT reject
    # the positionals, so the failure looked like a data problem, not a
    # command-construction problem.
    amb = ["data/external/mww_ambient/dinner_party",
           "data/external/mww_ambient/dinner_party_eval",
           "data/external/mww_ambient/speech"]
    cmd = sweep.trainer_cmd("mww", "python", "hey seeree", base_args=[],
                            point_args=["--training-steps", "10000"], seed=1,
                            tag="t", ambient=amb)
    assert "--ambient" in cmd, f"--ambient missing from {cmd!r}"
    i = cmd.index("--ambient")
    assert cmd[i + 1:i + 1 + len(amb)] == amb, f"ambient dirs not bound to the flag: {cmd!r}"
    # And nothing positional left over after the ambient block.
    tail = cmd[i + 1 + len(amb):]
    assert all(a.startswith("--") or j == 0 or tail[j - 1].startswith("--")
               for j, a in enumerate(tail) if not a.startswith("-")), \
        f"a value in {tail!r} is not attached to a flag"


def test_mww_no_ambient_means_no_empty_ambient_flag():
    cmd = sweep.trainer_cmd("mww", "python", "hey seeree", base_args=[],
                            point_args=[], seed=1, tag="t", ambient=[])
    joined = " ".join(cmd)
    assert "--ambient" not in joined, \
        "--ambient with no dirs would claim ambient training the run does not have"


def test_tag_naming_no_corpus_is_refused():
    # The resume/label half of the same accident: --print-tag run BEFORE a corpus
    # build files the point as <commit>-cabsent-<config>, and the ledger groups by
    # corpus id - so the baseline point of a build-first sweep lands beside its arm
    # but is not comparable to it. sweep.py now computes the tag after the corpus
    # stage and refuses a cabsent tag outright; this pins the refusal.
    assert sweep.tag_names_no_corpus("f2865bc-cabsent-hcbfb214")
    assert sweep.tag_names_no_corpus("cabsent-hcbfb214")     # corpus half first, too
    assert not sweep.tag_names_no_corpus("f2865bc-c6bb4cca-hcbfb214")
    assert not sweep.tag_names_no_corpus("f2865bc-dirty-c6bb4cca-h1")


# -- corpus verification is a whole-sweep gate --------------------------

def _fake_python(dir_path, exit_line):
    """A stand-in interpreter: a sh script that ignores its arguments and
    exits as directed, in the test's temp tree, executable. corpus_action
    launches it as a plain file, so no real module has to resolve."""
    fake = dir_path / "python"
    fake.write_text(f"#!/bin/sh\n{exit_line}\n")
    fake.chmod(0o755)
    return str(fake)


def test_a_corpus_that_fails_verification_is_fatal_not_advisory():
    # The claim this branch makes, pinned: on a later point (first_point
    # False) the mww reuse path runs the corpus stage with --skip purely to
    # VERIFY the on-disk manifest against this invocation, and a non-zero
    # exit must die the WHOLE sweep through die() (SystemExit) - never a
    # per-point skip. The downstream training stages survive a failed point
    # on purpose, because a re-run retries it; that is the wrong rule here,
    # because carrying on would train every arm against a corpus the sweep
    # never asked for and file them as if the arm had run - the frozen
    # corpus this sweep's measurements depend on, silently swapped. The
    # call is minimal on purpose: a manifest stub (the verify branch only
    # needs corpus.json to exist; the fake interpreter decides the outcome),
    # first_point False so the build and features branches stay out, and no
    # corpus_args - so nothing under data/ or output/ has to be present.
    tmp = Path(tempfile.mkdtemp(prefix="sweep-verify-"))
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "corpus.json").write_text("{}")
    features = tmp / "features"
    stage_times = {}
    try:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            try:
                sweep.corpus_action("dry word", "mww", corpus, features,
                                    False, False, _fake_python(tmp, "exit 1"),
                                    [], stage_times, first_job=True)
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    "corpus verification failed but the sweep kept going")
        # The SWEEP FAILED line (die's channel: stderr) carries the refusal.
        assert "not the corpus this sweep asked for" in err.getvalue(), \
            err.getvalue()

        # The converse: the same call with a verify that agrees returns the
        # reuse label instead of dying (and the features stage does not run:
        # that only happens on the first point).
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            action = sweep.corpus_action("dry word", "mww", corpus, features,
                                         False, False, _fake_python(tmp, "exit 0"),
                                         [], stage_times, first_job=True)
        assert action == "reuse (manifest verified, --skip)", f"{action!r}"
    finally:
        shutil.rmtree(tmp)


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
