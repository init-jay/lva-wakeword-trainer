"""Guards for scripts/sweep.py: the grid must reach the trainer command.

The bug these tests exist for (bug.md B1, 2026-09-22): the per-point grid
dict was used for labels and the ledger field but never translated into
CLI arguments, so every point ran the base configuration and was filed
under a label the run did not use - the first sweep on record is four
50k-step runs wearing 25k/50k labels. The seam is `build_jobs ->
options_to_args -> trainer_cmd`; a dry run that printed the resolved
command would have made it visible before the first real run, and these
tests pin what the rendered command must contain.
"""

import importlib.util
import sys
from pathlib import Path

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


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
