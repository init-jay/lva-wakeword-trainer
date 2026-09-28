"""Guards for train/external_data.py: the pre-corpus external-data check.

A missing or half-finished data/external used to fail after corpus
generation - the most expensive stage - with a deep stack trace. The guard
refuses before that spend, names the idempotent download script, and lists
exactly which paths are missing. Existence only, never completeness: a
half-finished download leaves the directory present, and a wrong size
check that rejects a good tree is worse than no check.

Plain-python convention (tests/_runner.py): no pytest, no fixture files -
the fake trees live in tempfile and the refusals are captured by catching
SystemExit.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from train import external_data  # noqa: E402
from train.mww import config as mww_config  # noqa: E402


def _refusal(target, paths):
    """Run check() on a freshly empty tree; return the refusal message, or None.

    A caught sys.exit(msg) prints nothing, so there is no stdout to capture.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            external_data.check(tmp, target, paths)
        except SystemExit as e:
            return str(e.code)
    return None


def _build(target, tmp, except_=()):
    """A complete fake tree under tmp, minus the named directories/files."""
    tmp = Path(tmp)
    paths = (external_data.oww_external_paths(tmp) if target == "oww"
             else external_data.mww_external_paths(tmp))
    for p in paths:
        if p.name in except_:
            continue
        p.mkdir(parents=True) if p.suffix != ".npy" else p.touch()
    return paths


def test_oww_missing_tree_refuses_naming_the_script_and_every_path():
    with tempfile.TemporaryDirectory() as tmp:
        paths = external_data.oww_external_paths(tmp)
        msg = _refusal("oww", paths)
    assert msg is not None, "an empty external tree must be refused"
    assert "./src/scripts/download-external-data.sh oww" in msg
    for p in paths:
        assert str(p) in msg, f"{p} missing from the refusal"


def test_oww_partial_tree_names_only_what_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _build("oww", tmp, except_={"validation_set_features.npy"})
        msg = _refusal("oww", paths)
    assert msg is not None
    assert "./src/scripts/download-external-data.sh oww" in msg
    for p in paths:
        if p.name == "validation_set_features.npy":
            assert str(p) in msg
        else:
            assert str(p) not in msg, f"{p} present, must not be named"


def test_oww_complete_tree_passes():
    with tempfile.TemporaryDirectory() as tmp:
        _build("oww", tmp)
        external_data.check(tmp, "oww",
                            external_data.oww_external_paths(tmp))  # no SystemExit


def test_mww_missing_tree_refuses_naming_the_script_and_every_path():
    with tempfile.TemporaryDirectory() as tmp:
        paths = external_data.mww_external_paths(tmp)
        msg = _refusal("mww", paths)
    assert msg is not None, "an empty external tree must be refused"
    assert "./src/scripts/download-external-data.sh mww" in msg
    for p in paths:
        assert str(p) in msg, f"{p} missing from the refusal"
    # The shared corpora are fetched by the oww target, not mww - the note.
    assert "not `mww`" in msg and "download-external-data.sh oww" in msg


def test_mww_only_the_ambient_set_missing_names_it_without_the_shared_note():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        paths = _build("mww", tmp, except_={"speech"})
        msg = _refusal("mww", paths)
    assert msg is not None
    assert str(tmp / "mww_ambient" / "speech") in msg
    assert "no_speech" not in msg
    assert "not `mww`" not in msg, "no shared corpora missing - no shared note"


def test_mww_complete_tree_passes():
    with tempfile.TemporaryDirectory() as tmp:
        _build("mww", tmp)
        external_data.check(tmp, "mww",
                            external_data.mww_external_paths(tmp))  # no SystemExit


def test_mww_paths_follow_the_mww_config_constants():
    # Derivation, not a retyped copy: a layout change in the config must move the guard.
    names = [p.name for p in external_data.mww_external_paths("x")]
    assert names == ([Path(d).name for d in mww_config.IMPULSE_DIRS
                      + mww_config.BACKGROUND_DIRS]
                     + [s.rsplit("/", 1)[-1] for s in external_data.AMBIENT_SETS])


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
