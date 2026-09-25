"""Guards for src/wordlists/__init__.py: the wordlist validation.

The rule the module exists to enforce: THE EVAL AND TRAINING LISTS MUST BE
DISJOINT. A phrase in both is trained on and then measured, which turns a
generalisation measurement into a memorisation one - and the number moves
in the direction that looks like success (module docstring). Three
hand-written comments in src/train/corpus/negatives.py used to ask the reader
to check this by hand; `load(..., validate_lists=True)` does it instead.
The tests pin that an overlapping phrase raises with the two lists named,
and that a clean wordlist passes.
"""

import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import wordlists  # noqa: E402

# Every eval category filled with non-overlapping phrases: a baseline that
# validate() accepts, so a test that wants ONE problem sees exactly one.
CLEAN = {
    "wake_word": "test word",
    "train": {
        "negatives": ["close but distinct"],
    },
    "eval": {
        "extend": ["test word continues"],
        "running": ["word and test"],
        "hey_other": ["test another name"],
        "command": ["start recording"],
        "other_ww": ["hey assistant"],
        "general": ["what is the weather today"],
    },
}


def _write_wordlist(data):
    # mkdtemp, deliberately not TemporaryDirectory: load() requires the
    # path to exist at call time, and a with-block would delete it before
    # the test got to read it.
    path = Path(tempfile.mkdtemp()) / "test_word.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_disjoint_wordlist_passes_validation():
    path = _write_wordlist(CLEAN)
    data = wordlists.load(path=path)          # validate_lists=True by default
    assert data is not None
    assert wordlists.validate(data) == []


def test_phrase_in_both_train_and_eval_raises_naming_both_lists():
    data = {
        "wake_word": "test word",
        "train": {
            "negatives": ["test word continues"],   # == eval.extend below
        },
        "eval": {
            "extend": ["test word continues"],
            "running": ["word and test"],
            "hey_other": ["test another name"],
            "command": ["start recording"],
            "other_ww": ["hey assistant"],
            "general": ["what is the weather today"],
        },
    }
    path = _write_wordlist(data)
    try:
        wordlists.load(path=path)
        assert False, "load() should have raised for a train/eval overlap"
    except wordlists.WordlistError as e:
        msg = str(e)
        assert "test word continues" in msg
        assert "train.negatives" in msg
        assert "eval.extend" in msg


def test_validation_can_be_skipped_explicitly():
    # The escape hatch exists (validate_lists=False) - e.g. for reading a
    # wordlist that is mid-migration - and must actually skip.
    data = dict(CLEAN)
    data["eval"] = dict(CLEAN["eval"], general=None)   # empty category
    path = _write_wordlist(data)
    assert wordlists.validate(data) != []
    wordlists.load(path=path, validate_lists=False)     # no raise


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
