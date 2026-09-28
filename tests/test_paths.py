"""Guards for src/eval/src/paths.py: the word-scoped data layout.

Every child of data/recordings/ and data/corpus/ is a wake word. The guarantees
these exist for:

* The holdout positional property now lives INSIDE the word dir -
  data/recordings/<word>/holdout/ is a sibling of <word>/samples/, never a
  child - so the word level must not flatten the two back together.
* A tool that forgets the word must fail loudly (ValueError), not fall back
  to a wordless or single-word default: a default is how one word becomes the
  default.
* The slug is the same rule recipes/, corpus/ and output/ use (spaces to
  underscores, lower) - a second slug would silently split one word's data.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval" / "src"))

import paths  # noqa: E402


def test_word_slug_matches_the_repo_convention():
    assert paths.word_name("okay jarvis") == "okay_jarvis"
    assert paths.word_name("Alpha Bravo") == "alpha_bravo"


def test_word_dirs_nest_under_the_word_not_the_root():
    # The word is the first level under recordings/ and corpus/; the old
    # layout put samples/ and holdout/ at that level, which is what this
    # asserts against.
    rec = paths.recordings_dir("okay jarvis")
    assert rec == paths.RECORDINGS_ROOT / "okay_jarvis"
    assert paths.samples_dir("okay jarvis") == rec / "samples"
    assert paths.holdout_dir("okay jarvis") == rec / "holdout"
    # siblings, not parent/child - the positional guarantee, one level down
    assert paths.holdout_dir("okay jarvis").parent == paths.samples_dir("okay jarvis").parent
    ev = paths.eval_corpus_dir("okay jarvis")
    assert ev == paths.CORPUS_ROOT / "okay_jarvis" / "eval"
    assert paths.negatives_dir("okay jarvis") == ev / "negatives_tts"
    assert paths.positives_dir("okay jarvis") == ev / "positives_tts"


def test_two_words_get_disjoint_trees():
    a = paths.recordings_dir("okay jarvis")
    b = paths.recordings_dir("other word")
    assert a != b and a.parent == b.parent
    assert paths.eval_corpus_dir("okay jarvis") != paths.eval_corpus_dir("other word")


def test_holdout_dirs_split_on_runon_per_word(tmp_path):
    word = "okay jarvis"
    holdout = tmp_path / "holdout"
    (holdout / "speaker1").mkdir(parents=True)
    (holdout / "speaker1_runon").mkdir(parents=True)
    plain = [d.name for d in paths.holdout_dirs(runon=False, root=holdout)]
    runon = [d.name for d in paths.holdout_dirs(runon=True, root=holdout)]
    assert plain == ["speaker1"], plain
    assert runon == ["speaker1_runon"], runon
    # a word with no recordings at all reports empty, not an error
    assert paths.holdout_dirs(runon=False, root=tmp_path / "nowhere") == []


def test_holdout_dirs_without_a_word_or_root_refuses():
    try:
        paths.holdout_dirs(runon=False)
    except ValueError:
        pass
    else:
        raise AssertionError("no word and no root must refuse, not guess")


def test_warn_if_trained_on_scopes_to_the_word(tmp_path):
    word = "okay jarvis"
    samples = paths.samples_dir(word, root=tmp_path)
    samples.mkdir(parents=True)
    speaker = samples / "speaker1"
    speaker.mkdir()
    inside = paths.warn_if_trained_on([str(speaker)], word, root=tmp_path)
    assert inside == [str(speaker)], inside
    # a clip outside the word's samples tree (another word, or the same word's
    # holdout) is not flagged
    other_word = paths.recordings_dir("other word", root=tmp_path) / "samples" / "speaker1"
    other_word.mkdir(parents=True)
    holdout_clip = paths.holdout_dir(word, root=tmp_path) / "speaker1"
    holdout_clip.mkdir(parents=True)
    assert paths.warn_if_trained_on([str(other_word), str(holdout_clip)], word, root=tmp_path) == []


def test_speaker_label_is_the_basename_without_a_word(tmp_path):
    speaker = tmp_path / "holdout" / "speaker1"
    speaker.mkdir(parents=True)
    # no word: bare basename - a directory somewhere else must still get a
    # label rather than colliding with everything
    assert paths.speaker_label(speaker) == "speaker1"
