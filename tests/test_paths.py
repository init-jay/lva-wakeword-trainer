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
* No tracked file reintroduces the wordless layout - in code (a join of the
  recordings dir straight to samples/holdout/raw) or in prose (the old path
  strings, which read exactly like the new ones until the word is missing).
"""

import re
import sys
import tempfile
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


def test_holdout_dirs_split_on_runon_per_word():
    word = "okay jarvis"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        holdout = tmp / "holdout"
        (holdout / "speaker1").mkdir(parents=True)
        (holdout / "speaker1_runon").mkdir(parents=True)
        plain = [d.name for d in paths.holdout_dirs(runon=False, root=holdout)]
        runon = [d.name for d in paths.holdout_dirs(runon=True, root=holdout)]
        assert plain == ["speaker1"], plain
        assert runon == ["speaker1_runon"], runon
        # a word with no recordings at all reports empty, not an error
        assert paths.holdout_dirs(runon=False, root=tmp / "nowhere") == []


def test_holdout_dirs_without_a_word_or_root_refuses():
    try:
        paths.holdout_dirs(runon=False)
    except ValueError:
        pass
    else:
        raise AssertionError("no word and no root must refuse, not guess")


def test_warn_if_trained_on_scopes_to_the_word():
    word = "okay jarvis"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        samples = paths.samples_dir(word, root=tmp)
        samples.mkdir(parents=True)
        speaker = samples / "speaker1"
        speaker.mkdir()
        inside = paths.warn_if_trained_on([str(speaker)], word, root=tmp)
        assert inside == [str(speaker)], inside
        # a clip outside the word's samples tree (another word, or the same
        # word's holdout) is not flagged
        other_word = paths.recordings_dir("other word", root=tmp) / "samples" / "speaker1"
        other_word.mkdir(parents=True)
        holdout_clip = paths.holdout_dir(word, root=tmp) / "speaker1"
        holdout_clip.mkdir(parents=True)
        assert paths.warn_if_trained_on(
            [str(other_word), str(holdout_clip)], word, root=tmp) == []


def test_speaker_label_is_the_basename_without_a_word():
    with tempfile.TemporaryDirectory() as tmp:
        speaker = Path(tmp) / "holdout" / "speaker1"
        speaker.mkdir(parents=True)
        # no word: bare basename - a directory somewhere else must still get a
        # label rather than colliding with everything
        assert paths.speaker_label(speaker) == "speaker1"


# ---------------------------------------------------------------------------
# the wordless-join regression guard: the layout above is only as good as the
# discipline that every path goes through a word. The one miss of the sweep
# (src/scripts/f0_coverage.py reading data/recordings/samples via a variable
# join, invisible to the path-constant greps) shows the guard has to scan,
# not spot-check.
# ---------------------------------------------------------------------------

# The old layout's path strings, in prose or code. The <word> placeholder is
# the only correct form: data/recordings/<word>/samples etc.
WORDLESS = {
    "data/recordings/samples": "data/recordings/<word>/samples",
    "data/recordings/holdout": "data/recordings/<word>/holdout",
    "data/recordings/raw": "data/recordings/<word>/raw",
}
# A join of a name straight to one of the recordings children, with no word
# expression in between:  rec / "samples"   (the f0_coverage shape)
ADJACENT_JOIN = re.compile(
    r"['\"]recordings['\"]\s*/\s*['\"](?:samples|holdout|raw)['\"]")
# ...and the same through a variable:  rec = ... / "recordings"  then rec / "samples".
# Only same-line adjacency is flagged - a correct chain rec / word / "samples"
# has the word in between and never matches.
ASSIGN_TO_RECORDINGS = re.compile(r"^(\w+)\s*=[^=].*['\"]recordings['\"]\s*(?:#.*)?$")
# The same through the variable, on any other line:  rec / "samples".
# The variable must sit at the LEFT of the join (start of line or after an
# opening paren/comma): `samples = word / "samples"` has its variable on the
# right and must not read as the recordings root being rejoined.
JOIN_BACK = re.compile(r"(?:^|[\s(,])(\w+)\s*/\s*['\"](?:samples|holdout|raw)['\"]")

SKIP_DIRS = {".git", "__pycache__", ".venv", "data", "output", "logs", "node_modules"}
SKIP_NAMES = {"uv.lock"}


def _tracked_files():
    # The same traversal test_record_pointers.py uses: everything in the repo
    # except the untracked/generated trees (data/, output/ hold all the
    # legitimate wordless-free content, and no tracked file should carry a
    # wordless path).
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or path == Path(__file__):
            continue
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        try:
            yield rel, path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            pass                          # binary: nothing to read


def test_no_tracked_file_carries_a_wordless_recordings_path():
    hits = []
    for rel, lines in _tracked_files():
        for i, ln in enumerate(lines, 1):
            for bad, good in WORDLESS.items():
                if bad in ln:
                    hits.append(f"{rel}:{i}: {ln.strip()}  (word-scoped form: {good})")
            if ADJACENT_JOIN.search(ln):
                hits.append(f"{rel}:{i}: {ln.strip()}  (join with no word in between)")
    assert not hits, (
        "wordless data/recordings path(s) back in the tree - the word is the "
        "first level under recordings/ in this layout:\n    " + "\n    ".join(hits[:20]))


def test_no_variable_rejoins_the_recordings_root_without_a_word():
    # The f0_coverage shape: a variable assigned the wordless recordings root,
    # then joined to a child directory. Two lines apart, invisible to a
    # per-line string grep, so this one tracks the assignment.
    hits = []
    for rel, lines in _tracked_files():
        if rel.suffix != ".py":
            continue
        for i, ln in enumerate(lines):
            m = ASSIGN_TO_RECORDINGS.match(ln.strip())
            if not m:
                continue
            for j, ln2 in enumerate(lines):
                if j == i:
                    continue
                m2 = JOIN_BACK.search(ln2)
                if m2 and m2.group(1) == m.group(1):
                    hits.append(f"{rel}:{j + 1}: {ln2.strip()}  "
                                f"(variable {m.group(1)} assigned the wordless "
                                f"recordings root at line {i + 1})")
    assert not hits, (
        "a variable assigned the wordless recordings root is joined to a "
        "recordings child with no word expression in between:\n    "
        + "\n    ".join(hits[:20]))


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
