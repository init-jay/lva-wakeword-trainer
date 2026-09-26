"""Guards for src/wordlists/__init__.py: the wordlist validation.

The rule the module exists to enforce: THE EVAL AND TRAINING LISTS MUST BE
DISJOINT. A phrase in both is trained on and then measured, which turns a
generalisation measurement into a memorisation one - and the number moves
in the direction that looks like success (module docstring). Three
hand-written comments in src/train/corpus/negatives.py used to ask the reader
to check this by hand; `load(..., validate_lists=True)` does it instead.
The tests pin that an overlapping phrase raises with the two lists named,
and that a clean wordlist passes.

The per-word TRAINING data lives here too, since it moved out of
src/train/corpus/: `train.confusable` (the phrases the trainer renders) and
`voices.<engine>.{mispronouncing,unaudited}` (the voices that render this phrase
wrong). Both used to be dicts keyed by wake word inside corpus modules, so a new
wake word meant editing code every word shares - and skipping that edit failed
silently, because a voice that mispronounces the target contributes audio
labelled as the target. These tests pin the schema (an unknown engine or a
misspelled key is rejected rather than ignored), the readers, and the migrated
contents of the example word's file.
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
        "confusable": ["close but distinct"],
    },
    "voices": {
        "kokoro": {"mispronouncing": ["af_wrong"]},
        "piper": {"mispronouncing": ["en_US-x-medium:SPK"], "unaudited": ["en_US-y-low"]},
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
            "confusable": ["test word continues"],   # == eval.extend below
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
        assert "train.confusable" in msg
        assert "eval.extend" in msg


def test_validation_can_be_skipped_explicitly():
    # The escape hatch exists (validate_lists=False) - e.g. for reading a
    # wordlist that is mid-migration - and must actually skip.
    data = dict(CLEAN)
    data["eval"] = dict(CLEAN["eval"], general=None)   # empty category
    path = _write_wordlist(data)
    assert wordlists.validate(data) != []
    wordlists.load(path=path, validate_lists=False)     # no raise


def test_train_phrases_strips_and_dedupes():
    # Category rates are read as fired/n, and a phrase counted twice in the
    # training list is rendered twice - the same reweighting the eval side
    # already guards against.
    data = {"train": {"confusable": ["  hey Serena ", "hey serena", "", "hey"]}}
    assert wordlists.train_phrases(data) == ["hey Serena", "hey"]


def test_voice_sections_are_optional_but_not_freeform():
    # A word nobody has audited yet must still train: the corpus builders warn
    # that they excluded nothing rather than refusing. A MISSPELLED key is the
    # opposite case - it reads as an exclusion that silently does nothing.
    assert wordlists.voice_exclusions({}, "kokoro") == {
        "mispronouncing": [], "unaudited": []}
    assert wordlists.validate({"wake_word": "w", "eval": CLEAN["eval"]}) == []

    typo = {"wake_word": "w", "eval": CLEAN["eval"],
            "voices": {"piper": {"mispronouncng": ["en_US-x-low"]}}}
    problems = "\n".join(wordlists.validate(typo))
    assert "mispronouncng" in problems and "voices.piper" in problems

    unknown_engine = {"wake_word": "w", "eval": CLEAN["eval"],
                      "voices": {"espeak": {"mispronouncing": ["v"]}}}
    assert any("unknown voice engines" in p
               for p in wordlists.validate(unknown_engine))

    unknown_category = {"wake_word": "w", "eval": CLEAN["eval"],
                        "train": {"negatives": ["x"]}}
    assert any("unknown train categories" in p
               for p in wordlists.validate(unknown_category))


def test_unknown_engine_raises_rather_than_returning_empty():
    # Empty is the answer for an unaudited word; for an engine the corpus
    # builders do not read it would hide a caller bug behind a silent no-op.
    try:
        wordlists.voice_exclusions(CLEAN, "espeak")
        assert False, "an unknown engine should raise"
    except wordlists.WordlistError as e:
        assert "espeak" in str(e)


def test_shipped_example_wordlist_carries_the_migrated_tables():
    # The migration's receipt: the four dicts that used to be keyed by wake word
    # in src/train/corpus/ are now data in src/wordlists/hey_seeree.yaml, with
    # nothing lost and nothing gained. The counts are the tables as they stood in
    # code, minus the nine phrases that were also eval phrases (see
    # _disjointness_problems): 38 confusables -> 29, kokoro 6, piper 14
    # mispronouncing + 10 unaudited.
    data = wordlists.load("hey seeree")
    assert wordlists.validate(data) == []

    confusables = wordlists.train_phrases(data)
    assert len(confusables) == 29, f"{len(confusables)} confusables"
    kokoro = wordlists.voice_exclusions(data, "kokoro")
    piper = wordlists.voice_exclusions(data, "piper")
    assert kokoro["mispronouncing"] == [
        "af_alloy", "am_echo", "bf_alice", "bf_lily", "bm_daniel", "bm_fable"]
    assert len(piper["mispronouncing"]) == 14
    assert len(piper["unaudited"]) == 10
    # Speaker-level entries survive the move: the exclusion unit is the speaker,
    # not the model, and a bare name would quietly widen to the whole model.
    assert any(":" in v for v in piper["mispronouncing"])

    # The nine dropped from the TRAINING side, and the eval side untouched: the
    # eval corpus is the measurement instrument, so its 366 clips (148 + 12 + 150
    # + 12 + 8 + 36) are what every recorded false-accept rate was measured on.
    dropped = {"hey season", "hey sedan", "hey seizure", "hey serene", "hey severe",
               "hey cynthia", "hey serena", "hey sienna", "hey simon"}
    assert dropped.isdisjoint({p.lower() for p in confusables})
    sizes = {k: len(v) for k, v in wordlists.eval_categories(data).items()}
    assert sizes == {"extend": 148, "running": 12, "hey_other": 150,
                     "command": 12, "other_ww": 8, "general": 36}, sizes
    assert sum(sizes.values()) == 366


def test_no_per_word_tables_left_in_corpus_code():
    # The invariant the migration bought: no dict keyed by wake word in a module
    # every wake word shares. A re-introduced one is the regression - it is how
    # the next word's author ends up editing corpus code, and how skipping that
    # edit fails silently.
    import re
    gone = ("CONFUSABLE_NEGATIVES", "MISPRONOUNCING_VOICES",
            "MISPRONOUNCING_PIPER_VOICES", "UNAUDITED_PIPER_VOICES")
    for rel in ("train/corpus/negatives.py", "train/corpus/piper.py",
                "train/oww/train.py", "train/mww/corpus.py"):
        text = (REPO_ROOT / "src" / rel).read_text()
        for name in gone:
            # A definition or a lookup, not the pointer comment that says where
            # the table went: `NAME =` / `NAME.get(` / `NAME[`. Neither form can
            # appear in prose.
            hits = [line for line in text.splitlines()
                    if re.search(rf"\b{re.escape(name)}\s*(=|\.get\(|\[)", line)]
            assert not hits, f"{rel} still defines or reads {name}: {hits}"


def test_build_negative_phrases_reads_the_wordlist():
    # The consumer contract: the trainer's negative list is the wordlist's
    # train.confusable plus the word-agnostic base phrases, and never the wake
    # word itself.
    from train.corpus.negatives import (BASE_NEGATIVES, TRAINING_COMMANDS,
                                       build_negative_phrases)
    phrases = build_negative_phrases("hey seeree")
    confusables = wordlists.train_phrases(wordlists.load("hey seeree"))
    lowered = [p.lower() for p in phrases]
    assert all(p.lower() in lowered for p in confusables)
    assert all(p.lower() in lowered for p in BASE_NEGATIVES)
    assert all(p.lower() in lowered for p in TRAINING_COMMANDS)
    assert "hey seeree" not in lowered
    assert len(phrases) == len(set(lowered)), "a phrase is counted twice"
    # --negatives-file still wins, and is the escape hatch for a list kept
    # somewhere else: it bypasses the wordlist entirely.
    path = Path(tempfile.mkdtemp()) / "extra.txt"
    path.write_text("# a comment\nonly this phrase\n")
    override = build_negative_phrases("hey seeree", str(path))
    assert "only this phrase" in override
    assert not any(p.lower() in {c.lower() for c in confusables} for p in override)


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
