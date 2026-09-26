"""Guards for the voice holdout (a `voice_holdout:` section of the wordlist,
improvement.md P1.2).

The phrase rule in src/wordlists/__init__.py keeps the eval and training
corpora disjoint on PHRASE; the voice holdout keeps them disjoint on VOICE.
One configuration per wake word: the section is a tracked part of the word,
and a voice reservation belongs to the word that measured it. The enforcement
contract, from the task it came out of:

* the catalog is the source of truth - a held-out entry the live catalog no
  longer offers is an error (the trainers and the renderer exit on the
  `missing` list the exclusion returns), never a silent skip, because the
  silent outcome is the exclusion ending up empty and the corpus quietly
  training on a voice that was supposed to be held out;
* a wordlist without the section is a NO-OP (a fresh word before its first
  reservation still trains), which is why every call site checks the returned
  lists rather than assuming the section exists.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import wordlists  # noqa: E402

from train.corpus.negatives import LEGACY_VOICE_MARKER  # noqa: E402


def _wl(section):
    # A loaded-wordlist shape with the holdout section under test: the reader
    # and the validator both work on the dict, so no temp file is needed. The
    # eval section carries the six categories validate() requires, with
    # phrases that are neither the wake word nor each other's (the
    # disjointness rule).
    data = {
        "wake_word": "test phrase", "_path": "(test)",
        "eval": {
            "extend": ["test phrase now"],
            "running": ["test the phrasebook"],
            "hey_other": ["hey someone else"],
            "command": ["open the light"],
            "other_ww": ["ok computer"],
            "general": ["what is the weather like"],
        },
    }
    if section is not None:
        data["voice_holdout"] = section
    return data


KOKORO_CATALOG = ["af_aoede", "af_bella", "am_santa", "am_adam", "bf_isabella"]


def test_tracked_section_parses_and_is_nonempty():
    data = wordlists.load("hey seeree")
    holdout = wordlists.voice_holdout(data)
    assert holdout, "the example wordlist's voice_holdout: section is missing or empty"
    assert set(holdout) == {"kokoro", "piper"}
    assert all(isinstance(v, str) and v for v in holdout["kokoro"])
    assert all(isinstance(p, tuple) and len(p) == 2 for p in holdout["piper"])
    assert holdout["kokoro"], "no Kokoro voices reserved - the set has no n"
    assert holdout["piper"], "no Piper pairs reserved - the set has no n"
    # A voice counted twice in the section is one voice counted twice: per-voice
    # n would be misread (the duplicate-drop rule the phrase side already has).
    assert len(holdout["kokoro"]) == len(set(holdout["kokoro"]))
    assert len(holdout["piper"]) == len(set(holdout["piper"]))


def test_heldout_voices_are_usable_not_mispronouncing():
    # A held-out voice that is KNOWN to mispronounce ITS OWN wake word (or is v0
    # legacy) cannot render an eval positive labelled with that word, so putting
    # one in the section is a stale-list error waiting to happen. The checkable
    # half of "in the live catalog" is the exclusion table; the live half is
    # enforced at corpus-build time (catalog is the source of truth).
    #
    # Per WORD, not repo-level: each section must be usable for the word that
    # carries it, because it only ever renders that word's eval set.
    for path in sorted(wordlists.WORDLIST_DIR.glob("*.yaml")):
        data = wordlists.load(path=path)
        holdout = wordlists.voice_holdout(data)
        if not holdout["kokoro"]:
            continue  # a sectionless word is a no-op, not a failure
        for voice in holdout["kokoro"]:
            assert LEGACY_VOICE_MARKER not in voice, f"{voice} is v0 legacy"
            bad = set(wordlists.voice_exclusions(data, "kokoro")["mispronouncing"])
            assert voice not in bad, (
                f"{voice} mispronounces {data.get('wake_word')!r} - it cannot render "
                f"an eval positive for that word")


def test_missing_section_is_a_noop():
    # An absent section reads as BOTH engines empty - the same shape a present
    # but unpopulated section reads as, so call sites check the lists, not
    # the presence of the section itself.
    holdout = wordlists.voice_holdout(_wl(None))
    assert holdout == {"kokoro": [], "piper": []}
    kept, missing = wordlists.exclude_voice_holdout("kokoro", KOKORO_CATALOG, holdout)
    assert kept == KOKORO_CATALOG and missing == []


def test_kokoro_exclusion_removes_and_reports():
    holdout = wordlists.voice_holdout(
        _wl({"kokoro": ["af_aoede", "am_santa"]}))
    kept, missing = wordlists.exclude_voice_holdout("kokoro", KOKORO_CATALOG, holdout)
    assert missing == []
    assert kept == ["af_bella", "am_adam", "bf_isabella"]


def test_kokoro_stale_entry_is_missing_not_silently_dropped():
    # The failure mode the contract exists for: the engine no longer offers a
    # held-out voice. `missing` is what the callers exit on; a silent skip
    # would let the corpus train on a voice that was supposed to be held out.
    holdout = wordlists.voice_holdout(
        _wl({"kokoro": ["af_aoede", "af_gone_away"]}))
    kept, missing = wordlists.exclude_voice_holdout("kokoro", KOKORO_CATALOG, holdout)
    assert missing == ["af_gone_away"]  # strings for kokoro, (voice, speaker) for piper
    assert kept == ["af_bella", "am_santa", "am_adam", "bf_isabella"]


def test_piper_pair_and_bare_entries():
    catalog = [
        ("en_GB-alan-medium", None),
        ("en_US-lessac-medium", None),
        ("en_US-libritts_r-medium", "USL-A"),
        ("en_US-libritts_r-medium", "USL-B"),
    ]
    holdout = wordlists.voice_holdout(
        _wl({"piper": ["en_GB-alan-medium",
                       "en_US-libritts_r-medium:USL-A"]}))
    kept, missing = wordlists.exclude_voice_holdout("piper", catalog, holdout)
    assert missing == []
    # The bare entry removes its (single) speaker; the pair entry removes only
    # that pair, the sibling speaker stays.
    assert kept == [("en_US-lessac-medium", None),
                    ("en_US-libritts_r-medium", "USL-B")]


def test_piper_stale_pair_is_missing():
    holdout = wordlists.voice_holdout(
        _wl({"piper": ["en_GB-alan-medium",
                       "en_GB-nobody-medium:XXX"]}))
    kept, missing = wordlists.exclude_voice_holdout(
        "piper", [("en_GB-alan-medium", None)], holdout)
    assert ("en_GB-nobody-medium", "XXX") in missing
    assert kept == []


def test_section_shape_is_validated():
    # The phrase-side checks run on every wordlist at load time; the holdout
    # section rides on the same validation, so a malformed section fails the
    # load, not a corpus build.
    assert wordlists.validate(_wl(None)) == []
    assert wordlists.validate(_wl({"kokoro": ["af_aoede"],
                                   "piper": ["en_GB-alan-medium"]})) == []
    problems = wordlists.validate(_wl({"espeak": ["x"]}))
    assert len(problems) == 1 and "espeak" in problems[0], problems
    problems = wordlists.validate(_wl({"kokoro": ["af_aoede", ""]}))
    assert len(problems) == 1 and "voice_holdout.kokoro" in problems[0], problems
    problems = wordlists.validate(_wl({"kokoro": "af_aoede"}))
    assert len(problems) == 1 and "voice_holdout.kokoro" in problems[0], problems
    # A section of the wrong SHAPE at read time is an error, not a no-op: a
    # no-op is the absent section, and a mapping that parses as a scalar would
    # otherwise be silently ignored.
    try:
        wordlists.voice_holdout(_wl("af_aoede"))
    except wordlists.WordlistError:
        pass
    else:
        raise AssertionError("a non-mapping section must raise, not no-op")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
