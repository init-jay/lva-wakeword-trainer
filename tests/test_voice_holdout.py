"""Guards for the voice holdout (src/wordlists/voice_holdout.yaml, improvement.md P1.2).

The phrase rule in src/wordlists/__init__.py keeps the eval and training corpora
disjoint on PHRASE; the voice holdout keeps them disjoint on VOICE. The
enforcement contract, from the task it came out of:

* the catalog is the source of truth - a held-out entry the live catalog no
  longer offers is an error (the trainers and the renderer exit on the
  `missing` list this module returns), never a silent skip, because the silent
  outcome is the exclusion ending up empty and the corpus quietly training on
  a voice that was supposed to be held out;
* a missing tracked file is a NO-OP (a fresh checkout predating it still
  trains), which is why every call site checks the returned lists rather than
  assuming the file exists.
"""

import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import wordlists  # noqa: E402

from train.corpus.negatives import LEGACY_VOICE_MARKER, MISPRONOUNCING_VOICES  # noqa: E402


def _write_holdout(data):
    # mkdtemp, not TemporaryDirectory: load requires the file to exist at call
    # time, and a with-block would delete it first (same note as test_wordlists).
    path = Path(tempfile.mkdtemp()) / "voice_holdout.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


KOKORO_CATALOG = ["af_aoede", "af_bella", "am_santa", "am_adam", "bf_isabella"]


def test_tracked_file_parses_and_is_nonempty():
    holdout = wordlists.load_voice_holdout()
    assert holdout, "the tracked holdout file is missing or empty"
    assert set(holdout) == {"kokoro", "piper"}
    assert all(isinstance(v, str) and v for v in holdout["kokoro"])
    assert all(isinstance(p, tuple) and len(p) == 2 for p in holdout["piper"])
    assert holdout["kokoro"], "no Kokoro voices reserved - the set has no n"
    assert holdout["piper"], "no Piper pairs reserved - the set has no n"
    # A voice counted twice in the list is one voice counted twice: per-voice
    # n would be misread (the duplicate-drop rule the phrase side already has).
    assert len(holdout["kokoro"]) == len(set(holdout["kokoro"]))
    assert len(holdout["piper"]) == len(set(holdout["piper"]))


def test_heldout_voices_are_usable_not_mispronouncing():
    # A held-out voice that is KNOWN to mispronounce the wake word (or is v0
    # legacy) cannot render an eval positive labelled with the wake word, so
    # putting one in the list is a stale-list error waiting to happen. The
    # checkable half of "in the live catalog" is this table; the live half is
    # enforced at corpus-build time (catalog is the source of truth).
    bad = set(MISPRONOUNCING_VOICES.get("hey_seeree", []))
    for voice in wordlists.load_voice_holdout()["kokoro"]:
        assert voice not in bad, f"{voice} mispronounces the wake word"
        assert LEGACY_VOICE_MARKER not in voice, f"{voice} is v0 legacy"


def test_missing_file_is_a_noop():
    holdout = wordlists.load_voice_holdout(path=Path(tempfile.mkdtemp()) / "nope.yaml")
    assert holdout == {}
    kept, missing = wordlists.exclude_voice_holdout("kokoro", KOKORO_CATALOG, holdout)
    assert kept == KOKORO_CATALOG and missing == []


def test_kokoro_exclusion_removes_and_reports():
    holdout = wordlists.load_voice_holdout(
        path=_write_holdout({"kokoro": ["af_aoede", "am_santa"]}))
    kept, missing = wordlists.exclude_voice_holdout("kokoro", KOKORO_CATALOG, holdout)
    assert missing == []
    assert kept == ["af_bella", "am_adam", "bf_isabella"]


def test_kokoro_stale_entry_is_missing_not_silently_dropped():
    # The failure mode the contract exists for: the engine no longer offers a
    # held-out voice. `missing` is what the callers exit on; a silent skip
    # would let the corpus train on a voice that was supposed to be held out.
    holdout = wordlists.load_voice_holdout(
        path=_write_holdout({"kokoro": ["af_aoede", "af_gone_away"]}))
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
    holdout = wordlists.load_voice_holdout(
        path=_write_holdout({"piper": ["en_GB-alan-medium",
                                       "en_US-libritts_r-medium:USL-A"]}))
    kept, missing = wordlists.exclude_voice_holdout("piper", catalog, holdout)
    assert missing == []
    # The bare entry removes its (single) speaker; the pair entry removes only
    # that pair, the sibling speaker stays.
    assert kept == [("en_US-lessac-medium", None),
                    ("en_US-libritts_r-medium", "USL-B")]


def test_piper_stale_pair_is_missing():
    holdout = wordlists.load_voice_holdout(
        path=_write_holdout({"piper": ["en_GB-alan-medium",
                                       "en_GB-nobody-medium:XXX"]}))
    kept, missing = wordlists.exclude_voice_holdout(
        "piper", [("en_GB-alan-medium", None)], holdout)
    assert ("en_GB-nobody-medium", "XXX") in missing
    assert kept == []


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
