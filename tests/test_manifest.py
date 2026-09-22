"""Guards for train/corpus/manifest.py: the frozen-corpus identity check.

The failure mode this module exists to make loud: `--skip-corpus` /
`--corpus reuse` used to fail SILENTLY in both directions - a changed
--samples-per-voice was ignored, a changed --augmentation-rounds reused
stale features as if nothing changed - so a comparison run was scored
against a corpus it believed it had asked for but had not (module
docstring, "THE REFUSE-STALE-REUSE RULE"). The tests pin the diff
contract `check_reuse` prints and exits on, and the identity number the
run tag's data half consumes.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.corpus.manifest as man  # noqa: E402

WW = "zz test ww"   # made-up word: no on-disk corpus to entangle the tests with


def _manifest(seed=42, shaping=None):
    return {
        "schema": man.SCHEMA,
        "wake_word": WW,
        "target": "mww",
        "seed": seed,
        "shaping": {"samples_per_voice": 10, "runon_fraction": 0.2}
        if shaping is None else shaping,
    }


# ---------------------------------------------------------------------------
# matches_requested
# ---------------------------------------------------------------------------

def test_matches_requested_matching_request_is_empty():
    assert man.matches_requested(_manifest(),
                                 {"samples_per_voice": 10, "runon_fraction": 0.2}) == []


def test_matches_requested_differing_key_names_both_values():
    diffs = man.matches_requested(_manifest(), {"samples_per_voice": 12})
    assert len(diffs) == 1
    assert "samples_per_voice" in diffs[0]
    assert "12" in diffs[0]       # what was requested
    assert "10" in diffs[0]       # what the corpus actually has


def test_matches_requested_missing_key_is_reported_not_skipped():
    diffs = man.matches_requested(_manifest(), {"no_such_flag": 3})
    assert len(diffs) == 1
    assert "no_such_flag" in diffs[0]
    assert "has no shaping.no_such_flag" in diffs[0]


def test_matches_requested_seed_is_compared_at_top_level_not_under_shaping():
    # The seed is recorded at the manifest's top level, not under "shaping"
    # (write_manifest) - a request for it must be checked there, and a
    # matching seed must not produce a diff just because "shaping" lacks it.
    assert man.matches_requested(_manifest(), {"seed": 42}) == []
    diffs = man.matches_requested(_manifest(), {"seed": 7})
    assert diffs == ["requested seed=7, corpus has 42"]


# ---------------------------------------------------------------------------
# matches_requested: the voice-set axis
# ---------------------------------------------------------------------------
#
# The cf9c065b reuse, 2026-09-22: a post-reservation run matched the
# pre-reservation manifest on EVERY shaping flag, because the voice holdout
# exclusion lives in the voice set (a top-level manifest field) and not in
# any flag - so it trained on seven held-out voices. These tests pin the
# axis: any movement of the effective voice set (voice added, voice removed,
# holdout introduced, piper pair changed) must refuse reuse, and a matching
# set must pass regardless of entry order or tuple/list spelling.

def _manifest_with_voices(voices):
    m = _manifest()
    m["voices"] = voices
    return m


def test_matches_requested_matching_voice_set_is_empty():
    # Recorded side as json hands it back (piper pairs as LISTS), requested
    # side as the trainer builds it (tuples) - one shape after normalisation,
    # and the ordering on each side must not matter.
    recorded = {"kokoro": ["af_aoede", "am_santa"],
                "piper": [["en_GB-alan-medium", None]]}
    requested = {"kokoro": ["am_santa", "af_aoede"],
                 "piper": [("en_GB-alan-medium", None)]}
    assert man.matches_requested(_manifest_with_voices(recorded),
                                 {"voices": requested}) == []


def test_matches_requested_voice_added_to_catalog_is_refused():
    # The live catalog grew since the build: the request carries a voice the
    # corpus was not built with, so reusing it would skip that voice's clips.
    recorded = {"kokoro": ["af_aoede", "am_santa"], "piper": []}
    diffs = man.matches_requested(_manifest_with_voices(recorded),
                                  {"voices": {"kokoro": ["af_aoede", "am_santa", "bf_new"],
                                              "piper": []}})
    assert len(diffs) == 1
    assert "voices.kokoro" in diffs[0]
    assert "3" in diffs[0] and "2" in diffs[0]     # requested vs recorded counts
    assert "bf_new" in diffs[0]                    # the voice that moved, named


def test_matches_requested_holdout_introduced_after_build_is_refused():
    # THE incident shape: the corpus was built from N voices, the tracked
    # holdout reserved some of them afterwards, and every shaping flag is
    # unchanged. The request is now smaller than the recorded set - refuse.
    recorded = {"kokoro": ["af_a", "af_b", "am_c"], "piper": []}
    diffs = man.matches_requested(_manifest_with_voices(recorded),
                                  {"voices": {"kokoro": ["af_a"], "piper": []}})
    assert len(diffs) == 1
    assert "af_b" in diffs[0] and "am_c" in diffs[0]
    assert "1" in diffs[0] and "3" in diffs[0]


def test_matches_requested_piper_pair_change_is_refused():
    # Piper entries are (voice, speaker) pairs: a swap of one pair is a
    # different corpus even though the counts agree.
    recorded = {"kokoro": ["af_a"], "piper": [["en_GB-alan-medium", None]]}
    diffs = man.matches_requested(_manifest_with_voices(recorded),
                                  {"voices": {"kokoro": ["af_a"],
                                              "piper": [("en_US-lessac-medium", None)]}})
    assert len(diffs) == 1
    assert "en_US-lessac-medium" in diffs[0] and "en_GB-alan-medium" in diffs[0]


def test_matches_requested_manifest_without_voices_field_is_refused():
    # A pre-feature manifest records no voice set at all: that axis cannot be
    # verified, so it must be reported, not silently skipped (the missing-key
    # contract the shaping flags already have).
    diffs = man.matches_requested(_manifest(),
                                  {"voices": {"kokoro": ["af_a"], "piper": []}})
    assert len(diffs) == 1
    assert "no voices field" in diffs[0]
    assert "cannot be reused" in diffs[0]


# ---------------------------------------------------------------------------
# corpus_identity
# ---------------------------------------------------------------------------

def test_corpus_identity_none_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        assert man.corpus_identity(tmp) is None


def test_corpus_identity_is_sha256_of_file_bytes_and_stable():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        body = json.dumps({"wake_word": WW}).encode("utf-8")
        (d / man.MANIFEST_NAME).write_bytes(body)
        expected = hashlib.sha256(body).hexdigest()[:man.SHORT]
        assert man.corpus_identity(d) == expected
        assert man.corpus_identity(d) == expected          # stable
        (d / man.MANIFEST_NAME).write_bytes(body + b"x")   # moved bytes -> moved id
        assert man.corpus_identity(d) != expected


# ---------------------------------------------------------------------------
# write_manifest -> matches_requested round trip
# ---------------------------------------------------------------------------

def test_write_then_match_round_trip(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "positives").mkdir()
        (d / "positives" / "a.wav").write_bytes(b"16kmono")
        man.write_manifest(d, WW, "mww", seed=42,
                           shaping={"samples_per_voice": 10})
        m = man.load_manifest(d)
        assert m is not None
        # A request matching the recorded build is clean...
        assert man.matches_requested(m,
                                     {"samples_per_voice": 10, "seed": 42}) == []
        # ...and the one that differs is refused with the value named.
        diffs = man.matches_requested(m, {"samples_per_voice": 12})
        assert len(diffs) == 1 and "12" in diffs[0] and "10" in diffs[0]
        # The content digest covered the wav tree.
        assert m["content_digest"]["files"] == 1
        assert m["content_digest"]["bytes"] == len(b"16kmono")


def test_write_then_match_voices_round_trip():
    # write_manifest must record the post-exclusion set in a shape that
    # matches_requested accepts again: piper tuples in, json lists out, and
    # the comparison must still be a match on the way back.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "positives").mkdir()
        (d / "positives" / "a.wav").write_bytes(b"16kmono")
        man.write_manifest(d, WW, "mww", seed=1,
                           shaping={"samples_per_voice": 10},
                           voices={"kokoro": ["af_a"],
                                   "piper": [("en_GB-alan-medium", None)]})
        m = man.load_manifest(d)
        assert m["voices"] == {"kokoro": ["af_a"],
                               "piper": [["en_GB-alan-medium", None]]}
        # The build's own set, as the trainer holds it (tuples), matches...
        assert man.matches_requested(m, {"voices": m["voices"]}) == []
        # ...and the one a later reservation shrank is refused, naming both
        # sides of the movement.
        diffs = man.matches_requested(m, {"voices": {"kokoro": [], "piper": []}})
        assert any("af_a" in line for line in diffs)
        assert any("en_GB-alan-medium" in line for line in diffs)


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
