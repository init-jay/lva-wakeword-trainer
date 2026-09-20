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


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
