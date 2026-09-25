"""Guards for src/train/provenance.py: the tag a training run is filed under.

A tag is CODE + DATA (+ CONFIG). The failures these guard:

* A copy (rsync) to the training box rewrites every mtime; a tag that
  moved because of a file copy would be worse than no tag (module docstring).
* A sweep point that shares its tag with the default run is one mww
  training refuses to run into ("model already exists in folder") - the
  config half must be sensitive to every setting the caller can change,
  seed included.
* The legacy `<commit>-d<...>` and targeted `<commit>-c<...>[-h<...>]`
  formats are contracts existing scripts parse; changing one silently
  breaks the archive naming in output/.

`code_tag` is stubbed out here: this box is a live git checkout, and the
tests want the documented non-git fallback path (a tag with no code half
of its own still names the data), not whatever commit HEAD happens to be.
"""

import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import train.provenance as prov  # noqa: E402


# ---------------------------------------------------------------------------
# digest_tree
# ---------------------------------------------------------------------------

def test_digest_tree_absent_dir_is_none_zero_zero():
    assert prov.digest_tree(REPO_ROOT / "no-such-dir" / "at-all") == (None, 0, 0)


def test_digest_tree_is_content_sensitive():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.wav").write_bytes(b"one")
        before, _, _ = prov.digest_tree(d)
        (d / "a.wav").write_bytes(b"two")
        after, _, _ = prov.digest_tree(d)
        assert before != after


def test_digest_tree_is_path_sensitive():
    # Same bytes, different speaker directory: which speaker a clip belongs
    # to is training data, so moving it must change the digest (docstring).
    with tempfile.TemporaryDirectory() as tmp:
        d1, d2 = Path(tmp) / "a", Path(tmp) / "b"
        (d1 / "sp1").mkdir(parents=True)
        (d2 / "sp2").mkdir(parents=True)
        (d1 / "sp1" / "clip.wav").write_bytes(b"data")
        (d2 / "sp2" / "clip.wav").write_bytes(b"data")
        h1, _, _ = prov.digest_tree(d1)
        h2, _, _ = prov.digest_tree(d2)
        assert h1 != h2


def test_digest_tree_is_mtime_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "a.wav"
        f.write_bytes(b"data")
        before, _, _ = prov.digest_tree(tmp)
        ts = os.times()
        os.utime(f, (ts[0] - 86400, ts[1] - 86400))
        after, _, _ = prov.digest_tree(tmp)
        assert before == after


def test_digest_tree_counts_only_audio_files():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.wav").write_bytes(b"ab")
        (d / "b.WAV").write_bytes(b"cde")
        (d / "notes.txt").write_bytes(b"not audio, must not move the digest")
        (d / "empty_dir").mkdir()
        hexd, count, total = prov.digest_tree(d)
        assert hexd is not None
        assert (count, total) == (2, 5)
        # Removing the non-audio file changes nothing.
        (d / "notes.txt").unlink()
        hexd2, count2, total2 = prov.digest_tree(d)
        assert (hexd2, count2, total2) == (hexd, count, total)


# ---------------------------------------------------------------------------
# config_tag
# ---------------------------------------------------------------------------

def test_config_tag_is_key_order_invariant():
    # json.dumps(sort_keys=True) is the mechanism; a dict the caller built
    # in a different order is the same resolved config.
    assert (prov.config_tag({"lr": 0.001, "seed": 42, "steps": 1000})
            == prov.config_tag({"seed": 42, "steps": 1000, "lr": 0.001}))


def test_config_tag_is_seed_sensitive():
    # Two runs differing only in seed produce different audio and must not
    # share a tag (config_tag docstring).
    a = prov.config_tag({"seed": 1, "lr": 0.001})
    b = prov.config_tag({"seed": 2, "lr": 0.001})
    assert a != b


def test_config_tag_differs_between_different_configs_and_is_short7():
    a = prov.config_tag({"lr": 0.001, "batch": 128})
    b = prov.config_tag({"lr": 0.002, "batch": 128})
    assert a != b
    assert re.fullmatch(r"[0-9a-f]{7}", a)


# ---------------------------------------------------------------------------
# run_tag formats
# ---------------------------------------------------------------------------

# A made-up wake word, deliberately: its corpus and recordings directories
# do not exist on this machine, so the tag shape the tests assert is
# independent of whatever corpora happen to be on disk.
WW = "zz test ww"


def _run_tag(**kwargs):
    original = prov.code_tag
    prov.code_tag = lambda: None       # force the documented non-git path
    try:
        return prov.run_tag(WW, **kwargs)
    finally:
        prov.code_tag = original


def test_run_tag_legacy_format_without_target():
    tag = _run_tag(fallback="f00")
    assert re.fullmatch(r"f00-d[0-9a-f]{7}", tag)


def test_run_tag_targeted_format_c_half():
    # No corpus for the made-up word: corpus_tag names that "absent", and
    # the format contract is the c- prefix, whatever fills it.
    tag = _run_tag(target="mww", fallback="f00")
    assert tag.startswith("f00-c")
    assert tag == "f00-cabsent"


def test_run_tag_targeted_with_config_gets_h_half():
    tag = _run_tag(target="mww", config={"seed": 42}, fallback="f00")
    h = prov.config_tag({"seed": 42})
    assert tag == f"f00-cabsent-h{h}"
    # And the h half moves with the seed, for the sweep-point reason above.
    other = _run_tag(target="mww", config={"seed": 43}, fallback="f00")
    assert tag != other


def test_run_tag_fallback_is_used_only_when_git_is_unavailable():
    # With the real code_tag restored, the code half is the git short hash,
    # never the caller's fallback.
    tag = prov.run_tag(WW, fallback="f00")
    assert not tag.startswith("f00-")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
