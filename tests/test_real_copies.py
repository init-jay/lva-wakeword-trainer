"""Guards for train/corpus/real.py's per-speaker copy weight
(--real-copies-override, added 2026-09-23 for the clean-detection work: the
seed-55/56 oww models measured ryan at 1/6 holdout and 36% on his own
training clips while jay measured 80% - the global 10x weight was the lever,
aimed per speaker).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.corpus.real import copy_real_samples  # noqa: E402

SR = 16000


def _make_tree(root, clips_per_speaker):
    for speaker, n in clips_per_speaker.items():
        d = Path(root) / speaker
        d.mkdir(parents=True)
        data = (20000 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)
                ).astype(np.int16)
        for i in range(n):
            scipy.io.wavfile.write(str(d / f"hey_seeree_{i:04d}.wav"), SR, data)


def test_per_speaker_override_changes_only_that_speakers_copies():
    # ryan=3 with base 1: ryan gets 3 copies per clip, jay stays at 1.
    # Destinations are real_{i}_{speaker}_... so the counts are checkable.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=1, per_speaker_copies={"ryan": 3})
        ryan = sum(1 for p in out.glob("*.wav") if "_ryan_" in p.name)
        jay = sum(1 for p in out.glob("*.wav") if "_jay_" in p.name)
        assert ryan == 2 * 3, f"ryan clips should carry 3 copies each, got {ryan}"
        assert jay == 1 * 1, f"jay keeps the base weight, got {jay}"
        assert count == ryan + jay


def test_override_none_is_the_unchanged_default_path():
    # per_speaker_copies=None must be byte-equivalent behaviour to the old
    # signature: every speaker at the base weight.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=2)
        ryan = sum(1 for p in out.glob("*.wav") if "_ryan_" in p.name)
        jay = sum(1 for p in out.glob("*.wav") if "_jay_" in p.name)
        assert (ryan, jay) == (4, 2)
        assert count == 6


def test_vtlp_copies_add_variants_not_raw_duplicates():
    # per_speaker_vtlp={"ryan": 3}: ryan gets 3 real_v*_<ratio>_ files per clip
    # (names carry the ratio, so duplicates are impossible by construction) and
    # jay is untouched. Content must actually differ from the source - a
    # "shifted" copy equal to the raw clip means the shift was inert.
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_tree(src, {"ryan": 2, "jay": 1})
        count = copy_real_samples(src, out, copies=1, per_speaker_vtlp={"ryan": 3})
        v = sorted(p for p in out.glob("*.wav") if p.name.startswith("real_v"))
        assert len(v) == 2 * 3, f"expected 6 shifted ryan clips, got {len(v)}"
        assert all("_ryan_" in p.name for p in v)
        # ratios land in the child-lever range and differ across copies
        ratios = {float(p.name.split("_")[2]) for p in v}
        assert all(1.15 <= r <= 1.30 for r in ratios), ratios
        raw = (out / "real_0_ryan_hey_seeree_0000.wav").read_bytes()
        assert any(p.read_bytes() != raw for p in v), \
            "shifted copies are byte-identical to the raw clip - inert"
        # NOT near-silence: the dtype bug (float into the int16 contract) produces
        # files that differ from the raw clip and are digitally silent.
        for p in v:
            sr_x, x = scipy.io.wavfile.read(p)
            assert int(np.abs(x.astype(np.int64)).max()) > 1000, \
                f"{p.name} is near-silent - shift dtype contract broken"
        assert count == 2 + 6 + 1  # ryan raw + ryan shifted + jay raw


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
