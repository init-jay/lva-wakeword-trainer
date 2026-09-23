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
        data = (np.arange(SR // 2, dtype=np.int16) % 1000)
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


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
