"""Guards for src/train/mww/corpus.py's --real-vtlp (the microWakeWord port of the
per-speaker copy lever).

The oww side measured the split between dilution and diversity: raw
repetition buys a weak voice presence at the cost of diluting the voices
that were already detected, while shifted copies are NEW acoustic variants
of the same recording - diversity is not paid for in row count. The shifted
copies are what port to mww, because mww generates its spectrogram feature
rows UP FRONT
(src/train/mww/features.py): a shifted wav is a distinct row, a raw copy is only
another draw of the same voice (and one more slot in the per-file train/val/
test split - the NOTE at the copy call in corpus.py is why there is no
--real-copies-override here).

The contract, from the task it came out of:

* the parser fails loud on every silent no-op: bad syntax, non-positive N,
  and a speaker name with no directory under --real-samples (an inert
  override is the label/config drift class this repo has paid for twice);
* shifted copies land for the NAMED speaker only, carry their ratio in the
  filename (CHILD_STRETCH["m"], the synthetic child-lever's range), and are
  not digitally silent - the int16-in/int16-out contract of vocal_tract_shift,
  whose float-input trap produced near-silence in a holdout probe
  (src/train/corpus/real.py);
* the parsed setting is part of the corpus manifest shaping, so a different
  --real-vtlp refuses the --skip reuse path instead of training on a corpus
  shaped differently than requested.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import train.mww.corpus as mww_corpus  # noqa: E402
from train.corpus import manifest as corpus_manifest  # noqa: E402
from train.corpus.augment import CHILD_STRETCH, CHILD_STRETCH_FRACTION  # noqa: E402
from train.corpus.real import copy_real_samples  # noqa: E402
from wordlists import path_for  # noqa: E402

SR = 16000
WW = "zz test vtlp"   # made-up word: no on-disk corpus to entangle the tests with
SAFE = WW.replace(" ", "_").lower()


def _sine(path, freq=220.0, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    data = (20000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    scipy.io.wavfile.write(str(path), SR, data)
    return data


def _make_samples(tmp):
    """One clip per fixture speaker, under <tmp>/samples/<speaker>/."""
    samples = Path(tmp) / "samples"
    for speaker in ("ryan", "jay"):
        d = samples / speaker
        d.mkdir(parents=True)
        _sine(d / f"hey_seeree_0000.wav", freq=220.0 if speaker == "ryan" else 150.0)
    return samples


def _expect_refusal(spec, samples, needle):
    try:
        mww_corpus._parse_real_vtlp(spec, samples)
    except SystemExit as e:
        # A caught sys.exit(message) never prints the message; it rides on .code.
        msg = str(e.code)
        assert "ERROR" in msg, f"refusal without an ERROR line: {msg!r}"
        assert needle in msg, f"{needle!r} not in: {msg!r}"
        return msg
    raise AssertionError(f"parser accepted {spec!r} - {needle!r} should have refused")


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

def test_parser_accepts_valid_spec_and_empty():
    with tempfile.TemporaryDirectory() as tmp:
        samples = _make_samples(tmp)
        assert mww_corpus._parse_real_vtlp("ryan=3, jay=2", samples) == {
            "ryan": 3, "jay": 2}
        assert mww_corpus._parse_real_vtlp("", samples) == {}


def test_parser_refuses_bad_syntax():
    with tempfile.TemporaryDirectory() as tmp:
        samples = _make_samples(tmp)
        # ",ryan=3" is NOT refused - an empty part is a separator, the same
        # leniency the mirrored oww parser has. A part with no = is the error.
        for spec in ("ryan", "ryan 3"):
            _expect_refusal(spec, samples, "expected speaker=N")
        assert mww_corpus._parse_real_vtlp(",ryan=3,", samples) == {"ryan": 3}
        # "ryan=" has an = but an empty value: the positive-int check refuses it
        _expect_refusal("ryan=", samples, "positive int")
    # "ryan=3=4": the value side ("3=4") fails the positive-int check instead,
    # which is also a loud refusal - pin that it exits, not which line it takes.
    with tempfile.TemporaryDirectory() as tmp:
        samples = _make_samples(tmp)
        try:
            mww_corpus._parse_real_vtlp("ryan=3=4", samples)
            raise AssertionError("ryan=3=4 accepted")
        except SystemExit:
            pass


def test_parser_refuses_non_positive_count():
    with tempfile.TemporaryDirectory() as tmp:
        samples = _make_samples(tmp)
        for spec in ("ryan=0", "ryan=-1", "ryan=x"):
            _expect_refusal(spec, samples, "positive int")


def test_parser_refuses_unknown_speaker():
    # The failure mode the check exists for: a typo'd name never matches
    # inside copy_real_samples, so without this exit the corpus would be
    # filed under a shaping that claims shifted variants it does not carry.
    with tempfile.TemporaryDirectory() as tmp:
        samples = _make_samples(tmp)
        msg = _expect_refusal("bob=3", samples, "bob")
        # Names the tree it actually checked, so the operator can fix the typo.
        assert "ryan" in msg and "jay" in msg, msg
    # An ABSENT samples tree still accepts (nothing to validate against -
    # copy_real_samples would find no recordings anyway, so no drift).
    with tempfile.TemporaryDirectory() as tmp:
        assert mww_corpus._parse_real_vtlp("ryan=3", Path(tmp) / "nope") == {"ryan": 3}


# ---------------------------------------------------------------------------
# the copies themselves (src/train/corpus/real.py, consumed by the mww corpus)
# ---------------------------------------------------------------------------

def test_vtlp_copies_written_for_named_speaker_only():
    # per_speaker_vtlp={"ryan": 3} at base weight 1: ryan gets 3 shifted
    # copies per clip, jay none. Named real_v{i}_<ratio>_ so they sort apart
    # from the raw copies and never collide (the ratio is in the name).
    with tempfile.TemporaryDirectory() as tmp:
        samples, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_samples(tmp)
        count = copy_real_samples(samples, out, copies=1,
                                  per_speaker_vtlp={"ryan": 3})
        ryan_v = sorted(p for p in out.glob("*.wav")
                        if p.name.startswith("real_v") and "_ryan_" in p.name)
        jay_v = [p for p in out.glob("*.wav")
                 if p.name.startswith("real_v") and "_jay_" in p.name]
        assert len(ryan_v) == 3, f"1 ryan clip at N=3 -> 3 shifted copies, got {len(ryan_v)}"
        assert jay_v == [], f"jay has no override and no shifted copies: {jay_v}"
        raw = [p for p in out.glob("*.wav") if not p.name.startswith("real_v")]
        assert len(raw) == 2  # one raw copy per clip, base weight 1
        assert count == 5


def test_vtlp_copies_non_silence_ratio_and_dtype():
    # The int16-in/int16-out contract of vocal_tract_shift: a float input
    # peak-normalises against 32767 and returns digital silence - the dtype
    # bug a holdout probe paid for (src/train/corpus/real.py). Every
    # shifted copy must be a real int16 signal, its ratio in the
    # CHILD_STRETCH["m"] range it drew from, and roughly the source's
    # duration (the shift preserves delivery speed).
    lo, hi = CHILD_STRETCH["m"]
    with tempfile.TemporaryDirectory() as tmp:
        samples, out = Path(tmp) / "samples", Path(tmp) / "out"
        _make_samples(tmp)
        src_len = int(SR)  # 1.0 s clips
        copy_real_samples(samples, out, copies=0,
                          per_speaker_vtlp={"ryan": 4, "jay": 2})
        v = sorted(p for p in out.glob("*.wav") if p.name.startswith("real_v"))
        assert len(v) == 6  # 4 ryan + 2 jay
        for p in v:
            sr_x, x = scipy.io.wavfile.read(p)
            assert sr_x == SR and x.dtype == np.int16, p.name
            peak = int(np.abs(x.astype(np.int64)).max())
            assert peak > 1000, f"{p.name} is near-silent (peak {peak}) - " \
                                f"vocal_tract_shift dtype contract broken"
            # ratio lands in the drawn range (it is in the filename)
            ratio = float(p.name.split("_")[2])
            assert lo <= ratio <= hi, (p.name, ratio, (lo, hi))
            # duration preserved: resample shortens, WSOLA stretch restores
            assert abs(len(x) - src_len) <= 0.2 * src_len, (p.name, len(x), src_len)
        # different draws: not every copy at the same ratio
        ratios = {float(p.name.split("_")[2]) for p in v}
        assert len(ratios) > 1, f"all 6 copies drew the same ratio: {ratios}"


# ---------------------------------------------------------------------------
# the manifest: the setting is corpus identity
# ---------------------------------------------------------------------------

def _mww_manifest(tmp, real_vtlp):
    """A finished-build mww corpus under <tmp>/corpus with a real manifest."""
    root = Path(tmp) / "corpus" / SAFE / "mww"
    for sub in ("positives", "negatives"):
        d = root / sub
        d.mkdir(parents=True)
        _sine(d / "clip_0000.wav")
    corpus_manifest.write_manifest(
        root, WW, "mww", seed=42,
        shaping={
            "samples_per_voice": 60,
            "negatives_per_voice": 12,
            "kokoro_fraction": 0.0,
            "child_fraction": CHILD_STRETCH_FRACTION,
            "real_copies": 1,
            "real_vtlp": real_vtlp,
            # A corpus manifest written before this key existed is refused for
            # reuse by design (src/train/corpus/manifest.py is strict, one variable at
            # a time), so a fixture that wants the reuse path must carry it.
            "balance_real_copies": "",
            "piper_speakers": 12,
            "piper_languages": "en_US,en_GB",
            "negatives_file": None,
            "no_trim": False,
        },
        engines={"piper": {"url": "tcp://127.0.0.1:8898", "version": None}},
        voices={"kokoro": [], "piper": [["en_US-lessac-medium", "S1"]]},
        per_voice_counts={"positives": 1, "negatives": 1},
        wordlist_path=path_for(WW), wall_time_s=1.0)
    return root


def test_manifest_records_real_vtlp_and_diffs_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = _mww_manifest(tmp, {"ryan": 3})
        man = corpus_manifest.load_manifest(root)
        assert man["shaping"]["real_vtlp"] == {"ryan": 3}
        base = {
            "samples_per_voice": 60, "negatives_per_voice": 12,
            "kokoro_fraction": 0.0, "child_fraction": 0.5, "real_copies": 1,
            "piper_speakers": 12, "piper_languages": "en_US,en_GB",
            "negatives_file": None, "no_trim": False,
            "voices": {"kokoro": [], "piper": [["en_US-lessac-medium", "S1"]]},
        }
        assert corpus_manifest.matches_requested(man, {**base, "real_vtlp": {"ryan": 3}}) == []
        # a different value (and the no-flag value) must diff
        diffs = corpus_manifest.matches_requested(man, {**base, "real_vtlp": {"ryan": 5}})
        assert len(diffs) == 1 and "real_vtlp" in diffs[0] and "5" in diffs[0], diffs
        diffs = corpus_manifest.matches_requested(man, {**base, "real_vtlp": {}})
        assert len(diffs) == 1 and "real_vtlp" in diffs[0], diffs


def _run_main(tmp, argv):
    """corpus.main() with the TTS probes stubbed: --skip needs the catalog to
    resolve but never renders, and the probes are exactly the seam to patch."""
    samples = Path(tmp) / "samples"
    (samples / "ryan").mkdir(parents=True, exist_ok=True)
    (samples / "jay").mkdir(parents=True, exist_ok=True)
    root = Path(tmp) / "corpus" / SAFE / "mww"
    for sub in ("positives", "negatives"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv
    old_probes = (mww_corpus.select_piper_voices, mww_corpus.load_voice_holdout)
    sys.argv = ["train.mww.corpus"] + argv
    mww_corpus.select_piper_voices = lambda *a, **k: [("en_US-lessac-medium", "S1")]
    mww_corpus.load_voice_holdout = lambda *a, **k: {}
    buf = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                mww_corpus.main()
            except SystemExit as e:
                # sys.exit(message) carries the message on .code and never
                # prints it when caught; sys.exit(int) prints nothing either.
                code = e.code
    finally:
        sys.argv = old_argv
        (mww_corpus.select_piper_voices,
         mww_corpus.load_voice_holdout) = old_probes
    return code, buf.getvalue()


def test_skip_reuse_refuses_a_different_real_vtlp():
    # The whole point: a corpus built WITH shifted ryan variants refuses a
    # --skip reuse that does not request them, and the inverse - a request
    # that names variants the corpus does not carry is refused the same way.
    # No TTS: --skip never renders (probes stubbed above).
    with tempfile.TemporaryDirectory() as tmp:
        root = _mww_manifest(tmp, {"ryan": 3})
        argv = [
            "--wake-word", WW,
            "--corpus-root", str(Path(tmp) / "corpus"),
            "--real-samples", str(Path(tmp) / "samples"),
            "--piper-url", "tcp://127.0.0.1:8898",
            "--skip",
        ]
        # corpus on disk says ryan=3; the request says nothing -> refuse
        code, out = _run_main(tmp, argv)
        assert code == 1, f"expected refusal, got code {code!r}:\n{out}"
        assert "real_vtlp" in out and "refusing" in out, out
        # the request names the variants the corpus does NOT have -> refuse
        code, out = _run_main(tmp, argv + ["--real-vtlp", "jay=2"])
        assert code == 1 and "real_vtlp" in out, out
        # the matching request reuses
        code, out = _run_main(tmp, argv + ["--real-vtlp", "ryan=3"])
        assert code is None, f"matching request should reuse, got code {code!r}:\n{out}"
        assert "reusing" in out, out
        # a bogus speaker name fails loud even in --skip mode: the message
        # rides on the (never-printed) SystemExit code, not the captured streams
        code, out = _run_main(tmp, argv + ["--real-vtlp", "bob=3"])
        assert code is not None and "bob" in str(code), (code, out)
        assert root.is_dir()


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()
