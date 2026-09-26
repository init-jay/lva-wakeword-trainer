#!/usr/bin/env python3
"""Is each real speaker's pitch range covered by each corpus's TTS voices?

WHY THIS EXISTS. A voice that the model does not fire on even in its own training
clips is a coverage failure, not a threshold failure: no cutoff choice rescues a
pitch range the corpus never renders, and the TTS mix is exactly where the
corpora differ. So the question to answer BEFORE spending a training run is
whether each corpus ever renders a voice in each real speaker's pitch range.

WHAT IT MEASURES. Median F0 per clip by autocorrelation (the same estimator
measure_voice_f0.py uses), then, per speaker, where that speaker sits inside the
corpus' TTS F0 distribution and what fraction of the corpus lies within +/-2 semitones
of the speaker's median. F0 is a proxy for timbre, not timbre itself: a covered pitch
range does not prove the voice is learnable, it only removes the cheapest explanation.

    python3 src/scripts/f0_coverage.py                     # every corpus, all speakers
    python3 src/scripts/f0_coverage.py --clips 200

Names: piper/kokoro clips carry the voice in the filename (piper_pf_en_GB-aru-medium_03_<uuid>.wav,
vtlp1.20_piper_...), and the real clips carry the speaker (real_N_<speaker>_<file>.wav on
both, mWW also keeps them under the flat name). Anything else is treated as unlabelled
TTS and contributes to the corpus distribution only.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
from measure_voice_f0 import estimate_f0  # noqa: E402

# The targets the pipeline produces - the same pair src/train/provenance.py:TARGETS names,
# spelled out here so this tool stays importable without the repo root on sys.path.
# Ad-hoc copies a session leaves beside them (a probe corpus, a pre-fix backup) are not
# corpora to profile by default; --corpus still reaches any of them explicitly.
_TARGETS = ("oww", "mww")


def clip_f0(path):
    """Median F0 over voiced frames, or None."""
    try:
        sr, data = scipy.io.wavfile.read(path)
    except Exception:
        return None
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.int16)
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
        data = np.clip(data, -32768, 32767).astype(np.int16)
    return estimate_f0(data.astype(np.float64))


def real_speaker(name, known):
    """real_7_spk_word_0012.wav -> spk; real_v3_1.22_spk_... -> spk too.

    The copy index sits between 'real' and the flattened relative path, and a VTLP
    variant inserts 'v<i>' and the ratio before it - so the speaker is whichever token
    is a known speaker directory rather than a fixed offset.
    """
    parts = Path(name).stem.split("_")
    if not parts or parts[0] != "real":
        return None
    for tok in parts[1:]:
        if tok in known:
            return tok
    return None


def describe(label, values, ref=None):
    v = np.asarray(sorted(x for x in values if x), dtype=np.float64)
    if not v.size:
        print(f"  {label:<40} no voiced frames")
        return
    q = np.percentile(v, [10, 50, 90])
    extra = ""
    if ref is not None:
        within = ((v >= ref / 2 ** (2 / 12)) & (v <= ref * 2 ** (2 / 12))).mean()
        pct = (v < ref).mean()
        extra = (f"   | within 2 st of ref: {within:>5.1%}   "
                 f"percentile of ref: {pct:>5.0%}")
    print(f"  {label:<40} n={v.size:<6} p10 {q[0]:5.0f} median {q[1]:5.0f} p90 {q[2]:5.0f} Hz{extra}")


def discover_corpora():
    """Every corpus dir under data/corpus/: one subdirectory per wake word, each
    holding the per-target corpora. Plain files and the shared 'eval' subtree
    are skipped; a subdirectory without a positives dir is dropped by the caller."""
    root = REPO_ROOT / "data" / "corpus"
    if not root.is_dir():
        return []
    out = []
    for word in sorted(root.iterdir()):
        if not word.is_dir() or word.name == "eval":
            continue
        out.extend(p for p in sorted(word.iterdir())
                   if p.is_dir() and p.name in _TARGETS)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", type=int, default=300, help="clips sampled per corpus")
    ap.add_argument("--speaker-clips", type=int, default=40, help="clips per real speaker")
    ap.add_argument("--corpus", action="append", default=[],
                    help="corpus dir to profile (default: every word under data/corpus/)")
    args = ap.parse_args()
    rng = random.Random(0)

    rec = REPO_ROOT / "data" / "recordings"
    print("REAL SPEAKERS (recordings, not the corpus)")
    speakers, known = {}, set()
    for sp in sorted(p for p in (rec / "samples").iterdir() if p.is_dir()):
        files = sorted(sp.rglob("*.wav"))
        vals = [clip_f0(f) for f in rng.sample(files, min(args.speaker_clips, len(files)))]
        known.add(sp.name)
        speakers[sp.name] = float(np.median([v for v in vals if v] or [0]))
        describe(sp.name, vals)
    print()

    corpora = [Path(REPO_ROOT / c) for c in args.corpus] or discover_corpora()
    if not corpora:
        sys.exit("no corpus dirs found under data/corpus/ - build a corpus first, "
                 "or pass --corpus explicitly")
    for corpus in corpora:
        pos = next((corpus / d for d in ("positive_train", "positives")
                    if (corpus / d).is_dir()), None)
        if pos is None:
            continue
        files = list(pos.rglob("*.wav"))
        sample = []
        for f in rng.sample(files, min(args.clips * 4, len(files))):
            sample.append((f, clip_f0(f)))
        # The corpus' own real clips carry the speaker in their flattened name; the TTS
        # clips are the population being tested for coverage. vtlp-prefixed files are
        # the synthetic child lever and are counted as TTS (they ARE coverage), while a
        # 'real_v<i>_<ratio>_' file is a shifted copy of a person and is not.
        by_speaker = defaultdict(list)
        tts, vtlp_tts = [], []
        for f, v in sample:
            sp = real_speaker(f.name, known)
            if sp:
                by_speaker[sp].append(v)
            elif v:
                (vtlp_tts if f.name.startswith("vtlp") else tts).append(v)
        print(f"CORPUS {corpus.relative_to(REPO_ROOT)}  ({len(files)} positive clips sampled "
              f"{len(sample)}: {len(tts)} TTS, {len(vtlp_tts)} synthetic-shifted, "
              f"{sum(len(v) for v in by_speaker.values())} real)")
        describe("TTS, unshifted", tts)
        describe("TTS, vtlp-shifted", vtlp_tts)
        pooled = np.asarray([x for x in tts + vtlp_tts if x], dtype=np.float64)
        unsh = np.asarray([x for x in tts if x], dtype=np.float64)
        # The coverage question, against each speaker's OWN recorded median: how many
        # of this corpus' TTS clips sit within +/-2 semitones of it. Two semitones is
        # deliberately generous - F0 is a proxy for timbre, not timbre itself.
        print("  TTS coverage of each speaker's pitch band (+/- 2 semitones):")
        for s, ref in sorted(speakers.items()):
            band = ((pooled >= ref / 2 ** (2 / 12)) & (pooled <= ref * 2 ** (2 / 12)))
            band_u = ((unsh >= ref / 2 ** (2 / 12)) & (unsh <= ref * 2 ** (2 / 12)))
            n_real = len(by_speaker.get(s, []))
            print(f"    {s:<6} median {ref:5.0f} Hz -> {int(band.sum()):>4}/{pooled.size}"
                  f" ({band.mean():>5.1%}) of all TTS, {int(band_u.sum()):>4}/{unsh.size}"
                  f" ({band_u.mean():>5.1%}) unshifted     real clips in corpus: {n_real}"
                  f"{'   <- thin' if band.mean() < 0.06 else ''}")
        print()


if __name__ == "__main__":
    main()
