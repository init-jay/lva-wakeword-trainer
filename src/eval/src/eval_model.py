#!/usr/bin/env python3
"""
Score a trained wake-word model against the four gates below.

Everything here is measured by streaming - `Model.predict_clip` slides the model
over the clip 80 ms at a time, exactly as live detection does - because that is
what the gates are about. `src/eval/src/check_model_alignment.py` answers a different question
(where in the window the model wants the phrase) by placing clips at fixed offsets;
a clip that misses at one offset may well fire at the next one in streaming, so the
two scripts are not interchangeable.

Gates:

    extend + hey_other false accepts at 0.5    < 2/32
    clean positive detection at 0.5            >= 97%
    the same, for the WEAKEST speaker          >= 97%
    detection with a command immediately after >= 27/30
    median latency from end of speech          < 120 ms

The weakest-speaker gate is not one of the original four. It is here because everything
else on that list is an average over speakers, and an average is what let a 4-year-old
sit at 24% detection behind a 97% adult for long enough to need src/train/corpus/augment.py.

Two details of the method matter enough to state:

* Clips are padded with a low noise floor, never digital silence. Pure zeros are a
  pathological input to the melspectrogram and shift scores, so `predict_clip`'s own
  zero padding is bypassed (`padding=0`).
* "End of speech" is the last sample above 2% of peak amplitude. Latency is the audio
  offset where the score first crosses the threshold, minus that marker.

The --json output also records a THRESHOLD SWEEP over a fixed 0.05-0.95 grid: a
re-threshold of the per-clip peaks already computed in this one run, so the run
ledger (src/train/ledger.py) can compare models at a matched false-accept budget
instead of a single threshold - 'Never compare models at a fixed threshold'
(CLAUDE.md); bug.md C3 step 2 (2026-09-22), the 40-eval manual 0.25-0.85 job.

Usage, from the repo root:

Negatives are reported PER CATEGORY, never pooled. The corpus from
generate_negatives.py is adversarial by construction - a fifth of it is
phrase-extending - so a pooled false-accept rate means nothing. Category comes from
the filename prefix (`extend_000_af_bella.wav` -> `extend`).

BOTH TRAINERS ARE SCORED THROUGH THE SAME CODE. `src/eval/src/backends.py` picks an
openWakeWord or a microWakeWord backend by inspecting the model, so everything below
is arithmetic over scores. Two consequences worth stating rather than discovering:

* The gates were calibrated on openWakeWord and are reported for either, but a
  microWakeWord score is a sliding-window average with different threshold
  semantics - `--sliding-window-size` is printed with every result for that reason.
  Keep mWW numbers apart from the oww ones.
* Latency is measured the same way for both and is the one number that transfers
  directly: it is the deployed quantity either way.

POSITIVES DEFAULT TO THE HELD-OUT RECORDINGS, not to everything recorded. The trainer
globs data/recordings/samples/ recursively, so scoring these gates against that tree
measures memorisation; `src/eval/src/paths.py` carries the split and warns if a run is pointed
back inside it. The `_runon` directories are excluded here on purpose - this file
builds its own command-following case by concatenating a command onto a plain clip,
so a real run-on recording among the positives would be scored as the phrase alone.

Usage, from the repo root:
    python -m eval.eval_model --model output/hey_seeree/oww/hey_seeree_705c23b.onnx   # the eval image
    src/eval/.venv/bin/python src/eval/src/eval_model.py --model output/hey_seeree/oww/hey_seeree_705c23b.onnx   # the host env
    python -m eval.eval_model --model M --positives data/recordings/holdout/speaker1
    python -m eval.eval_model --model M --threshold 0.7 --verbose

Needs onnxruntime and an importable openwakeword for .onnx models, plus a TFLite
runtime and pymicro-features for microWakeWord ones. The `eval` compose service has
all of it and runs on the Mac:

    docker compose run --rm eval python -m eval.eval_model --model M
"""

import argparse
import json
import math
import re
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io.wavfile

# Runnable as `python src/eval/src/eval_model.py` as well as `python -m
# eval.eval_model`. The module form is the eval image's: src/ is mounted as the
# `eval` package, so the package is importable. The plain-path form - the host
# invocation, src/scripts/setup-eval-host.sh - only has this directory on
# sys.path, so try both. The same guard the other scripts here carry.
try:
    from eval import backends, paths
except ImportError:
    import backends, paths

SR = 16000
NOISE_FLOOR = 30.0          # std dev in 16-bit counts; stands in for room tone
PAD_S = 1.0                 # noise before and after each clip
SPEECH_END_FRAC = 0.02      # "end of speech" = last sample above 2% of peak

# Gates, as rates so they survive a different corpus size.
GATE_FALSE_ACCEPT = 2 / 32
# 97%, not 98%. A gate is a rate rather than a count so that it survives a different
# corpus size - but a zero-misses bar does not survive a small holdout: with tens of
# clips per speaker it is arithmetically out of reach, and "does this speaker pass"
# degrades into "did they miss literally nothing today". 97% admits exactly one miss
# for the larger speaker sets while staying effectively zero-miss for the small ones -
# the asymmetry is honest: those sets need more recordings, not a softer bar.
GATE_POSITIVE = 0.97
GATE_COMMAND = 27 / 30
GATE_LATENCY_MS = 120

# Categories that carry the signal; `general` is the realistic background rate.
ADVERSARIAL = ("extend", "hey_other")

# generate_negatives.py names files "{category}_{index:03d}_{voice}.wav", and two
# category names contain an underscore themselves (hey_other, other_ww). Splitting
# on the first underscore silently renames those to "hey" and "other", which drops
# hey_other out of the adversarial gate entirely.
CATEGORY_RE = re.compile(r"^(.+?)_\d{3}_")


def read_wav(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        return None
    return data.astype(np.int16)


def wilson_interval(k, n, z=1.959964):
    """95% Wilson interval on a proportion k/n, as (low, high). None when n=0.

    The Wilson interval is the right one here because the holdout is TINY: ryan is
    n=6, so one clip is 16.7 points, and jen n=10, one clip is 10 points. A bare
    rate on that n swings 16.7 points per clip, which is exactly the scale of the
    10-point run-to-run variance this repo has measured at an identical config - so
    without the interval a tuning loop will chase a 10-point 'win' that is one clip.
    Wilson rather than the normal approximation because it stays sane at small n
    and at 0/100 (the normal interval goes negative).
    """
    if n == 0:
        return None
    z2 = z * z
    p = k / n
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    # The interval is on a proportion, so it cannot leave [0, 1]; the formula can
    # overshoot by a rounding hair (which prints as "-0%"), so clamp it back.
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def speech_end(data):
    """Last sample above 2% of peak amplitude, or None if the clip is silent."""
    a = np.abs(data.astype(np.float64))
    peak = a.max()
    if peak <= 0:
        return None
    above = np.flatnonzero(a > SPEECH_END_FRAC * peak)
    return int(above[-1]) if above.size else None


def clip_rng(clip_name):
    """A generator seeded by the clip's name.

    One shared stream would make a clip's padding noise depend on how many clips
    came before it, so adding files to the corpus would silently change the result
    for every clip after them - 6/6 and 5/6 on identical audio. crc32 rather than
    hash() because hash() is salted per process.
    """
    return np.random.default_rng(zlib.crc32(clip_name.encode()))


def with_noise_floor(data, rng, pad_s=PAD_S):
    """Pad with room tone rather than zeros, and report where the clip starts."""
    pad = int(SR * pad_s)
    lead = rng.normal(0, NOISE_FLOOR, pad).astype(np.int16)
    tail = rng.normal(0, NOISE_FLOOR, pad).astype(np.int16)
    return np.concatenate([lead, data, tail]), pad


def stream(backend, audio):
    """Per-frame scores for one clip, plus the audio offset each frame reflects.

    The step differs by backend - 80 ms for openWakeWord, 30 ms for microWakeWord -
    which is why the offsets come back from the model rather than from a constant
    here. Everything downstream reads them and stays backend-agnostic.
    """
    return backend.score(audio)


def first_crossing(scores, offsets, threshold):
    hit = np.flatnonzero(scores >= threshold)
    return (offsets[hit[0]] if hit.size else None)


# The grid the sweep re-thresholds over: 0.05..0.95, 19 points (SWEEP_GRID).
# 0.5 is on it, so the sweep's own 0.5 point cross-checks the single-threshold
# numbers the report printed above it - the two cannot drift apart silently.
SWEEP_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def threshold_sweep(pos_peaks, adv_peaks, grid=SWEEP_GRID):
    """Re-threshold already-computed per-clip PEAK scores over a fixed grid.

    A pure filter, not a re-scoring: a clip fires at threshold t iff its peak
    is >= t, because both the positive and negative paths above decide on any
    frame crossing t (first_crossing / scores.max() >= t) and the peak is the
    max over frames. The model runs exactly once, in stream(); everything here
    is arithmetic on the peaks it left behind.

    The false-accept axis is the extend+hey_other subset (ADVERSARIAL), never
    pooled with the other categories: 'Never pool negative categories'
    (CLAUDE.md) applies to the curve exactly as it does to the gate - a pooled
    rate would let general's background dilute the extend signal. Both subsets
    (extend+hey_other and the full negative set) re-threshold identically
    because by_category holds a peak for every clip; the FA axis here is the
    adversarial one, matching the eval block's "adversarial" field.

    bug.md C3 step 2 (2026-09-22): the training-steps verdict cost 40
    hand-driven evals (one per 0.25-0.85 threshold) for a two-point sweep,
    because this harness took one --threshold and the ledger had no curve to
    compare on. This is what lets a sweep runner conclude itself.
    """
    pos = np.asarray(pos_peaks, dtype=np.float64)
    adv = np.asarray(adv_peaks, dtype=np.float64)
    thr = [float(t) for t in grid]
    return {
        "thresholds": thr,
        "positives_rate": [float((pos >= t).mean()) for t in thr],
        "adv_rate": [float((adv >= t).mean()) for t in thr],
        "pos_n": int(pos.size),
        "adv_n": int(adv.size),
    }


def load_dir(directory, recursive=True):
    globber = Path(directory).rglob if recursive else Path(directory).glob
    out, skipped = [], 0
    for wav in sorted(globber("*.wav")):
        data = read_wav(wav)
        if data is None:
            skipped += 1
            continue
        out.append((wav.name, data))
    return out, skipped


def load_by_speaker(directories, limit=None):
    """{speaker: clips}, one entry per directory, in the order given.

    THE SPEAKER TRAVELS BESIDE THE CLIPS, NOT INSIDE THEM. Folding it into the clip
    name instead - `speaker1/hey_seeree_0001.wav` - would be tidier and would silently
    invalidate every number this harness has ever produced: `clip_rng` derives each
    clip's padding noise from its name, so renaming the clips reseeds the noise and
    moves the scores. Hence a mapping alongside, and `wav.name` left alone.

    `limit` applies PER SPEAKER, which is what --limit has always claimed ("the first
    N clips of each set") and what keeps a limited run comparable across speakers.
    """
    by_speaker, skipped_total = {}, 0
    for directory in directories:
        clips, skipped = load_dir(directory)
        skipped_total += skipped
        if limit:
            clips = clips[:limit]
        if not clips:
            continue
        label = paths.speaker_label(directory)
        # Two directories can share a basename when they come from different trees.
        by_speaker[str(directory) if label in by_speaker else label] = clips
    return by_speaker, skipped_total


def spans_for(by_speaker):
    """[(speaker, start, end)] into the flat clip list `by_speaker` concatenates to.

    Scoring stays a single pass over one list - the per-speaker view is slicing done
    afterwards, so nothing is scored twice and the pooled and per-speaker numbers
    cannot disagree.
    """
    spans, offset = [], 0
    for speaker, clips in by_speaker.items():
        spans.append((speaker, offset, offset + len(clips)))
        offset += len(clips)
    return spans


def report_by_speaker(rows, spans):
    """Detection, score and latency per speaker.

    NOT behind a flag, and printed whenever there is more than one speaker. The
    pooled number above it is an average over speakers, and an average over speakers
    is precisely what hides the one who fails: the run that motivated the child-range
    shifting in src/train/corpus/augment.py measured a 4-year-old at 24% while the adult
    read 97%, and the pooled figure looked healthy throughout.
    """
    print(f"\n  {'speaker':<20}{'n':>4}{'detected':>12}{'median score':>14}"
          f"{'median latency':>16}")
    rates = []
    for speaker, start, end in spans:
        entries = rows[start:end]
        ok = sum(1 for e in entries if e[1])
        lats = [e[2] for e in entries if e[2] is not None]
        peaks = np.array([e[3] for e in entries])
        lat = f"{np.median(lats):.0f}ms" if lats else "-"
        # n= and the Wilson CI travel with the rate (wilson_interval for why).
        ci = wilson_interval(ok, len(entries))
        ci_text = (f" (n={len(entries)}, 95% CI {ci[0]:.0%}-{ci[1]:.0%})"
                   if ci else " (n=0)")
        print(f"  {speaker[:19]:<20}{len(entries):>4}{f'{ok}/{len(entries)}':>12}"
              f"{np.median(peaks):>14.3f}{lat:>16}  {ci_text}")
        rates.append((ok / len(entries), speaker, ok, len(entries), ci))

    if len(rates) > 1:
        worst, best = min(rates), max(rates)
        spread = best[0] - worst[0]
        print(f"\n  Weakest speaker: {worst[1]} at {worst[0]:.0%} "
              f"({worst[2]}/{worst[3]}), {spread * 100:.0f} points below "
              f"{best[1]} at {best[0]:.0%}.")
        # The threshold is a judgement, not a gate: what matters is that a large
        # spread is READ rather than averaged away. Real clips from the speaker who
        # fails beat any amount of augmentation - see the skill and augment.py.
        if spread >= 0.25:
            print("  That spread is the failure augmentation does not fix. The model "
                  "generalises\n  to some voices and not others; record more of "
                  f"{worst[1]} rather than retuning.")
    return rates


def evaluate_positives(backend, clips, threshold, rng, verbose):
    """Per-clip (name, detected, latency_ms or None, peak score)."""
    rows = []
    for clip_name, data in clips:
        audio, pad = with_noise_floor(data, clip_rng(clip_name))
        scores, offsets = stream(backend, audio)
        crossing = first_crossing(scores, offsets, threshold)
        latency = None
        if crossing is not None:
            end = speech_end(data)
            if end is not None:
                latency = (crossing - (pad + end)) / SR * 1000
        rows.append((clip_name, crossing is not None, latency, float(scores.max())))
        if verbose:
            lat = f"{latency:>6.0f}ms" if latency is not None else "      -"
            print(f"    {clip_name[:36]:<38}peak {scores.max():.3f}  latency {lat}")
    return rows


def group_key(clip_name):
    """Sweep and value from a generate_positives.py filename (speed_0.55_af_bella)."""
    parts = clip_name.rsplit(".", 1)[0].split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else "(ungrouped)"


def evaluate_with_command(backend, clips, commands, threshold, rng, gap_ms, verbose):
    """Detection when a command follows the phrase, with an optional pause between.

    The gap is the discriminating variable: 20/30 was measured with the command butted
    straight on and 28/30 with 300 ms of pause, which is the signature of a model that
    learned the phrase is followed by quiet.
    """
    detected, misses = 0, []
    for i, (clip_name, data) in enumerate(clips):
        command = commands[i % len(commands)][1]
        gap = np.zeros(int(SR * gap_ms / 1000), dtype=np.int16) if gap_ms else None
        parts = [data] + ([gap] if gap is not None else []) + [command]
        audio, _ = with_noise_floor(np.concatenate(parts), clip_rng(clip_name))
        scores, _ = stream(backend, audio)
        if scores.max() >= threshold:
            detected += 1
        else:
            misses.append((clip_name, float(scores.max())))
    if verbose and misses:
        for clip_name, peak in misses[:8]:
            print(f"    missed {clip_name[:36]:<38}peak {peak:.3f}")
    return detected, misses


def evaluate_negatives(backend, clips, threshold, rng):
    """Max score per clip, grouped by the category in the filename prefix."""
    by_category = defaultdict(list)
    for clip_name, data in clips:
        audio, _ = with_noise_floor(data, clip_rng(clip_name))
        scores, _ = stream(backend, audio)
        match = CATEGORY_RE.match(clip_name)
        category = match.group(1) if match else "(uncategorised)"
        by_category[category].append((clip_name, float(scores.max())))
    return by_category


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    parser = argparse.ArgumentParser(
        description="Score a wake-word model against the pipeline's gates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Trained .onnx or .tflite model")
    parser.add_argument("--positives", nargs="+", default=None,
                        help="Directories of positive clips, searched recursively "
                             "(default: the held-out speaker directories under "
                             "data/recordings/holdout/, excluding the _runon ones)")
    parser.add_argument("--negatives", default=str(paths.NEGATIVES_DIR),
                        help="Directory from generate_negatives.py (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--command-gap-ms", type=float, default=300,
                        help="Pause to test alongside the no-pause case (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="Only use the first N clips of each set")
    parser.add_argument("--by-group", action="store_true",
                        help="Break positives down by the sweep encoded in their "
                             "filename (speed_0.55_af_bella -> speed_0.55), as "
                             "generate_positives.py names them")
    parser.add_argument("--voice-holdout-set", default=None,
                        help="Directory of the voice-holdout synthetic ranking set "
                             "(generate_positives.py --voice-holdout, with its "
                             "set.json label). Scored as a SEPARATE, clearly "
                             "labelled block after the gates - a synthetic voice "
                             "is not a person, so it is a low-variance ranking "
                             "signal for sweep points, never merged into the "
                             "gates above, which stay on the real held-out "
                             "recordings (improvement.md P1.2)")
    parser.add_argument("--verbose", action="store_true", help="Print a row per positive")
    parser.add_argument("--sliding-window-size", type=int, default=None,
                        help="microWakeWord only: probabilities averaged before "
                             "thresholding, as the runtime does. A cutoff is only "
                             "meaningful alongside this. Default: whatever the "
                             "manifest says, so the manifest is under test too")
    parser.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                        help="Also write every number printed here as machine-readable "
                             "JSON to PATH (same values, for the run ledger - see "
                             "improvement.md P0.5). Does not change the report.")
    args = parser.parse_args()

    # The backend picks itself by inspecting the model, so the gates can be scored on
    # the artifact that actually ships rather than on the ONNX it came from - and on
    # a microWakeWord streaming tflite, which openwakeword.model.Model cannot load.
    backend = backends.load(args.model, sliding_window_size=args.sliding_window_size)
    rng = np.random.default_rng(0)

    positive_dirs = args.positives or [str(d) for d in paths.holdout_dirs(runon=False)]
    paths.warn_if_trained_on(positive_dirs)

    by_speaker, skipped_p = load_by_speaker(positive_dirs, limit=args.limit)
    positives = [clip for clips in by_speaker.values() for clip in clips]
    speaker_spans = spans_for(by_speaker)

    negatives, skipped_n = load_dir(args.negatives)
    if args.limit:
        negatives = negatives[:args.limit]
    if not positives:
        print(f"No usable WAV files in {paths.describe(positive_dirs)}. Held-out "
              f"clips are recorded with `record_samples.py --holdout --speaker NAME`.")
        sys.exit(1)
    for label, skipped in (("positives", skipped_p), ("negatives", skipped_n)):
        if skipped:
            print(f"WARNING: skipped {skipped} {label} not at {SR} Hz")

    print("=" * 70)
    print(f"{Path(args.model).name}   threshold {args.threshold}")
    print(backend.describe())
    print(f"{len(positives)} positives from {len(by_speaker)} speaker(s) "
          f"({', '.join(f'{s} {len(c)}' for s, c in by_speaker.items())}), "
          f"{len(negatives)} negatives from {paths.describe([args.negatives])}")
    print("=" * 70)

    # --- negatives, per category -------------------------------------------------
    by_category = evaluate_negatives(backend, negatives, args.threshold, rng)
    print("\nFALSE ACCEPTS BY CATEGORY (never read these pooled)")
    print(f"  {'category':<12}{'n':>4}{'fired':>7}{'rate':>8}{'median':>9}{'worst':>8}")
    adversarial_n = adversarial_fired = 0
    for category in sorted(by_category):
        rows = by_category[category]
        peaks = np.array([p for _, p in rows])
        fired = int((peaks >= args.threshold).sum())
        flag = " <-" if category in ADVERSARIAL else ""
        print(f"  {category:<12}{len(rows):>4}{fired:>7}{fired / len(rows):>7.0%}"
              f"{np.median(peaks):>9.3f}{peaks.max():>8.3f}{flag}")
        if category in ADVERSARIAL:
            adversarial_n += len(rows)
            adversarial_fired += fired

    worst = sorted((r for c in ADVERSARIAL for r in by_category.get(c, [])),
                   key=lambda r: -r[1])[:5]
    if worst and worst[0][1] >= args.threshold:
        print("\n  worst adversarial clips:")
        for clip_name, peak in worst:
            print(f"    {clip_name[:44]:<46}{peak:.3f}")

    # --- positives ---------------------------------------------------------------
    print("\nPOSITIVES")
    rows = evaluate_positives(backend, positives, args.threshold, rng, args.verbose)
    detected = sum(1 for r in rows if r[1])
    latencies = [r[2] for r in rows if r[2] is not None]
    misses = [(r[0], r[3]) for r in rows if not r[1]]
    print(f"  detected            {detected}/{len(positives)} ({detected / len(positives):.0%})")
    if latencies:
        lat = np.array(latencies)
        print(f"  latency from speech end   median {np.median(lat):>6.0f}ms   "
              f"p90 {np.percentile(lat, 90):>6.0f}ms")
        # Firing before the speech-end marker is normal for this corpus rather than a
        # defect: the marker is the last sample above 2% of peak, and these recordings
        # have a high noise floor (median -26 dB), so it lands on room tone after the
        # phrase. The model fires on the phrase, correctly, before the marker. It does
        # drag the mean negative, which is why median and p90 are reported instead.
        early = int((lat < 0).sum())
        if early:
            print(f"  NOTE: {early} clip(s) fired before their speech-end marker, which "
                  f"puts a floor under how low the median can read.")
    if misses:
        print(f"  missed {len(misses)}:")
        for clip_name, peak in sorted(misses, key=lambda r: -r[1])[:8]:
            print(f"    {clip_name[:44]:<46}{peak:.3f}")

    # The measurement the pipeline is built around: does the model work for EVERY
    # speaker, or only on average across them.
    speaker_rates = report_by_speaker(rows, speaker_spans) if len(speaker_spans) > 1 else []

    # A swept corpus pooled into one number says nothing - the whole point of a sweep
    # is where along it the model stops working.
    groups = defaultdict(list)
    for clip_name, ok, latency, peak in rows:
        groups[group_key(clip_name)].append((ok, latency, peak))
    if args.by_group and len(groups) > 1:
        print(f"\n  {'group':<16}{'n':>4}{'detected':>10}{'median score':>14}"
              f"{'median latency':>16}")
        for key in sorted(groups):
            entries = groups[key]
            ok = sum(1 for e in entries if e[0])
            lats = [e[1] for e in entries if e[1] is not None]
            peaks = np.array([e[2] for e in entries])
            lat = f"{np.median(lats):.0f}ms" if lats else "-"
            print(f"  {key:<16}{len(entries):>4}{ok:>6}/{len(entries):<3}"
                  f"{np.median(peaks):>14.3f}{lat:>16}")

    # --- positives with a command following ---------------------------------------
    commands = [(n, d) for n, d in negatives if n.startswith("command_")]
    detected_cmd = detected_gap = None
    if commands:
        print(f"\nPOSITIVES WITH A COMMAND FOLLOWING ({len(commands)} commands, cycled)")
        detected_cmd, _ = evaluate_with_command(
            backend, positives, commands, args.threshold, rng, 0, args.verbose)
        detected_gap, _ = evaluate_with_command(
            backend, positives, commands, args.threshold, rng,
            args.command_gap_ms, False)
        n = len(positives)
        print(f"  command immediately after {detected_cmd}/{n} ({detected_cmd / n:.0%})")
        print(f"  {args.command_gap_ms:.0f}ms pause, then command  "
              f"{detected_gap}/{n} ({detected_gap / n:.0%})")
        if detected_gap - detected_cmd >= max(2, 0.05 * n):
            print("  The pause recovers detections, which is the signature of a model")
            print("  trained on 'wake word, then quiet' (a measured result).")
    else:
        print(f"\nNo command_*.wav in {args.negatives}; skipping the command-following gate.")

    # --- threshold sweep (bug.md C3 step 2, 2026-09-22) -------------------------
    # The scores are all in hand: rows holds a peak per positive, by_category a
    # peak per negative, so the sweep is one pure pass over the peaks, not 19
    # model runs. It goes in the --json output (and the ledger verbatim with
    # it) so a matched-FA comparison is a lookup instead of a 40-eval manual
    # job.
    sweep = None
    if adversarial_n and positives:
        sweep = threshold_sweep(
            [r[3] for r in rows],
            [p for c in ADVERSARIAL for _, p in by_category.get(c, [])],
            SWEEP_GRID)

    # --- gates -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GATES")
    checks = []
    if adversarial_n:
        rate = adversarial_fired / adversarial_n
        checks.append((f"extend+hey_other false accepts  {adversarial_fired}/"
                       f"{adversarial_n} ({rate:.0%})", f"< {GATE_FALSE_ACCEPT:.0%}",
                       rate < GATE_FALSE_ACCEPT))
    else:
        # Scoring FAIL on a gate with no data reads as a real failure. Say so instead.
        print(f"  [ -- ]  extend+hey_other false accepts  no clips in those categories")
    checks.append((f"clean positive detection        {detected}/{len(positives)} "
                   f"({detected / len(positives):.0%})", f">= {GATE_POSITIVE:.0%}",
                   detected / len(positives) >= GATE_POSITIVE))
    # The same gate applied to the WEAKEST speaker rather than to the average. A model
    # that reads PASS pooled while failing one voice is not shippable to that person,
    # and the pooled row cannot show it - which is the whole reason this gate exists.
    if speaker_rates:
        rate, speaker, ok_n, total, _ci = min(speaker_rates)
        checks.append((f"weakest speaker ({speaker[:14]})".ljust(32)
                       + f"{ok_n}/{total} ({rate:.0%})",
                       f">= {GATE_POSITIVE:.0%}", rate >= GATE_POSITIVE))
    if detected_cmd is not None:
        checks.append((f"detection with command after    {detected_cmd}/{len(positives)} "
                       f"({detected_cmd / len(positives):.0%})", f">= {GATE_COMMAND:.0%}",
                       detected_cmd / len(positives) >= GATE_COMMAND))
    if latencies:
        median_latency = float(np.median(latencies))
        checks.append((f"median latency                  {median_latency:.0f}ms",
                       f"< {GATE_LATENCY_MS}ms", median_latency < GATE_LATENCY_MS))
    for text, gate, ok in checks:
        print(f"  [{verdict(ok)}]  {text:<48}{gate}")
    print("=" * 70)

    # One compact line, the detection@FA readings a human would have produced
    # by hand-driving this harness across 0.25-0.85 (40 evals; bug.md C3,
    # 2026-09-22). At-most-B semantics: the best detection on the curve with
    # FA <= B - the curve is a step function, never interpolated.
    if sweep:
        print()
        print(f"THRESHOLD SWEEP (re-thresholded over {len(sweep['thresholds'])} "
              "points; the model ran once)")
        parts = []
        for b in (0.005, 0.01, 0.02, 0.05):
            cands = [(d, a, t) for t, a, d in
                     zip(sweep["thresholds"], sweep["adv_rate"],
                         sweep["positives_rate"])
                     if a <= b]
            if cands:
                det, fa, t = max(cands, key=lambda c: (c[0], -c[1]))
                parts.append(f"det@FA<={b:.1%}: {det:.1%} (t={t:.2f})")
            else:
                parts.append(f"det@FA<={b:.1%}: - (no grid point at or below it)")
        print("  " + "   ".join(parts))

    # --- voice-holdout synthetic ranking set (improvement.md P1.2) ------------
    # A separate block on purpose: these clips are rendered from the voices the
    # corpus builders never train on, at in-distribution speeds, so the held-out
    # axis is the voice alone. They rank sweep points (low variance, a real n),
    # they do not gate anything - the gates above stay on real recordings, and
    # a synthetic voice is not a person.
    voice_holdout_result = None
    if args.voice_holdout_set:
        vdir = Path(args.voice_holdout_set)
        vclips, vskipped = load_dir(vdir)
        if vskipped:
            print(f"WARNING: skipped {vskipped} voice-holdout clips not at {SR} Hz")
        print("=" * 70)
        print("VOICE-HOLDOUT SYNTHETIC RANKING SET (NOT A GATE)")
        print(f"  {vdir}  ({len(vclips)} clips)")
        print("  Every clip is a voice the wordlist's `voice_holdout:` section holds out")
        print("  corpus build, at speeds inside the 0.7-1.3 training range: voice-"
              "disjoint from training, in-distribution everywhere else. A synthetic")
        print("  voice is not a person - this is a low-variance ranking signal for")
        print("  sweep points. It never stands in for the gates above, which stay on")
        print("  the real held-out recordings; the top two or three of the points it")
        print("  ranks go there.")
        if vclips:
            vrows = evaluate_positives(backend, vclips, args.threshold, rng, False)
            vdet = sum(1 for r in vrows if r[1])
            vlat = [r[2] for r in vrows if r[2] is not None]
            vrate = vdet / len(vclips)
            vlat_med = float(np.median(vlat)) if vlat else None
            voice_holdout_result = {
                "directory": str(vdir),
                "n": len(vclips),
                "detected": vdet,
                "rate": vrate,
                "latency_median_ms": vlat_med,
                "missed": [r[0] for r in vrows if not r[1]],
            }
            print(f"  threshold {args.threshold}: detected {vdet}/{len(vclips)} "
                  f"({vrate:.0%})"
                  + (f", median latency {vlat_med:.0f} ms" if vlat_med is not None else ""))
            print("  Compare this rate ACROSS sweep points; a movement smaller than")
            print("  this set's own run-to-run variance is not a result (see the five")
            print("  misreadings above). The set's set.json names its voices and the")
            print("  holdout file they came from - read it before trusting a comparison.")
        else:
            print("  no clips found - render it with:")
            print(f"    python -m eval.generate_positives --wake-word <phrase> --voice-holdout")
        print("=" * 70)

    # --- machine-readable copy of the same numbers -------------------------------
    # --json is the hook the run ledger (improvement.md P0.5) reads; the values are
    # exactly what the report above printed, nothing recomputed a second way.
    if args.json_path:
        per_speaker = {}
        for speaker, start, end in speaker_spans:
            entries = rows[start:end]
            ok = sum(1 for e in entries if e[1])
            ci = wilson_interval(ok, len(entries))
            lats = [e[2] for e in entries if e[2] is not None]
            per_speaker[speaker] = {
                "n": len(entries),
                "detected": ok,
                "rate": ok / len(entries),
                "ci95": [round(x, 4) for x in ci] if ci else None,
                "median_latency_ms": float(np.median(lats)) if lats else None,
            }
        out = {
            "model": str(args.model),
            "threshold": args.threshold,
            "backend": backend.describe(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "false_accepts_by_category": {
                c: {
                    "n": len(r),
                    "fired": int((np.array([p for _, p in r]) >= args.threshold).sum()),
                    "rate": float((np.array([p for _, p in r]) >= args.threshold).mean()),
                    "median_peak": float(np.median([p for _, p in r])),
                    "worst_peak": float(max(p for _, p in r)),
                }
                for c, r in by_category.items()
            },
            "adversarial": {
                "n": adversarial_n, "fired": adversarial_fired,
                "rate": adversarial_fired / adversarial_n if adversarial_n else None,
            },
            "positives": {
                "n": len(positives), "detected": detected,
                "rate": detected / len(positives),
                "latency_median_ms": float(np.median(latencies)) if latencies else None,
                "latency_p90_ms": float(np.percentile(latencies, 90)) if latencies else None,
                "missed": [n for n, _ in misses],
            },
            "per_speaker": per_speaker,
            "command_following": (
                {"n": len(positives),
                 "immediately_after": detected_cmd,
                 "pause_then_command": detected_gap,
                 "pause_ms": args.command_gap_ms}
                if detected_cmd is not None else None
            ),
            "gates": [{"check": text, "gate": gate, "pass": ok} for text, gate, ok in checks],
            # The matched-FA curve (bug.md C3 step 2, 2026-09-22): 19 x 2 rates
            # plus two counts - a few hundred bytes, one model run. The FA axis
            # is the extend+hey_other subset, exactly the "adversarial" field
            # above, never the pooled set. None for a run with no adversarial
            # clips or no positives; the ledger then falls back to the
            # one-threshold reading, as for every pre-sweep record on file.
            "threshold_sweep": sweep,
            # Deliberately a top-level sibling of "positives" and "gates", never
            # inside either: the voice-holdout set is a synthetic ranking signal
            # and a ledger reader must not be able to mistake it for the
            # real-speaker numbers it sits beside (improvement.md P1.2).
            "voice_holdout_set": voice_holdout_result,
        }
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nJSON written to {path}")


if __name__ == "__main__":
    main()
