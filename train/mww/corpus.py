#!/usr/bin/env python3
"""Build the microWakeWord corpus at my_custom_model/<wake_word>/mww/.

Sibling to the openWakeWord corpus at .../oww/, and deliberately not the same
directory: train.py rmtree's its own at the start of every run, so a shared corpus
would be destroyed by whichever pipeline ran next.

WHAT IS SHARED IS THE CODE, NOT THE OUTPUT. Everything here comes from corpus/ - the
same trimming, the same child-range copies, the same audited Piper voices, the same
tuned phrase texts and speed grid. Two corpora built by one set of rules.

    python -m mww.corpus --wake-word "hey seeree" --piper-url piper:10200

THREE DIFFERENCES FROM THE openWakeWord CORPUS, all deliberate:

1. REAL RECORDINGS ARE COPIED ONCE, not ten times. openWakeWord's --real-copies 10
   exists because it augments by globbing the directory once, so N copies become N
   independently augmented variants - the largest single lever measured there (run
   10, run-on 53% -> 77%). microWakeWord augments on every read instead, so copies
   would only bias sampling, and `sampling_weight` in the feature set is the honest
   knob for that. See corpus/real.py.

2. PIPER ONLY, FOR NOW. The Kokoro client still lives inside train.py rather than
   corpus/, so this cannot render with it yet. That is a known gap and not a
   preference: run 17 measured two engines beating one on the openWakeWord side by
   the largest margin since run 10. Closing it means extracting corpus/kokoro.py.

3. NO RUN-ON POSITIVES YET. Their cut point comes from Kokoro's word timestamps, and
   Wyoming exposes no equivalent - the fallback estimate measured a median +153 ms
   late, against a RUNON_TAIL_MS of 150-300 ms. On the openWakeWord side run-ons took
   held-out run-on detection from 5% to the 80s, so this is the most valuable gap
   here, and it needs solving properly rather than with the degraded estimate.
"""

import argparse
import shutil
import sys
from pathlib import Path

# The REPO ROOT. This package sits at train/mww/ since the reorg, so the root is
# two levels up, not one - the old value now points at train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.corpus.augment import (CHILD_STRETCH_FRACTION,  # noqa: E402
                                  add_child_range_copies, trim_directory)
from train.corpus.negatives import build_negative_phrases  # noqa: E402
from train.corpus.piper import (generate_piper_samples,  # noqa: E402
                                select_piper_voices)
from train.corpus.positives import (PLAIN_SPEED_GRID,  # noqa: E402
                                    plain_positive_texts)
from train.corpus.real import copy_real_samples  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--piper-url", default="piper:10200",
                   help="Wyoming TTS host:port (default: %(default)s)")
    p.add_argument("--piper-speakers", type=int, default=12,
                   help="speakers sampled per multi-speaker voice (default: "
                        "%(default)s). libritts_r alone carries 904.")
    p.add_argument("--piper-languages", default="en_US,en_GB")
    p.add_argument("--samples-per-voice", type=int, default=60,
                   help="phrase-alone clips per voice (default: %(default)s). Lower "
                        "than openWakeWord's 300 because there are ~82 usable Piper "
                        "voices against ~36 Kokoro ones.")
    p.add_argument("--negatives-per-voice", type=int, default=12)
    p.add_argument("--real-copies", type=int, default=1,
                   help="copies of each real recording (default: %(default)s). See "
                        "the module docstring for why this is not 10.")
    p.add_argument("--child-fraction", type=float, default=CHILD_STRETCH_FRACTION)
    p.add_argument("--corpus-root", default="my_custom_model")
    p.add_argument("--real-samples", default="my_real_samples")
    p.add_argument("--negatives-file", default=None)
    p.add_argument("--clean", action="store_true",
                   help="delete an existing corpus first. Required to regenerate - "
                        "appending merges two runs and keeps clips from voices "
                        "excluded since.")
    p.add_argument("--no-trim", action="store_true",
                   help="skip silence trimming. Almost certainly wrong: Piper "
                        "renderings carry a median 248 ms of trailing silence "
                        "(p90 555 ms), against 0 ms for real recordings.")
    args = p.parse_args()

    safe = args.wake_word.replace(" ", "_").lower()
    root = Path(args.corpus_root) / safe / "mww"
    positives, negatives = root / "positives", root / "negatives"

    # REFUSE TO APPEND TO AN EXISTING CORPUS. Generating into a non-empty directory
    # silently merges two runs, and the merge is worse than it sounds:
    #
    #   * clips from voices excluded since the last run stay in the corpus - the
    #     exclusion list is applied when GENERATING, not when reading
    #   * add_child_range_copies globs the whole directory, so the previous run's
    #     clips get a second set of shifted copies
    #   * real recordings are copied again, changing their share of the corpus
    #
    # The result is a corpus no one intended, with no error and only a clip count to
    # notice it by. train.py's setup_training_dirs rmtree's for the same reason.
    existing = {d: len(list(d.glob("*.wav"))) for d in (positives, negatives)
                if d.is_dir()}
    if any(existing.values()):
        if not args.clean:
            print("REFUSING TO GENERATE: corpus already exists")
            for d, n in existing.items():
                print(f"  {d}  ({n} wav)")
            print("\nGenerating on top of it would merge two runs - including clips")
            print("from voices excluded since, and a second round of child-range")
            print("copies over the old ones. Re-run with --clean to replace it.")
            sys.exit(1)
        for d in (positives, negatives):
            if d.is_dir():
                print(f"  removing {d} ({existing.get(d, 0)} wav)")
                shutil.rmtree(d)

    positives.mkdir(parents=True, exist_ok=True)
    negatives.mkdir(parents=True, exist_ok=True)

    host, _, port = args.piper_url.rpartition(":")
    print(f"[Piper] {args.piper_url}")
    voices = select_piper_voices(
        host, port, args.wake_word,
        languages=tuple(args.piper_languages.split(",")),
        max_speakers=args.piper_speakers)
    if not voices:
        sys.exit("  no usable Piper voices - nothing to generate")

    print(f"\n[Positives] -> {positives}")
    generate_piper_samples(host, int(port), voices, positives,
                           args.samples_per_voice,
                           plain_positive_texts(args.wake_word),
                           PLAIN_SPEED_GRID, "Piper positives")

    # The adversarial negatives - "hey serious", "hey Sienna", and the same sounds
    # inside running speech. These are what the large ambient sets do NOT contain,
    # and `extend` false accepts have been the unsolved problem on the openWakeWord
    # side since run 6.
    print(f"\n[Negatives] -> {negatives}")
    phrases = build_negative_phrases(args.wake_word, args.negatives_file)
    generate_piper_samples(host, int(port), voices, negatives,
                           args.negatives_per_voice, phrases,
                           PLAIN_SPEED_GRID, "Piper negatives")

    # Before the real clips, so only synthetic output is shifted - and before
    # trimming, so the shifted copies are trimmed like everything else. Same order
    # as train.py, for the same reasons.
    if args.child_fraction > 0:
        print("\n[Child-range copies]")
        add_child_range_copies(positives, "VTLP positives", args.child_fraction)

    print("\n[Real Voice]")
    copy_real_samples(Path(args.real_samples), positives, args.real_copies)

    if not args.no_trim:
        print("\n[Trim]")
        for directory, label in ((positives, "positives"), (negatives, "negatives")):
            n, mean_ms = trim_directory(directory, f"Trim {label}")
            print(f"  {label}: trimmed {n} clips, mean {mean_ms:.0f} ms removed")

    n_pos = len(list(positives.glob("*.wav")))
    n_neg = len(list(negatives.glob("*.wav")))
    print(f"\nDONE  {n_pos} positives, {n_neg} negatives under {root}")
    print("\nNext:")
    print(f'  python -m mww.config --wake-word "{args.wake_word}" \\')
    print("      --ambient data/mww_ambient/speech data/mww_ambient/no_speech \\")
    print("      --data-dir data --out training_parameters.yaml")


if __name__ == "__main__":
    main()
