#!/usr/bin/env python3
"""Build the microWakeWord corpus at data/corpus/<wake_word>/mww/.

Sibling to the openWakeWord corpus at .../oww/, and deliberately not the same
directory: train.py rmtree's its own at the start of every run, so a shared corpus
would be destroyed by whichever pipeline ran next.

WHAT IS SHARED IS THE CODE, NOT THE OUTPUT. Everything here comes from corpus/ - the
same trimming, the same child-range copies, the same audited Piper voices, the same
tuned phrase texts and speed grid. Two corpora built by one set of rules.

    python -m train.mww.corpus --wake-word "hey seeree" --piper-url piper:10200 \
        --kokoro-url http://127.0.0.1:8880 --kokoro-fraction 0.3

DIFFERENCES FROM THE openWakeWord CORPUS, all deliberate:

1. REAL RECORDINGS ARE COPIED ONCE, not ten times. openWakeWord's --real-copies 10
   exists because it augments by globbing the directory once, so N copies become N
   independently augmented variants - the largest single lever measured there (run
   10, run-on 53% -> 77%). microWakeWord augments on every read instead, so copies
   would only bias sampling, and `sampling_weight` in the feature set is the honest
   knob for that. See corpus/real.py.

2. PIPER-MAJORITY, WITH KOKORO AS A SUPPLEMENT. --kokoro-fraction renders that
   share of the PHRASE-ALONE positive budget with Kokoro instead of Piper. It
   SUBSTITUTES rather than adds, the same discipline the openWakeWord side applies
   to its --piper-fraction: total clip count, real-clip share of the positive set,
   and the negative set all stay fixed, so a comparison against an all-Piper run
   means exactly one thing - where part of the phrase-alone budget came from. Run 17
   measured two engines beating one on the openWakeWord side by the largest margin
   since run 10; this is that lever, engines swapped (that corpus is Kokoro-primary
   with a Piper fraction, this one the mirror image). The module default is 0.0 (all
   Piper, the historical behaviour); the Apple Silicon run script defaults to 0.3,
   mirroring the 30% its oWW sibling already runs. Negatives stay Piper-only on
   purpose: that is where the per-category signal (extend, hey_other) lives, and a
   second engine would blur attribution of a false accept to an engine. The Kokoro
   voices get the same exclusions the oWW corpus applies (MISPRONOUNCING_VOICES and
   the v0 legacy set, corpus/negatives.py) and the shared speed grid, so the two
   engines differ in timbre, not in speed or text. The mlx:// in-process backend
   works here too but is not installed in this environment (see
   corpus/kokoro_mlx.py) - from this venv, use the host server:
   scripts/start-kokoro-host.sh.

3. NO RUN-ON POSITIVES YET. Their cut point comes from Kokoro's word timestamps, and
   Wyoming exposes no equivalent - the fallback estimate measured a median +153 ms
   late, against a RUNON_TAIL_MS of 150-300 ms. On the openWakeWord side run-ons took
   held-out run-on detection from 5% to the 80s, so this is the most valuable gap
   here, and it needs solving properly rather than with the degraded estimate.
   The client that would solve it (corpus/kokoro.py, with phrase_end_sample) is now
   importable from here; the constants it needs (RUNON_TAIL_MS) stay
   openWakeWord-local until this gap is closed.

4. DEPTH IS NOT THE LEVER. The default is 60 phrase-alone clips per voice.
   Doubling that to 120 (2026-09-08, both runs trained 20,000 steps so the extra
   data was actually seen) never produced a deployable gain, and the two 2x runs
   failed in opposite ways. The 30% mix collapsed on held-out detection: at
   12/32 adversarial false accepts, plain 86 -> 59 and run-on 93 -> 28 against the
   1x mix, 17/32 adversarial false accepts at the 0.5 reference. The all-Piper 2x
   kept the best held-out detection of any run (84% plain / 87% run-on at 0.5,
   ceilings 86/90) but lost its operating point entirely: 37.5% of its own training
   negatives score above 0.99 (training-ROC AUC 0.295, below chance) and no cutoff
   meets the 0.2 FAPH budget, so the manifest stage refused to write. So: the
   Kokoro share has a sweet spot at 1x (the mix's run-on win came from the engines,
   not the volume), and at doubled depth the mix was the poison while Piper-only
   depth was neutral-to-harmful. The remaining levers are the ones depth cannot
   touch: more REAL recordings (the per-speaker spread - jen at 0-30% against jay
   at 69-89% - is the standing failure in every configuration) and the run-on
   positives in point 3.

5. REJECTION IS TRAINED, NOT FREE. The negative set is small on purpose in this
   module's history (12 adversarial clips per voice, 984 total against ~8,000
   positives), and the 2026-09-08 doubled-depth runs showed what that leaves out:
   a model trained on 15,156 positives and 984 adversarial clips became a firehose
   - 37.5% of its own training negatives above 0.99, no FAPH operating point,
   manifest unwritable. Doubling the negatives to 24 per voice (1,968 total; tag
   ecbf160-dirty-da01854d, all-Piper 1x depth, 15m54s on the Apple Silicon host)
   fixed exactly that and nothing else: at its calibrated 0.09 cutoff it passed the
   extend+hey_other gate for the first time in this repo (1/32, versus 5-17/32 for
   every earlier run) with zero training false accepts at 0.81, and per-speaker
   plain detection became the best measured (jay 94, jen 40, ryan 83 at 4/32
   matched). The price was recall, not rejection: run-on 37 (against 65-93 for the
   1x runs), detection-with-command 57%, median latency 261 ms - the conservative
   model fires late - and the per-speaker wall (jen 20-40%) survived it, as it has
   every lever so far. If a run stops rejecting things that used to be rejected,
   reach for this knob before reaching for depth.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# The REPO ROOT. This package sits at train/mww/ since the reorg, so the root is
# two levels up, not one - the old value now points at train/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.corpus.augment import (CHILD_STRETCH_FRACTION,  # noqa: E402
                                  add_child_range_copies, trim_directory)
from train.corpus.kokoro import (KokoroPool,  # noqa: E402
                                 generate_kokoro_samples, probe_kokoro_servers)
from train.corpus.negatives import (LEGACY_VOICE_MARKER,  # noqa: E402
                                    MISPRONOUNCING_VOICES, build_negative_phrases)
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
    p.add_argument("--kokoro-url",
                   default=os.environ.get("KOKORO_URL", "http://localhost:8880"),
                   help="Kokoro TTS URL, comma-separated for a pool; 'mlx://' is "
                        "in-process (default: %%(default)s). Used only when "
                        "--kokoro-fraction > 0.")
    p.add_argument("--kokoro-fraction", type=float, default=0.0,
                   help="Share of the PHRASE-ALONE positive budget rendered by "
                        "Kokoro instead of Piper (default: %%(default)s = all "
                        "Piper). Substitutes rather than adds - see the module "
                        "docstring. Piper stays primary: negatives are "
                        "Piper-only, so the fraction must be < 1.")
    p.add_argument("--samples-per-voice", type=int, default=60,
                   help="phrase-alone clips per voice (default: %(default)s). Lower "
                        "than openWakeWord's 300 because there are ~82 usable Piper "
                        "voices against ~36 Kokoro ones.")
    p.add_argument("--negatives-per-voice", type=int, default=12)
    p.add_argument("--real-copies", type=int, default=1,
                   help="copies of each real recording (default: %(default)s). See "
                        "the module docstring for why this is not 10.")
    p.add_argument("--child-fraction", type=float, default=CHILD_STRETCH_FRACTION)
    p.add_argument("--corpus-root", default="data/corpus")
    p.add_argument("--real-samples", default="data/recordings/samples")
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
    # The result is a corpus no one intended, with no error and only a clip count
    # to notice it by. train.py's setup_training_dirs rmtree's for the same reason.
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
    if not 0.0 <= args.kokoro_fraction < 1.0:
        sys.exit("  --kokoro-fraction must be in [0, 1) - Piper stays primary in "
                 "this corpus, because the negatives are Piper-only")

    voices = []
    if args.kokoro_fraction < 1.0:
        print(f"[Piper] {args.piper_url}")
        voices = select_piper_voices(
            host, port, args.wake_word,
            languages=tuple(args.piper_languages.split(",")),
            max_speakers=args.piper_speakers)
        if not voices:
            sys.exit("  no usable Piper voices - nothing to generate")

    # KOKORO SUPPLEMENTS THE PHRASE-ALONE BUDGET (see the module docstring):
    # a share of what Piper would have rendered is rendered by it instead.
    kokoro_voices, kokoro_pool = [], None
    if args.kokoro_fraction > 0.0:
        print(f"\n[Kokoro] {args.kokoro_url}")
        kokoro_pool = KokoroPool(args.kokoro_url.split(","))
        kokoro_voices = probe_kokoro_servers(kokoro_pool)
        # The same exclusions the openWakeWord corpus applies, for the same
        # reason: a voice that says something other than the wake word is a
        # mislabelled positive regardless of engine, and the v0 legacy set is
        # older renderings of speakers already in the set. Six of 42 Kokoro
        # voices did exactly this for "hey seeree" and went unnoticed for
        # eleven runs - this list is not optional.
        excluded = set(MISPRONOUNCING_VOICES.get(safe, []))
        legacy = sorted(v for v in kokoro_voices if LEGACY_VOICE_MARKER in v)
        if legacy:
            excluded.update(legacy)
            print(f"  Skipping {len(legacy)} v0 legacy voice(s) - older "
                  f"renderings of speakers already in the set, for no measured "
                  f"gain")
        mispron = sorted(excluded & set(kokoro_voices))
        if mispron:
            print(f"  Excluding {len(mispron)} voice(s) that mispronounce the "
                  f"wake word: {', '.join(mispron)}")
        kokoro_voices = [v for v in kokoro_voices if v not in excluded]
        if not kokoro_voices:
            sys.exit("  no usable Kokoro voices - re-run with "
                     "--kokoro-fraction 0 (all Piper)")

    # The split, in TOTAL clips: Piper keeps its per-voice budget scaled down by
    # the fraction, and the difference is spread over however many Kokoro voices
    # there are. The two engines do not have the same voice count, so a
    # per-voice figure would not substitute one-for-one. (The mirror image of
    # openWakeWord's --piper-fraction arithmetic.)
    piper_per_voice = args.samples_per_voice
    kokoro_per_voice = 0
    if 0.0 < args.kokoro_fraction < 1.0:
        piper_per_voice = max(1, int(round(args.samples_per_voice
                                           * (1 - args.kokoro_fraction))))
        kokoro_total = (args.samples_per_voice - piper_per_voice) * len(voices)
        kokoro_per_voice = max(1, kokoro_total // len(kokoro_voices))

    print(f"\n[Positives] -> {positives}")
    texts = plain_positive_texts(args.wake_word)
    generate_piper_samples(host, int(port), voices, positives,
                           piper_per_voice,
                           texts,
                           PLAIN_SPEED_GRID, "Piper positives")
    if kokoro_voices:
        generate_kokoro_samples(kokoro_pool, kokoro_voices, positives,
                                kokoro_per_voice,
                                texts, "Kokoro positives")

    # The adversarial negatives - "hey serious", "hey Sienna", and the same sounds
    # inside running speech. These are what the large ambient sets do NOT contain,
    # and `extend` false accepts have been the unsolved problem on the openWakeWord
    # side since run 6. Piper-only on purpose: this is where the per-category
    # signal lives, and a second engine would blur the attribution.
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
    print("\nNext - FEATURES, not config: the config points at "
          "features/positives, which the next step creates.")
    print(f'  python -m train.mww.features --wake-word "{args.wake_word}"')
    print("\nOr let a wrapper chain all four stages:")
    print(f'  ./scripts/run-mww-training.sh "{args.wake_word}"   (Docker)')
    print(f'  ./scripts/run-mww-training-applesilicon.sh "{args.wake_word}"   (host, Apple Silicon)')


if __name__ == "__main__":
    main()
