#!/usr/bin/env python3
"""Build the microWakeWord corpus at data/corpus/<wake_word>/mww/.

Not the same directory as the openWakeWord corpus: train.py rmtree's its own
corpus at the start of every run. The build code is shared (corpus/) - same
trimming, child-range copies, audited voices, phrase texts, speed grid.

    python -m train.mww.corpus --wake-word "<wake word>" \
        --piper-url tcp://127.0.0.1:8898 [--kokoro-url ... --kokoro-fraction 0.3]

--piper-url is a comma-separated FLEET: the corpus shards BY VOICE, each model
pinned to one instance (one instance is one serial lane), so throughput scales
with instances.

Differences from the openWakeWord corpus, all deliberate:
1. REAL RECORDINGS ARE COPIED ONCE BY DEFAULT; 10x is the candidate, not the
   rule. openWakeWord's 10x works because it augments by globbing the directory
   once (measured 53% -> 77%); microWakeWord augments every read, so raw copies
   only bias sampling. The per-file split that made copies leak across splits -
   biasing the weights mWW SELECTS on validation average_viable_recall - is gone
   (features.py's group_partition splits by recording identity). What 10x is
   worth here is an open measurement; the default stays 1. --real-vtlp:
   formant-shifted real-clip copies are a DISTINCT feature row, not another draw
   of the same voice.
2. PIPER-MAJORITY, KOKORO AS A SUPPLEMENT: --kokoro-fraction SUBSTITUTES that
   share of the phrase-alone positive budget; totals and the negative set stay
   fixed. Default 0.0; the Apple Silicon run script 0.3. Negatives stay
   Piper-only: the per-category signal (extend, hey_other) lives there.
3. NO RUN-ON POSITIVES YET: the cut point needs Kokoro word timestamps; the
   fallback estimate measured a median +153 ms late against RUNON_TAIL_MS
   150-300 ms.
4. DEPTH IS NOT THE LEVER: doubling 60 -> 120 phrase-alone clips (both runs
   trained 20,000 steps) produced no deployable gain in either engine mix.
5. REJECTION IS TRAINED: the negative set is small on purpose (12 per voice);
   a 2x-depth run trained on it became a firehose (37.5% of its training
   negatives above 0.99, no FAPH operating point). Doubling to 24 per voice
   fixed exactly that - reach for this knob before depth.
"""

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# The import root src/, two levels up (not the git root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.corpus.augment import (CHILD_STRETCH_FRACTION,  # noqa: E402
                                  add_child_range_copies, trim_directory)
from train.corpus.kokoro import (KokoroPool,  # noqa: E402
                                 generate_kokoro_samples, probe_kokoro_servers)
from train.corpus.negatives import (LEGACY_VOICE_MARKER,  # noqa: E402
                                    build_negative_phrases,
                                    load_recipe_or_exit)
from train.corpus.piper import (generate_piper_samples,  # noqa: E402
                                select_piper_voices)
from train.corpus.positives import (PLAIN_SPEED_GRID,  # noqa: E402
                                    plain_positive_texts)
from train.corpus.real import (  # noqa: E402
    balanced_copy_weights, copy_real_samples, parse_balance_spec,
    speaker_clip_counts,
)
from train.corpus import manifest as corpus_manifest  # noqa: E402
from recipe import (exclude_voice_holdout, path_for,  # noqa: E402
                       voice_exclusions, voice_holdout)


def _parse_real_vtlp(spec: str, samples_dir, flag: str = "--real-vtlp") -> dict:
    """Parse 'speaker=N[,speaker=N]' into {speaker: N}, failing loud.

    Mirror of _parse_real_copies_override in train/oww/train.py, with the samples
    tree as a parameter: the names must be validated against the tree
    copy_real_samples actually reads, or a typo'd speaker is silently inert and
    the corpus is filed under a shaping that claims variants it does not carry.
    """
    if not spec:
        return {}
    overrides = {}
    samples = Path(samples_dir)
    known = {p.name for p in samples.iterdir() if p.is_dir()} if samples.is_dir() else set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"ERROR: {flag} {part!r}: expected speaker=N")
        speaker, _, n = part.partition("=")
        speaker, n = speaker.strip(), n.strip()
        if not n.isdigit() or int(n) < 1:
            sys.exit(f"ERROR: {flag} {part!r}: variants must be a positive int")
        if known and speaker not in known:
            sys.exit(f"ERROR: {flag} names {speaker!r}, but the samples "
                     f"tree has {sorted(known)} - the override would be inert")
        overrides[speaker] = int(n)
    return overrides


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", required=True)
    p.add_argument("--piper-url", default=os.environ.get("PIPER_URL", "tcp://127.0.0.1:8898"),
                   help="Piper protocol server(s), tcp:// URLs, comma-separated "
                        "to run a fleet (sharded by VOICE - every model pinned "
                        "to one instance for the run, so an instance never "
                        "reloads a model per request; src/scripts/start-tts-fleet.sh "
                        "N launches N on this Mac) (default: %%(default)s)")
    p.add_argument("--piper-speakers", type=int, default=12,
                   help="speakers sampled per multi-speaker voice (default: "
                        "%(default)s). libritts_r alone carries 904.")
    p.add_argument("--piper-languages", default="en_US,en_GB")
    p.add_argument("--kokoro-url",
                   default=os.environ.get("KOKORO_URL", "tcp://127.0.0.1:8899"),
                   help="Kokoro protocol server(s), tcp:// URLs, comma-separated "
                        "for a pool (default: %%(default)s). Used only when "
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
                   help="copies of each real recording (default: %(default)s - 10 is "
                        "the openWakeWord value, and the candidate here: with the "
                        "identity-aware split in train/mww/features.py the copies no "
                        "longer leak across splits, so raising the weight is a fair "
                        "measurement question now. See the module docstring, point 1, "
                        "for what is and is not settled.")
    p.add_argument("--balance-real-copies", default="",
                   help="Derive per-speaker --real-copies so each named speaker "
                        "contributes the SAME number of positive rows: 'all' or "
                        "'speakerA,speakerB'. Computed from the clip counts under "
                        "--real-samples at build time, so recording more of a thin "
                        "speaker shrinks their lift without touching a flag. Equalise UP "
                        "only. WHY: at any flat weight the least-recorded speaker is the "
                        "thinnest voice in the positive set, and can read worse on their "
                        "own training clips than on the holdout - a coverage problem, "
                        "not a threshold problem. Part of the corpus identity (the "
                        "spec; the multipliers follow from the clip counts, which "
                        "already are).")
    p.add_argument("--balance-max-multiplier", type=float, default=0.0,
                   help="Cap the derived lift at this multiple of --real-copies (0 = none).")
    p.add_argument("--real-vtlp", default="",
                   help="Per-speaker formant-shifted variants of the REAL clips, "
                        "'speaker=N[,speaker=N]' (speaker = the directory name under "
                        "--real-samples): adds N vocal-tract-shifted copies "
                        "(1.15-1.30x, the synthetic child-lever's range) per real clip, "
                        "at the base --real-copies weight. The mww port of the oww "
                        "clean-detection lever: shifted copies are NEW acoustic variants - new "
                        "feature rows - not more draws of the same voice. Part of the "
                        "corpus identity: a different value refuses the --skip reuse.")
    p.add_argument("--child-fraction", type=float, default=CHILD_STRETCH_FRACTION)
    p.add_argument("--corpus-root", default="data/corpus")
    p.add_argument("--real-samples", default=None,
                   help="the word's real recordings (default: "
                        "data/recordings/<wake_word>/samples)")
    p.add_argument("--negatives-file", default=None)
    p.add_argument("--clean", action="store_true",
                   help="delete an existing corpus first. Required to regenerate - "
                        "appending merges two runs and keeps clips from voices "
                        "excluded since.")
    p.add_argument("--skip", action="store_true",
                   help="reuse the existing corpus instead of building one, for a "
                        "run whose corpus is a held-fixed variable. Requires the "
                        "corpus to exist; when a corpus.json manifest exists it is "
                        "checked against the shaping flags AND the effective voice "
                        "set (live catalog minus exclusions minus the voice holdout) "
                        "this invocation would use, and a mismatch exits with a "
                        "diff - silently reusing a differently-shaped corpus or one "
                        "built from a different voice set is the failure this "
                        "refuses (the cf9c065b reuse, 2026-09-22, on the oww side, "
                        "matched every shaping flag and still trained on held-out "
                        "voices). The voice set is probed first: the catalog fetch "
                        "is cheap (no rendering), but the engines must be UP for a "
                        "--skip run - an unverifiable catalog is exactly the reuse "
                        "this check exists to refuse.")
    p.add_argument("--no-trim", action="store_true",
                   help="skip silence trimming. Almost certainly wrong: Piper "
                        "renderings carry a median 248 ms of trailing silence "
                        "(p90 555 ms), against 0 ms for real recordings.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed the drawing/sampling of the corpus (default: "
                        "%(default)s = unseeded). Seeds which voices, speakers, "
                        "phrases and speeds get chosen - NOT how they are rendered: "
                        "the TTS engines sample noise per call and cannot be "
                        "seeded, so a same-seed rebuild is a same-shape corpus with "
                        "different audio. That is why a sweep REUSES this corpus "
                        "(the manifest written at the end of this stage) rather "
                        "than rebuilding it.")
    args = p.parse_args()
    if args.real_samples is None:
        args.real_samples = f"data/recordings/{args.wake_word.replace(' ', '_').lower()}/samples"

    if args.seed:
        # BEFORE any draw below: the stage must be a function of the seed, not
        # the clock. numpy carries the helpers' draws; random too, because the
        # helpers are allowed to use either.
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[seed] {args.seed} (drawing is seeded; rendering is not - the "
              f"TTS engines cannot be)")

    t_start = time.time()

    safe = args.wake_word.replace(" ", "_").lower()
    root = Path(args.corpus_root) / safe / "mww"
    positives, negatives = root / "positives", root / "negatives"

    # The fraction gates which engines are probed below, so validate it first.
    if not 0.0 <= args.kokoro_fraction < 1.0:
        sys.exit("  --kokoro-fraction must be in [0, 1) - Piper stays primary in "
                 "this corpus, because the negatives are Piper-only")

    # --real-vtlp: parsed BEFORE the voice probes and the --skip decision, so an
    # inert override fails loud on a reuse run too; the parsed dict is part of
    # the requested shaping the reuse check diffs against the manifest.
    real_vtlp = _parse_real_vtlp(args.real_vtlp, args.real_samples)

    # VOICE SET: resolved BEFORE the --skip decision, from the same code path
    # the build uses - one computation, so the check and the build cannot
    # drift. (The oww-side cf9c065b reuse, 2026-09-22, matched every flag but
    # trained on seven held-out voices.) Probes are cheap catalog fetches,
    # but the engines must be UP for a --skip run: an unverifiable catalog is
    # exactly the reuse the check refuses.

    # Voice holdout: loaded HERE, before either engine
    # branch and the corpus-mode decision; enforced against the live catalog,
    # so a list that drifted from it fails loudly. Absent section is a no-op.
    recipe = load_recipe_or_exit(args.wake_word)
    holdout = voice_holdout(recipe)
    voices = []
    if args.kokoro_fraction < 1.0:
        print(f"[Piper] {args.piper_url}")
        voices = select_piper_voices(
            args.piper_url, args.wake_word,
            languages=tuple(args.piper_languages.split(",")),
            max_speakers=args.piper_speakers)
        if not voices:
            sys.exit("  no usable Piper voices - nothing to generate")
        if holdout.get("piper"):
            # A holdout pair the audited selection no longer carries cannot serve as an
            # eval voice either: the tracked list moves, not the audit.
            n_before = len(voices)
            voices, holdout_missing = exclude_voice_holdout("piper", voices, holdout)
            if holdout_missing:
                sys.exit(f"  ERROR: the voice holdout ({recipe['_path']}) names "
                         f"Piper (voice, speaker) pair(s) the live audited "
                         f"selection does not carry: {holdout_missing}. Update the "
                         f"recipe's `voice_holdout:` section to match the catalog this corpus is "
                         f"built from.")
            print(f"  Excluding {n_before - len(voices)} voice-holdout (voice, speaker) "
                  f"pair(s) reserved for the synthetic ranking set")
        else:
            print(f"  NOTE: the recipe carries no `voice_holdout:` section - the "
                  f"synthetic ranking set has no reserved voices")

    # KOKORO SUPPLEMENTS THE PHRASE-ALONE BUDGET (module docstring, point 2):
    # a share of what Piper would have rendered is rendered by it instead.
    kokoro_voices, kokoro_pool = [], None
    if args.kokoro_fraction > 0.0:
        print(f"\n[Kokoro] {args.kokoro_url}")
        kokoro_pool = KokoroPool(args.kokoro_url.split(","))
        kokoro_voices = probe_kokoro_servers(kokoro_pool)
        # Same exclusions the openWakeWord corpus applies: a voice that says something
        # other than the wake word is a mislabelled positive regardless of engine, and
        # the v0 legacy set is older renderings of speakers already in the set. The list
        # is per-word data read from the recipe - how a voice renders one phrase says
        # nothing about another.
        excluded = set(voice_exclusions(recipe, "kokoro")["mispronouncing"])
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
        if holdout.get("kokoro"):
            n_before = len(kokoro_voices)
            kokoro_voices, holdout_missing = exclude_voice_holdout(
                "kokoro", kokoro_voices, holdout)
            if holdout_missing:
                sys.exit(f"  ERROR: the voice holdout ({recipe['_path']}) names "
                         f"Kokoro voice(s) the live catalog does not offer: "
                         f"{holdout_missing}. Update the recipe's `voice_holdout:` section.")
            print(f"  Excluding {n_before - len(kokoro_voices)} voice-holdout "
                  f"voice(s) reserved for the synthetic ranking set: "
                  f"{', '.join(holdout.get('kokoro') or [])}")
        if not kokoro_voices:
            sys.exit("  no usable Kokoro voices - re-run with "
                     "--kokoro-fraction 0 (all Piper)")

    # What a --skip run validates against, and what the manifest records: the
    # requested shaping flags plus the TOP-LEVEL voice set - a different axis
    # from every flag; piper entries are (voice, speaker) pairs, the shape
    # select_piper_voices returns and the manifest stores.
    requested = {
        "samples_per_voice": args.samples_per_voice,
        "negatives_per_voice": args.negatives_per_voice,
        "kokoro_fraction": args.kokoro_fraction,
        "child_fraction": args.child_fraction,
        "real_copies": args.real_copies,
        "real_vtlp": real_vtlp,
        "balance_real_copies": args.balance_real_copies,
        "piper_speakers": args.piper_speakers,
        "piper_languages": args.piper_languages,
        "negatives_file": args.negatives_file,
        "no_trim": args.no_trim,
        "voices": {"kokoro": kokoro_voices, "piper": voices},
    }

    # REUSE MODE: the sweep's front door. Decided AFTER the probes, because the
    # check diffs the voice set the probes just resolved.
    if args.skip:
        existing = {d: len(list(d.glob("*.wav"))) for d in (positives, negatives)
                    if d.is_dir()}
        if not any(existing.values()):
            sys.exit(f"\n--skip, but no corpus at {root} - build it first "
                     f"(drop --skip)")
        manifest = corpus_manifest.load_manifest(root)
        if manifest is not None:
            corpus_manifest.check_reuse(root, requested)
        else:
            print(f"  NOTE: no corpus.json manifest at {root} - the corpus predates "
                  f"the manifest stage, so its shaping and voice set cannot be "
                  f"verified. Reusing as-is; a rebuild would write one.")
        for d, n in existing.items():
            print(f"  reusing {d} ({n} wav)")
        return

    # REFUSE TO APPEND TO AN EXISTING CORPUS. A silent merge of two runs is
    # worse than it sounds: exclusion is applied when GENERATING, so clips
    # from voices excluded since stay in; add_child_range_copies globs the whole
    # directory, so old clips get a second set of shifted copies; and real
    # recordings are copied again, changing their share. train.py's
    # setup_training_dirs rmtree's for the same reason.
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

    # The split, in TOTAL clips: Piper keeps its per-voice budget scaled down by
    # the fraction, the difference spread over the Kokoro voices - voice counts
    # differ, so a per-voice figure would not substitute one-for-one.
    piper_per_voice = args.samples_per_voice
    kokoro_per_voice = 0
    if 0.0 < args.kokoro_fraction < 1.0:
        piper_per_voice = max(1, int(round(args.samples_per_voice
                                           * (1 - args.kokoro_fraction))))
        kokoro_total = (args.samples_per_voice - piper_per_voice) * len(voices)
        kokoro_per_voice = max(1, kokoro_total // len(kokoro_voices))

    print(f"\n[Positives] -> {positives}")
    texts = plain_positive_texts(args.wake_word)
    generate_piper_samples(args.piper_url, voices, positives,
                           piper_per_voice,
                           texts,
                           PLAIN_SPEED_GRID, "Piper positives")
    if kokoro_voices:
        generate_kokoro_samples(kokoro_pool, kokoro_voices, positives,
                                kokoro_per_voice,
                                texts, "Kokoro positives")

    # The adversarial negatives - the sounds the large ambient sets do NOT
    # contain. Piper-only on purpose (module docstring, point 2).
    print(f"\n[Negatives] -> {negatives}")
    phrases = build_negative_phrases(args.wake_word, args.negatives_file)
    generate_piper_samples(args.piper_url, voices, negatives,
                           args.negatives_per_voice, phrases,
                           PLAIN_SPEED_GRID, "Piper negatives")

    # Before the real clips, so only synthetic output is shifted - and before
    # trimming, so the shifted copies are trimmed like everything else.
    if args.child_fraction > 0:
        print("\n[Child-range copies]")
        add_child_range_copies(positives, "VTLP positives", args.child_fraction)

    print("\n[Real Voice]")
    # NOTE - what does NOT port is the oww-side PER-SPEAKER raw-copy weight
    # (--real-copies-override): mww's sampling weight is ONE number per FEATURE
    # SET, and synthetic and real clips share the positives directory, so it
    # cannot aim at one speaker. The identity-aware split (features.py's
    # group_partition) is what made --real-copies safe to raise here. The
    # per-speaker DIVERSITY lever is --real-vtlp. See corpus/real.py's
    # NOTE FOR THE microWakeWord PORT.
    real_overrides = None
    balance_spec = parse_balance_spec(args.balance_real_copies)
    if balance_spec is not None:
        real_overrides, balance_notes = balanced_copy_weights(
            speaker_clip_counts(Path(args.real_samples)), args.real_copies,
            None if balance_spec == "all" else balance_spec,
            max_multiplier=args.balance_max_multiplier)
        print("  --balance-real-copies "
              f"{args.balance_real_copies!r}: " + "; ".join(balance_notes))
    copy_real_samples(Path(args.real_samples), positives, args.real_copies,
                      per_speaker_copies=real_overrides,
                      per_speaker_vtlp=real_vtlp)

    if not args.no_trim:
        print("\n[Trim]")
        for directory, label in ((positives, "positives"), (negatives, "negatives")):
            n, mean_ms = trim_directory(directory, f"Trim {label}")
            print(f"  {label}: trimmed {n} clips, mean {mean_ms:.0f} ms removed")

    n_pos = len(list(positives.glob("*.wav")))
    n_neg = len(list(negatives.glob("*.wav")))

    # FREEZE THE CORPUS: the manifest is the identity the run tag hashes and a
    # later --skip reuse (or the sweep runner) validates against. Written last,
    # after trimming and the child copies, so its digest covers the final tree
    # - the state training will actually consume.
    corpus_manifest.write_manifest(
        root, args.wake_word, "mww",
        seed=args.seed,
        # The REQUESTED shaping (what a later --skip reuse check diffs) plus the
        # holdout list, so the manifest names the exclusion. real_vtlp is the
        # PARSED dict - the reuse check compares structure, and the string form
        # could not be diffed against a later parsed request.
        shaping={
            "samples_per_voice": args.samples_per_voice,
            "negatives_per_voice": args.negatives_per_voice,
            "kokoro_fraction": args.kokoro_fraction,
            "child_fraction": args.child_fraction,
            "real_copies": args.real_copies,
            # parsed dict, not raw string: the --skip check compares structure
            "real_vtlp": real_vtlp,
            "balance_real_copies": args.balance_real_copies,
            "piper_speakers": args.piper_speakers,
            "piper_languages": args.piper_languages,
            "negatives_file": args.negatives_file,
            "no_trim": args.no_trim,
            "voice_holdout": holdout,
        },
        engines={
            "piper": {"url": args.piper_url, "version": None},
            **({"kokoro": {"url": args.kokoro_url, "version": None}}
               if args.kokoro_fraction > 0 else {}),
        },
        voices={"kokoro": kokoro_voices, "piper": voices},
        per_voice_counts={"positives": n_pos, "negatives": n_neg},
        recipe_path=path_for(args.wake_word),
        wall_time_s=time.time() - t_start)

    print(f"\nDONE  {n_pos} positives, {n_neg} negatives under {root}")
    print("\nNext - FEATURES, not config: the config points at "
          "features/positives, which the next step creates.")
    print(f'  python -m train.mww.features --wake-word "{args.wake_word}"')
    print("\nOr let a wrapper chain all four stages:")
    print(f'  ./src/scripts/run-mww-training.sh "{args.wake_word}"   (Docker)')
    print(f'  ./src/scripts/run-mww-training-applesilicon.sh "{args.wake_word}"   (host, Apple Silicon)')


if __name__ == "__main__":
    main()
