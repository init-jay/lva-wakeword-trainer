#!/usr/bin/env python3
"""Build the microWakeWord corpus at data/corpus/<wake_word>/mww/.

Sibling to the openWakeWord corpus at .../oww/, and deliberately not the same
directory: train.py rmtree's its own at the start of every run, so a shared corpus
would be destroyed by whichever pipeline ran next.

WHAT IS SHARED IS THE CODE, NOT THE OUTPUT. Everything here comes from corpus/ - the
same trimming, the same child-range copies, the same audited Piper voices, the same
tuned phrase texts and speed grid. Two corpora built by one set of rules.

    python -m train.mww.corpus --wake-word "hey seeree" \
        --piper-url tcp://127.0.0.1:8898 \
        --kokoro-url tcp://127.0.0.1:8900 --kokoro-fraction 0.3

A Piper FLEET is one comma-separated --piper-url
(tcp://127.0.0.1:8898,tcp://127.0.0.1:8899,... - the list scripts/
start-tts-fleet.sh prints): the corpus shards it BY VOICE, each model pinned to
one instance for the whole run, so an instance loads each of its models once
rather than reloading on most requests (corpus/piper.py, PiperFleet). One
instance is one serial lane - the lane is the engine's lock, not the client's -
so throughput scales with instances, and this is the fast path for the corpus
stage (improvement.md P2.1).

DIFFERENCES FROM THE openWakeWord CORPUS, all deliberate:

1. REAL RECORDINGS ARE COPIED ONCE BY DEFAULT - and 10x is the candidate, not the
   rule. openWakeWord's --real-copies 10 exists because it augments by globbing the
   directory once, so N copies become N independently augmented variants - the largest
   single lever measured there (run 10, run-on 53% -> 77%). microWakeWord generates its
   feature rows UP FRONT (features.py) and augments on every read instead, so raw copies
   only bias sampling - and for as long as the split was per FILE they were worse than
   useless: N copies of one recording scattered that speaker across train, validation
   and testing, and mWW SELECTS the weights it ships on validation
   average_viable_recall. That obstacle is gone: train/mww/features.py's group_partition
   splits by the identity of the underlying recording, so copies and vocal-tract
   variants of one utterance always land together. What 10x is worth here is a
   measured question, not a settled one, and the default stays at 1 until a
   leak-free run says what the openWakeWord measurement said there.
   What also ported from the oww clean-detection work is --real-vtlp:
   formant-shifted copies of the named speakers' real clips. A shifted wav is a DISTINCT
   feature row, not another draw of the same voice, so it buys the diversity the weak
   voice needs without the cost raw repetition pays - repetition buys that voice
   presence at the cost of diluting the voices that were already detected. The variants
   land here as new rows, not dilution - and they never needed the split fix, which is
   why this is what ported first.

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
   engines differ in timbre, not in speed or text. Both URLs are tcp://
   tts-protocol servers (src/tts-service/): on a Mac that is the in-process
   kokoro-mlx engine (`uv run --project src/tts-service/engines/kokoro_mlx
   python -m kokoro_mlx_engine`), in Docker the wrapped Kokoro-FastAPI service.

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
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# The import root src/. This package sits at src/train/mww/, two levels
# below it, so src/ is two levels up, not the git root the pre-reorg value
# (parents[2]) points at.
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
from train.corpus.real import (  # noqa: E402
    balanced_copy_weights, copy_real_samples, parse_balance_spec,
    speaker_clip_counts,
)
from train.corpus import manifest as corpus_manifest  # noqa: E402
from wordlists import exclude_voice_holdout, load_voice_holdout  # noqa: E402
from wordlists import path_for, voice_holdout_path  # noqa: E402


def _parse_real_vtlp(spec: str, samples_dir, flag: str = "--real-vtlp") -> dict:
    """Parse 'speaker=N[,speaker=N]' into {speaker: N}, failing loud.

    The mirror of _parse_real_copies_override in train/oww/train.py, with the
    samples tree as a parameter instead of a module constant: this module's
    recordings live at --real-samples, and the name must be validated against
    the tree copy_real_samples actually reads, so the check and the copy
    cannot drift apart. A typo'd speaker name would otherwise be silently
    inert (the dict just never matches inside copy_real_samples) and the
    corpus would be filed under a shaping that claims shifted variants it
    does not carry - the label/config drift class this repo has paid for
    twice (the oww-side parser docstring carries the history).
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
    p.add_argument("--wake-word", default="hey seeree")
    p.add_argument("--piper-url", default=os.environ.get("PIPER_URL", "tcp://127.0.0.1:8898"),
                   help="Piper protocol server(s), tcp:// URLs, comma-separated "
                        "to run a fleet (sharded by VOICE - every model pinned "
                        "to one instance for the run, so an instance never "
                        "reloads a model per request; scripts/start-tts-fleet.sh "
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
    p.add_argument("--real-samples", default="data/recordings/samples")
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

    if args.seed:
        # BEFORE any draw below: the whole stage must be a function of the seed,
        # not of the clock. numpy carries the corpus helpers' draws (piper speed
        # choice, child-stretch draws, trim jitter); random is seeded too because
        # the helpers are allowed to use either.
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

    # The real-clip shifted variants (above): parsed BEFORE the voice probes
    # and the --skip decision. A speaker name that is not a directory under
    # --real-samples must fail loud on a reuse run too - an inert override
    # filed under a shaping that claims variants the corpus does not carry is
    # the drift the reuse check exists to refuse - and the parsed dict is part
    # of the requested shaping that check diffs against the manifest.
    real_vtlp = _parse_real_vtlp(args.real_vtlp, args.real_samples)

    # VOICE SET: resolved BEFORE the --skip decision, from the same code path
    # the build below uses - one computation, so the check and the build cannot
    # drift apart. The cf9c065b reuse, 2026-09-22 (oww side) showed the hole
    # this mirrors: every shaping flag matched, but the check never compared
    # the voice set, so a post-reservation run reused a pre-reservation corpus
    # and trained on seven held-out voices. The probes are cheap catalog
    # fetches (no rendering, no model load), but they DO need the fleet up -
    # a --skip run no longer works with the engines down, because an
    # unverifiable catalog is exactly the reuse the check exists to refuse.

    # The voice holdout (improvement.md P1.2): loaded once here, enforced below
    # against whichever engine is actually in play - the live catalog is the
    # source of truth, so a list that drifted from it fails loudly instead of
    # silently excluding nothing. No tracked file (a checkout predating it) is
    # a no-op, and says so.
    holdout = load_voice_holdout()
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
            # Same fail-loudly rule as the openWakeWord side: a holdout pair the
            # audited selection no longer carries (dropped by the audit tables,
            # or gone from the catalog) cannot serve as an eval voice either,
            # so the tracked list must move, not the audit.
            n_before = len(voices)
            voices, holdout_missing = exclude_voice_holdout("piper", voices, holdout)
            if holdout_missing:
                sys.exit(f"  ERROR: the voice holdout ({voice_holdout_path()}) names "
                         f"Piper (voice, speaker) pair(s) the live audited "
                         f"selection does not carry: {holdout_missing}. Update the "
                         f"tracked list to match the catalog this corpus is "
                         f"built from.")
            print(f"  Excluding {n_before - len(voices)} voice-holdout (voice, speaker) "
                  f"pair(s) reserved for the synthetic ranking set")
        else:
            print(f"  NOTE: no voice holdout at {voice_holdout_path()} - the "
                  f"synthetic ranking set has no reserved voices")

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
        if holdout.get("kokoro"):
            n_before = len(kokoro_voices)
            kokoro_voices, holdout_missing = exclude_voice_holdout(
                "kokoro", kokoro_voices, holdout)
            if holdout_missing:
                sys.exit(f"  ERROR: the voice holdout ({voice_holdout_path()}) names "
                         f"Kokoro voice(s) the live catalog does not offer: "
                         f"{holdout_missing}. Update the tracked list.")
            print(f"  Excluding {n_before - len(kokoro_voices)} voice-holdout "
                  f"voice(s) reserved for the synthetic ranking set: "
                  f"{', '.join(holdout.get('kokoro') or [])}")
        if not kokoro_voices:
            sys.exit("  no usable Kokoro voices - re-run with "
                     "--kokoro-fraction 0 (all Piper)")

    # What a --skip run validates against, and what the manifest records at
    # the end of a build: the requested shaping flags plus the TOP-LEVEL
    # voice set (manifest["voices"]), a different axis from every flag -
    # piper entries are (voice, speaker) pairs, the same shape select_piper_
    # voices returns and the manifest stores.
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

    # REUSE MODE: the sweep's front door. Decided AFTER the probes, because
    # the check diffs the voice set the probes just resolved (module: the
    # cf9c065b reuse was blind on exactly that axis).
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
    generate_piper_samples(args.piper_url, voices, positives,
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
    generate_piper_samples(args.piper_url, voices, negatives,
                           args.negatives_per_voice, phrases,
                           PLAIN_SPEED_GRID, "Piper negatives")

    # Before the real clips, so only synthetic output is shifted - and before
    # trimming, so the shifted copies are trimmed like everything else. Same order
    # as train.py, for the same reasons.
    if args.child_fraction > 0:
        print("\n[Child-range copies]")
        add_child_range_copies(positives, "VTLP positives", args.child_fraction)

    print("\n[Real Voice]")
    # --real-vtlp (the mww port of the oww clean-detection lever):
    # shifted copies of the named speakers' real clips. They transfer cleanly
    # here for the same reason they won on the oww side: a shifted wav is a NEW
    # acoustic variant, and in this pipeline it is a DISTINCT feature row, not
    # another draw of one voice (module docstring, point 1).
    #
    # NOTE - what does NOT port is the oww-side PER-SPEAKER raw-copy weight
    # (--real-copies-override), and this module deliberately has no equivalent. The
    # reason is no longer the split: train/mww/features.py's group_partition keeps every
    # copy and every shifted variant of a recording in one split, which is what made
    # --real-copies safe to raise here at all (see corpus/real.py's NOTE FOR THE
    # microWakeWord PORT, rewritten to that effect). What remains is
    # granularity: the honest sampling knob (sampling_weight, config.py) is ONE number
    # per FEATURE SET, and the positives are one directory holding synthetic and real
    # clips together - so it cannot aim at one speaker. Per-speaker row weights would
    # need upstream microwakeword support; the pinned clone has none. Per-speaker
    # DIVERSITY is available, and is what --real-vtlp below consumes.
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

    # FREEZE THE CORPUS: the manifest is the identity the run tag hashes and the
    # check a later `--corpus reuse` (or the sweep runner) validates against. It
    # is written last, after trimming and the child copies, so its digest covers
    # the final tree - the state training will actually consume.
    corpus_manifest.write_manifest(
        root, args.wake_word, "mww",
        seed=args.seed,
        # The REQUESTED shaping (what a later --skip reuse check diffs) plus the
        # holdout list, so the manifest names the exclusion: the voice set the
        # manifest records is already holdout-free, and the list says why.
        shaping={
            "samples_per_voice": args.samples_per_voice,
            "negatives_per_voice": args.negatives_per_voice,
            "kokoro_fraction": args.kokoro_fraction,
            "child_fraction": args.child_fraction,
            "real_copies": args.real_copies,
            # The parsed dict, not the raw string: the --skip check compares
            # structure, and a manifest that recorded "speakerA=3" as a string
            # could not be diffed against a later request parsed to {speakerA: 3}.
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
        wordlist_path=path_for(args.wake_word),
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
