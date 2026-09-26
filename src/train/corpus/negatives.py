"""The negative wordlist: what the model must learn to reject.

Engine-agnostic because it is text - the phrases are rendered to WAVs by whichever
TTS the trainer uses, and both trainers need the same list for the same measured
reason.

Kept deliberately DISJOINT from the eval corpus in generate_negatives.py. The
false-accept gates in the tuning log are scored on that corpus, so a phrase appearing
in both would turn a generalisation measurement into a memorisation one. This module
used to ask the reader to check that by hand before adding a phrase; the phrases now
live in the wake word's wordlist, where wordlists.validate() checks it.
"""

import sys
from pathlib import Path

# The per-word tables - confusable phrases and the voices that render this phrase
# wrong - live in src/wordlists/<word>.yaml, not here. This module keeps only the
# word-agnostic lists: a wake-word-keyed dict in a module every wake word shares
# makes a new phrase a code edit, and skipping that edit fails silently.
import wordlists

# Negatives that are useful whatever the wake word is: ordinary openers, and the
# wake words of other assistants.
BASE_NEGATIVES = [
    "hello", "hi there", "good morning", "excuse me", "okay",
    "hey google", "alexa", "hey jarvis", "computer",
]

# Commands used two ways: appended to the wake word to build run-on positives
# ("hey seeree what's the time"), and rendered on their own as negatives.
#
# Both halves are needed. The positives teach that the phrase can be followed
# immediately by speech; without the matching negatives the model can learn the
# shortcut "speech after ~ wake word" instead, since in training every clip with
# trailing speech would be positive.
#
# Deliberately disjoint from generate_negatives.py's COMMAND list, which
# generate_positives.py also uses for its cmd_run/cmd_pause sweeps - those are the
# eval corpus, and training on them would turn that measurement into memorisation.
TRAINING_COMMANDS = [
    "open the garage door", "how cold is it outside", "start the kettle",
    "find my phone", "skip this song", "dim the bedroom lights",
    "how long is left on the timer", "put the heating on", "read my messages",
    "lock the back door", "what is on tonight", "call the office",
]

# Voices that mispronounce the wake word are per-word data and live in the
# wordlist: `voices.kokoro.mispronouncing` in src/wordlists/<word>.yaml, read
# through wordlists.voice_exclusions(). The rationale that used to sit here - how
# the six were judged, why they are excluded from the negatives as well as the
# positives, and why duration is not a usable proxy for listening - moved with
# them, so it is beside the list it explains.
LEGACY_VOICE_MARKER = "_v0"
"""Kokoro's v0 voices, skipped by default: they cost generation time and buy nothing.

Kokoro-FastAPI serves 42 English voices; 13 carry this marker and are OLDER
RENDERINGS OF SPEAKERS ALREADY IN THE SET - af_v0bella beside af_bella, am_v0michael
beside am_michael, bf_v0emma beside bf_emma. Not 13 additional speakers, which is
what a raw voice count suggests and what made them look load-bearing.

MEASURED, not assumed. Two openWakeWord corpora, same engine and trainer, differing
only in whether the v0 voices were included, scored on the same held-out recordings
at 4 adversarial false accepts:

    36 voices, v0 included     plain 76%   run-on 56%
    22 voices, v0 excluded     plain 82%   run-on 65%

So excluding them did not cost accuracy - slightly ahead at every matched false-accept
point, though several gaps sit near the ~10 point run-to-run variance this repo has
measured, so the honest claim is "no measurable loss".

What it definitely buys is time: 13 of 36 voices is 36% of the Kokoro clips, and the
corpus stage is the largest in an openWakeWord run.

THE MEASURED RUN USED 22 VOICES, THIS FILTER LEAVES 23. That run also dropped
af_jadzia, to match the voice set kokoro-mlx offers for an engine comparison.
af_jadzia is a genuine distinct speaker rather than a v0 duplicate, so it is kept
here - the filter drops legacy renderings, not voices MLX happens to lack.

kokoro-mlx does not serve them at all, which is why its 28-voice set is not the
handicap it first appears - see src/tts-service/engines/kokoro_mlx/.

--include-legacy-voices puts them back, for reproducing a pre-2026-09 corpus.
"""

# Both per-word tables that used to live here - MISPRONOUNCING_VOICES and
# CONFUSABLE_NEGATIVES, each a dict keyed by wake word - moved to
# src/wordlists/<word>.yaml, as `voices.kokoro.mispronouncing` and
# `train.confusable`. The rationale moved with them, so the measurements that
# justified each entry sit beside the entries instead of in a module every wake
# word shares, where the next word's author has to work out which parts transfer.


def load_wordlist_or_exit(wake_word: str) -> dict:
    """wordlists.load(wake_word), or its actionable message and exit(1).

    Shared by every corpus entry point that needs the per-word tables, so the
    failure reads the same whichever stage hits it first: no wordlist for this
    wake word, or one that would produce a misleading measurement. Exiting rather
    than raising is the convention of the modules around this one - they are CLI
    stages, and a traceback for a missing data file hides the instruction that
    wordlists already wrote ("Write one with the `write-wordlists` skill").
    """
    try:
        return wordlists.load(wake_word)
    except wordlists.WordlistError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


def build_negative_phrases(wake_word: str, negatives_file: str = None,
                           with_commands: bool = True) -> list:
    """Assemble the negative wordlist: base phrases plus confusables.

    Confusables come from --negatives-file if given, otherwise from the wake
    word's own wordlist (`train.confusable` in src/wordlists/<word>.yaml).
    Training without any is the single biggest measured cause of false accepts,
    so it warns rather than proceeding quietly.

    A word with NO wordlist is an error rather than a warning: the wordlist is
    step 1 of the pipeline and the eval harness cannot run without one either, so
    a corpus built for a word that has none is a corpus that cannot be measured.
    --negatives-file stays as the escape hatch for a list kept elsewhere.
    """
    phrases = list(BASE_NEGATIVES)

    # The commands that appear after the wake word in the run-on positives, here on
    # their own. Without them every clip containing trailing command speech would be
    # a positive, and "speech after" is a far easier feature to learn than the wake
    # word itself.
    if with_commands:
        phrases += TRAINING_COMMANDS

    if negatives_file:
        path = Path(negatives_file)
        if not path.exists():
            print(f"ERROR: negatives file not found: {path}")
            sys.exit(1)
        confusables = [line.strip() for line in path.read_text().splitlines()]
        confusables = [p for p in confusables if p and not p.startswith("#")]
        print(f"  Confusable negatives: {len(confusables)} from {path}")
    else:
        data = load_wordlist_or_exit(wake_word)
        confusables = wordlists.train_phrases(data)
        name = Path(data["_path"]).name
        if confusables:
            print(f"  Confusable negatives: {len(confusables)} from {name} "
                  f"for '{wake_word}'")
        else:
            print(f"  WARNING: {name} carries no train.confusable phrases.")
            print("           The model will reject what it is shown here and fire on")
            print("           anything adjacent to the wake word. Write them with the")
            print("           `write-wordlists` skill, or pass --negatives-file.")

    # A confusable that is also a positive text would teach the two classes the
    # same clip; cheap to check, expensive to debug.
    positives = {wake_word.lower()}
    duplicates = [p for p in confusables if p.lower() in positives]
    if duplicates:
        print(f"ERROR: these negatives are the wake word itself: {duplicates}")
        sys.exit(1)

    seen, phrases = set(), phrases + confusables
    return [p for p in phrases if not (p.lower() in seen or seen.add(p.lower()))]
