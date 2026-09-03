"""The negative wordlist: what the model must learn to reject.

Moved verbatim from train.py. This is engine-agnostic because it is text - the
phrases are rendered to WAVs by whichever TTS the trainer uses, and both trainers
need the same list for the same measured reason.

Kept deliberately DISJOINT from the eval corpus in generate_negatives.py. The
false-accept gates in tuning.md are scored on that corpus, so a phrase appearing in
both would turn a generalisation measurement into a memorisation one. Check any new
phrase against EXTEND/RUNNING/HEY_OTHER over there before adding it here.
"""

import sys
from pathlib import Path

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

# Voices that mispronounce the wake word, per wake word.
#
# A wake word worth having is not a dictionary word, so Kokoro's g2p has to guess at
# it - and some voices guess differently. These six say something that is not "hey
# seeree", judged by ear over all 42 English voices rendering the phrase once
# (vtlp_demo/voices/). Every clip such a voice produces is a mislabelled positive,
# and at 1/42 of the voice list that is ~2.4% of the Kokoro corpus each, ~14% for
# the six together - across plain AND run-on, since both draw from this list.
#
# Keyed per wake word: how a voice handles "seeree" says nothing about how it would
# handle another phrase, so a global blocklist would be wrong for the next model.
# Same reasoning as CONFUSABLE_NEGATIVES.
#
# HOW TO REBUILD THIS FOR A NEW WAKE WORD: render every voice saying the phrase once
# and listen to all of them. It takes a couple of minutes and there is no shortcut -
# duration does not work as a proxy. bm_fable sits at exactly the median length
# (1121 ms, 1.00x) and is wrong; af_v0sky is 16% below median and is fine. The same
# proxy also cleared "HEY SEEREE" as merely emphatic when it was spelled out.
# audit_voices.py automates the screen; it does not replace the listening.
#
# Excluded from negatives too, not just positives. A mispronunciation is arguably a
# useful near-miss to train against, but it is much closer to the real phrase than
# CONFUSABLE_NEGATIVES entries are, and teaching the model to REJECT something that
# close risks costing detection on genuine variants. Untested either way.
#
# NOTE: these are Kokoro voice ids. The Piper equivalent is a separate list - Piper
# phonemises with espeak-ng per MODEL rather than per speaker, so the unit being
# excluded is a whole voice model, not a speaker within one (audit_voices.py:148).
MISPRONOUNCING_VOICES = {
    "hey_seeree": [
        "af_alloy", "am_echo", "bf_alice", "bf_lily", "bm_daniel", "bm_fable",
    ],
}

# Confusable negatives, per wake word.
#
# A model trained only on BASE_NEGATIVES rejects exactly what it was shown and
# nothing adjacent: hey_seeree.onnx scored 0/8 on other assistants and 0/36 on
# general conversation, but 13/20 on the phrase continuing into another word
# ("hey serious" -> 0.995) and 5/12 on "hey" plus a different name. Those two
# categories are the entire false-accept problem, and neither was in the wordlist.
#
# Three shapes matter, and all three want the wake word's own consonants:
#   - the phrase, continuing into a different word ("hey Serena", "hey season")
#   - "hey" attached to some other name ("hey Sienna", "hey Cynthia")
#   - the same sounds inside running speech, with no "hey" at all
# Bare "hey" belongs here too: it is what teaches that the second syllable is
# required rather than optional.
#
# These are deliberately DISJOINT from the eval corpus in generate_negatives.py.
# The gates in tuning.md are scored on that corpus, so any phrase appearing in
# both turns a generalisation measurement into a memorisation one. When adding
# phrases here, check them against EXTEND/RUNNING/HEY_OTHER over there first.
CONFUSABLE_NEGATIVES = {
    "hey_seeree": [
        # the phrase, continuing into another word
        "hey Serena", "hey serene", "hey serenade", "hey Syria", "hey syringe",
        "hey sincere", "hey sincerely", "hey severe", "hey season",
        "hey seasoning", "hey seizure", "hey ceases", "hey scenery",
        "hey scenario", "hey CEO", "hey seatbelt", "hey sedan",
        "hey ceremony", "hey sequin", "hey search for it",
        # "hey" plus another name, and "hey" on its own
        "hey Sienna", "hey Selena", "hey Sirena", "hey Cerys", "hey Cynthia",
        "hey Sabrina", "hey Sylvia", "hey Simon", "hey Sadie", "hey Cecil",
        "hey", "hey, come here a minute",
        # the same sounds in running speech, with no "hey"
        "The scenery on the coast road is worth the detour.",
        "She was sincere about wanting to see the city again.",
        "Season the sauce properly before you serve it.",
        "Serena said she would meet us down by the seafront.",
        "The ceremony starts at three and runs for about an hour.",
        "It has been a severe winter by any measure.",
    ],
}


def build_negative_phrases(wake_word: str, negatives_file: str = None,
                           with_commands: bool = True) -> list:
    """Assemble the negative wordlist: base phrases plus confusables.

    Confusables come from --negatives-file if given, otherwise from
    CONFUSABLE_NEGATIVES for this wake word. Training without any is the single
    biggest measured cause of false accepts, so it warns rather than proceeding
    quietly.
    """
    safe_name = wake_word.replace(" ", "_").lower()
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
    elif safe_name in CONFUSABLE_NEGATIVES:
        confusables = list(CONFUSABLE_NEGATIVES[safe_name])
        print(f"  Confusable negatives: {len(confusables)} built in for '{safe_name}'")
    else:
        confusables = []
        print(f"  WARNING: no confusable negatives for '{safe_name}'.")
        print("           The model will reject what it is shown here and fire on")
        print("           anything adjacent to the wake word. Add an entry to")
        print("           CONFUSABLE_NEGATIVES or pass --negatives-file.")

    # A confusable that is also a positive text would teach the two classes the
    # same clip; cheap to check, expensive to debug.
    positives = {wake_word.lower()}
    duplicates = [p for p in confusables if p.lower() in positives]
    if duplicates:
        print(f"ERROR: these negatives are the wake word itself: {duplicates}")
        sys.exit(1)

    seen, phrases = set(), phrases + confusables
    return [p for p in phrases if not (p.lower() in seen or seen.add(p.lower()))]
