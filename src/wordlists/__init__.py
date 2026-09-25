"""Load and validate the per-wake-word phrase lists.

THE WORDLISTS ARE THE PART OF THIS PIPELINE THAT DOES NOT TRANSFER. Everything else
- the trainers, the augmentation, the gates, the eval harness - works unchanged for
any wake word. The phrases do not: "hey serious" probes the decision boundary of
"hey seeree" and says nothing at all about "okay jarvis". Leaving them hardcoded in
generate_negatives.py made the repo look general while quietly being about one phrase.

So they live here, one YAML file per wake word, and `.claude/skills/write-wordlists`
is how a new one gets written.

WHAT THIS MODULE EXISTS TO ENFORCE, beyond parsing:

* THE EVAL AND TRAINING LISTS MUST BE DISJOINT. The false-accept gates are scored on
  the eval corpus. A phrase that appears in both is trained on and then measured,
  which turns a generalisation measurement into a memorisation one - and the number
  moves in the direction that looks like success. Three separate comments in
  train/corpus/negatives.py ask the reader to check this by hand before adding a
  phrase; `validate` does it instead.

* A PHRASE MUST NOT BE THE WAKE WORD. A "negative" identical to the positive teaches
  the two classes the same audio.

* DUPLICATES ARE DROPPED, NOT COUNTED TWICE. Category rates are read as
  fired/n, so a phrase appearing twice in one category quietly reweights it.

Both trainers and the eval harness read this, so it imports nothing from either.

* THE TRAINING CORPORA AND THE SYNTHETIC EVAL POSITIVES MUST NOT SHARE A VOICE.
  The phrase rule above keeps the two corpora from memorising each other's TEXT;
  `voice_holdout.yaml` (the second file in this directory) keeps them from
  sharing TIMBRE. The training corpora build from every usable engine voice, so
  an eval positive rendered from one of those voices is inside the training
  distribution no matter how novel its phrasing - and the only axis left to
  generalise on, cheaply and with a real n, is the voice. The holdout list is
  a tracked file beside the wordlists because a list that lives only in a
  comment drifts: the trainers exclude it from the live catalog and fail when
  an entry the catalog no longer offers, which is what pins it.
"""

from pathlib import Path

import yaml

# The GIT root: the YAML files beside this module moved with it under src/,
# but the data/ and output/ trees this module's callers anchor on stay at
# the git root, which is one level up.
REPO_ROOT = Path(__file__).resolve().parents[2]
WORDLIST_DIR = Path(__file__).resolve().parent

# The eval corpus categories, and what each one is FOR. Read as documentation for
# whoever writes the next wordlist: a category is a question about the model, not a
# bag of phrases, and the answer is only meaningful per category (see
# eval/src/generate_negatives.py on why a pooled false-accept rate means nothing).
EVAL_CATEGORIES = {
    "extend":    "the phrase, then the word keeps going ('hey serious')",
    "running":   "the wake word's sounds inside ordinary speech, with no 'hey'",
    "hey_other": "the carrier word plus a different name ('hey Sarah')",
    "command":   "bare commands, to check speech alone cannot trip it",
    "other_ww":  "other assistants' wake words",
    "general":   "ordinary conversation, for a baseline false-accept rate",
}

# The categories that must be rewritten for a new wake word, because they are built
# out of that phrase's own consonants and vowels. The rest are reusable as they are:
# ordinary speech is ordinary speech whatever the wake word is.
PHRASE_SPECIFIC = ("extend", "running", "hey_other")


class WordlistError(Exception):
    """Raised for a wordlist that would produce a misleading measurement."""


def path_for(wake_word):
    """wordlists/<safe_wake_word>.yaml - the same slug the rest of the repo uses."""
    return WORDLIST_DIR / f"{wake_word.replace(' ', '_').lower()}.yaml"


def load(wake_word=None, path=None, validate_lists=True):
    """Return the parsed wordlist for a wake word.

    Raises rather than falling back to a default: a silently-empty category reads as
    "the model never false-accepts here", which is the most expensive wrong answer
    this harness can give.
    """
    if path is None:
        if wake_word is None:
            raise WordlistError("need either a wake word or a path")
        path = path_for(wake_word)
    path = Path(path)
    if not path.is_file():
        available = sorted(p.stem for p in WORDLIST_DIR.glob("*.yaml"))
        raise WordlistError(
            f"no wordlist at {path}. Available: {', '.join(available) or '(none)'}. "
            f"Write one with the `write-wordlists` skill - the phrases are specific "
            f"to the wake word and cannot be defaulted.")

    data = yaml.safe_load(path.read_text()) or {}
    data["_path"] = path
    if validate_lists:
        problems = validate(data)
        if problems:
            raise WordlistError(
                f"{path} would produce a misleading measurement:\n  "
                + "\n  ".join(problems))
    return data


def eval_categories(data, names=None):
    """{category: [phrase, ...]} for the eval corpus, de-duplicated, order kept."""
    section = data.get("eval") or {}
    out = {}
    for name in (names or EVAL_CATEGORIES):
        seen, phrases = set(), []
        for phrase in section.get(name) or []:
            key = phrase.strip().lower()
            if key and key not in seen:
                seen.add(key)
                phrases.append(phrase.strip())
        out[name] = phrases
    return out


def validate(data):
    """Return a list of problems; empty means the wordlist is usable."""
    problems = []
    wake_word = (data.get("wake_word") or "").strip()
    if not wake_word:
        problems.append("no `wake_word:` at the top level")

    section = data.get("eval")
    if not isinstance(section, dict):
        problems.append("no `eval:` section")
        return problems

    unknown = set(section) - set(EVAL_CATEGORIES)
    if unknown:
        problems.append(
            f"unknown eval categories {sorted(unknown)}; the harness reports per "
            f"category and only knows {sorted(EVAL_CATEGORIES)}")

    for name in EVAL_CATEGORIES:
        phrases = section.get(name)
        if not phrases:
            problems.append(f"eval.{name} is empty - "
                            f"{EVAL_CATEGORIES[name]} would go unmeasured")

    # A phrase that is the wake word itself is a positive, whatever list it sits in.
    lowered_wake = wake_word.lower()
    for name, phrases in (section or {}).items():
        for phrase in phrases or []:
            if phrase.strip().lower() == lowered_wake:
                problems.append(f"eval.{name} contains the wake word itself: "
                                f"{phrase!r} - that is a positive, not a negative")

    # Across categories, because the harness reports each as its own rate and a
    # phrase counted in two of them is weighted twice in the pooled view.
    seen = {}
    for name, phrases in (section or {}).items():
        for phrase in phrases or []:
            key = phrase.strip().lower()
            if key in seen and seen[key] != name:
                problems.append(f"{phrase!r} is in both eval.{seen[key]} and "
                                f"eval.{name}")
            seen[key] = name

    problems.extend(_disjointness_problems(data, seen))
    return problems


# ---------------------------------------------------------------------------
# Voice holdout (improvement.md P1.2)
# ---------------------------------------------------------------------------

# One file for the whole REPO, not one per wake word: a voice is an engine
# property, not a property of the phrase it renders, and the same reserved
# voice set serves every wake word this machine trains.
VOICE_HOLDOUT_NAME = "voice_holdout.yaml"


def voice_holdout_path():
    """wordlists/voice_holdout.yaml - the tracked voice-holdout list."""
    return WORDLIST_DIR / VOICE_HOLDOUT_NAME


def load_voice_holdout(path=None):
    """The voices reserved OUT OF every corpus build.

    Returns {"kokoro": [voice, ...], "piper": [(voice, speaker-or-None), ...]}.
    """
    if path is None:
        path = voice_holdout_path()
    path = Path(path)
    if not path.is_file():
        # No tracked file: the exclusion is a no-op, the same way an empty
        # entry in the per-wake-word mispronunciation tables is. Deliberately
        # not an error - a checkout of a fresh wake word predating this file
        # must still train. The trainers print that they ran with no holdout.
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    kokoro = [str(v).strip() for v in (data.get("kokoro") or []) if str(v).strip()]
    piper = []
    for entry in data.get("piper") or []:
        entry = str(entry).strip()
        if not entry:
            continue
        # The "voice:speaker" spelling of the exclusion tables in
        # train/corpus/piper.py: a bare name is a single-speaker model.
        voice, _sep, speaker = entry.partition(":")
        piper.append((voice.strip(), speaker.strip() or None))
    return {"kokoro": kokoro, "piper": piper}


def exclude_voice_holdout(engine, catalog, holdout=None):
    """Drop the holdout entries from a live engine catalog -> (kept, missing).

    `engine` is "kokoro" (catalog = voice strings) or "piper" (catalog =
    (voice, speaker-or-None) pairs); `holdout` is a load_voice_holdout()
    result (loaded from the tracked file when not given).

    `missing` is the failure to read: a holdout entry the catalog does not
    offer means the tracked list has drifted from the engine, and the caller
    must exit. The catalog is the source of truth - a stale list is an error,
    not a silent skip, because the silent outcome is the bad one: the
    exclusion ends up empty and the corpus quietly trains on a voice that is
    supposed to be held out, which converts the ranking set into another
    training-distribution measurement with a label that says otherwise.

    A bare voice entry (speaker None) on the piper side removes EVERY speaker
    of that model, matching the is_excluded convention in
    train/corpus/piper.py; a "voice:speaker" entry removes just that pair.
    """
    if holdout is None:
        holdout = load_voice_holdout()
    entries = holdout.get(engine) or []
    catalog = list(catalog)
    if not entries:
        return catalog, []

    # Normalise both sides to (voice, speaker-or-None) pairs: piper entries
    # arrive as tuples from load_voice_holdout, kokoro as bare strings.
    pairs = [v if isinstance(v, tuple) else (v, None) for v in catalog]
    norm = [e if isinstance(e, tuple) else (e, None) for e in entries]
    missing = []
    for entry, original in zip(norm, entries):
        hits = [i for i, (cv, cs) in enumerate(pairs)
                if cv == entry[0] and (entry[1] is None or entry[1] == cs)]
        if not hits:
            missing.append(original)
        else:
            drop = set(hits)
            pairs = [p for i, p in enumerate(pairs) if i not in drop]
    kept = [p[0] for p in pairs] if engine == "kokoro" else pairs
    return kept, missing


def _disjointness_problems(data, eval_phrases):
    """The eval corpus and the training corpus must not share a phrase.

    Only checked when the wordlist carries a `train:` section. The trainer still
    keeps its lists in train/corpus/negatives.py, so today this catches nothing -
    it is here so that migrating those lists into this file cannot reintroduce the
    overlap the comments over there warn about three times.
    """
    section = data.get("train")
    if not isinstance(section, dict):
        return []
    problems = []
    for name, phrases in section.items():
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            if not isinstance(phrase, str):
                continue
            key = phrase.strip().lower()
            if key in eval_phrases:
                problems.append(
                    f"{phrase!r} is in train.{name} AND eval.{eval_phrases[key]} - "
                    f"training on a phrase the gates are scored on measures "
                    f"memorisation, not generalisation")
    return problems
