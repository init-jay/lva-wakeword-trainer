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
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORDLIST_DIR = Path(__file__).resolve().parent

# The eval corpus categories, and what each one is FOR. Read as documentation for
# whoever writes the next wordlist: a category is a question about the model, not a
# bag of phrases, and the answer is only meaningful per category (see
# eval/generate_negatives.py on why a pooled false-accept rate means nothing).
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
