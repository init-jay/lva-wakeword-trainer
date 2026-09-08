---
name: write-wordlists
description: Write the per-wake-word phrase lists this pipeline needs - the adversarial evaluation negatives (extend, running, hey_other) and the training confusables and run-on commands. Use when the user picks a new wake word, adds a language or accent, sees false accepts on a particular kind of phrase, or asks why a model fires on something that only sounds like the wake word.
---

# Writing wordlists for a wake word

Everything else in this pipeline transfers between wake words. **The phrases do not.**
"hey serious" probes the decision boundary of "hey seeree" and says nothing whatever
about "okay jarvis". A new wake word needs new phrases before any measurement of it
means anything.

You are generating these. It is a language task, not a code task — which is why the
pipeline can be handed a wake word and produce its own adversarial corpus.

## The one thing to understand before writing anything

A wake-word model that is already quiet on ordinary speech scores **zero** on random
sentences. A hundred of them measure nothing. Every phrase you write must probe the
decision the model actually makes, and there are only three shapes that do:

| Shape | What it catches |
|---|---|
| the phrase, then the word keeps going | the model learned "starts like the phrase" |
| the carrier word plus a different name | the model learned the carrier word is enough |
| the same sounds inside running speech | the model learned one distinctive syllable |

Measured on this repo's own model: `extend` fired **12/20** while `general` fired
**0/36**. The adversarial shapes are the entire false-accept problem. Sentences about
train timetables contribute nothing but a denominator.

## Method

**1 · Break the wake word into its sounds, not its spelling.** "hey seeree" is
`HEY` + `S-EE-R-EE`. What matters is what the phrase *sounds like* to a model that has
never seen it written. Spelling is a trap: "cereal", "Sirius" and "searing" all begin
with the same sound as "seeree" and none of them share its letters.

**2 · For `extend`, find real words that begin with those sounds and then diverge.**
Aim for 20. They should be words a person might actually say — a model that rejects
nonsense but fires on "hey serious" has not been tested. Prefer the ones that diverge
*late*: "hey serious" is a harder negative than "hey sandwich" because it stays
identical for longer.

**3 · For `hey_other`, take the carrier word and attach different names.** Aim for 12.
The hard ones start with the wake word's second syllable — for "seeree": Sarah, Cindy,
Sydney, Cecily. Include two or three bare openers ("hey there, how are you?") so the
easy case is covered too.

**4 · For `running`, write full sentences containing those sounds with no carrier
word.** Aim for 12. Full sentences, not fragments — the surrounding speech is what
makes it a realistic test. This is the category that checks the model needs the *whole*
phrase rather than its most distinctive syllable.

**5 · Leave `command`, `other_ww` and `general` alone.** Ordinary speech is ordinary
speech whatever the wake word is. Copy them from an existing wordlist. If a `general`
sentence happens to share sounds with the new wake word, move it to `running` — that is
where it now belongs.

## Where it goes

`wordlists/<wake_word>.yaml`, with underscores and lowercase — `wordlists/okay_jarvis.yaml`.
Copy `wordlists/hey_seeree.yaml` as the worked example; its comments explain each
category in place. Then:

```bash
cd eval
docker compose run --rm eval python -m eval.generate_negatives \
    --wake-word "okay jarvis" --dry-run
```

`--dry-run` validates and prints what would be rendered without calling the TTS
server. Fix anything it reports before generating audio.

## What validation will reject

`wordlists/__init__.py` checks these because each one produces a *misleading number*
rather than an error:

- **An empty category.** Reads as "the model never false-accepts here."
- **A phrase that is the wake word.** That is a positive, whatever list it sits in.
- **The same phrase in two categories.** Rates are read per category as fired/n.
- **A phrase in both `eval:` and `train:`.** See below — this is the important one.

## The disjointness rule

**The evaluation phrases and the training phrases must never overlap.** The
false-accept gates are scored on the eval corpus. A phrase that is both trained on and
measured on turns a generalisation measurement into a memorisation one — and it moves
the number in the direction that looks like success, so nothing downstream catches it.

When you write training confusables for a wake word, they must be *different phrases
probing the same boundary* as the eval ones. Not harder, not easier — different. If you
write "hey serious" for eval, write "hey Serena" for training.

## The training side

Same method, different destination. The trainer's lists currently live as Python
constants in `train/corpus/negatives.py`, keyed by wake word:

- **`CONFUSABLE_NEGATIVES`** — the same three adversarial shapes as above. This is the
  single biggest measured cause of false accepts: a model trained without them scored
  0/8 on other assistants and 0/36 on general speech, but **13/20** on the phrase
  continuing into another word. Aim for ~35, split roughly 20 / 12 / 4 across the three
  shapes, and include bare `"hey"` — that is what teaches the second syllable is
  required rather than optional.
- **`TRAINING_COMMANDS`** — ordinary commands, used twice over: appended to the wake
  word to build run-on positives, and rendered alone as negatives. Both halves are
  needed. Without the negatives the model can learn "speech after ≈ wake word", because
  in training every clip with trailing speech would otherwise be positive. Aim for 12,
  and keep them disjoint from the eval `command` list.

`wordlists/__init__.py` already validates a `train:` section against `eval:`, so those
constants can move into the YAML whenever the trainer is migrated — the check is
waiting for them.

## One thing you cannot generate

`MISPRONOUNCING_VOICES` in `train/corpus/negatives.py` is per wake word and **must be
found by listening**, not written. Some TTS voices guess the wake word's pronunciation
wrong, and every clip such a voice produces is a mislabelled positive — six voices out
of 42 was ~14% of the corpus. Duration is not a usable proxy: `bm_fable` sits at exactly
the median length and is wrong. `tools/audit_voices.py` narrows the field; it does not
replace the listening. Tell the user this is a manual step rather than producing a list
that looks authoritative.
