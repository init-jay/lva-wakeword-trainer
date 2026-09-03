# `train/corpus/` — engine-agnostic corpus construction

Everything here operates on 16 kHz mono WAVs and knows nothing about how those clips
later become features. That is the seam between openWakeWord (melspectrogram →
embedding model → 96-dim embeddings, 2000 ms window) and microWakeWord (40 features
per 10 ms into a streaming MixConv net, 1500 ms clip): shared up to a directory of
WAVs, separate after it. See `../plan.md`.

| module | what it holds |
|---|---|
| `augment.py` | silence trimming, and the child-range pitch/formant copies |
| `negatives.py` | the negative wordlist, and the Kokoro mispronunciation list |
| `real.py` | real recordings into a corpus, weighted by repetition |
| `piper.py` | Piper generation over Wyoming, plus Piper voice metadata |

These were moved out of `train/oww/train.py` without behaviour change — sixteen runs of
`../tuning.md` are calibrated against that behaviour. Anything that looks like it
wants tidying probably encodes a measured result; check the notebook first.

## Auditing TTS voices before you generate a corpus

**This is not optional, and skipping it is the most expensive mistake this repo has
made.** A wake word worth having is not a dictionary word, so a TTS engine's
grapheme-to-phoneme has to guess at it — and voices guess differently. Six of
Kokoro's 42 English voices say something other than "hey seeree". Every clip such a
voice produces is a **mislabelled positive**, and at six voices that was ~14% of the
synthetic corpus, undetected for eleven training runs.

Duration is not a usable proxy: `bm_fable` sits at exactly the median clip length and
is wrong, while `af_v0sky` is 16% below median and is fine.

### Running it

`tools/audit_voices.py` needs only numpy, scipy and requests — it is a network client, and
all the work happens on the TTS and ASR services. **It does not need the CUDA trainer
image**, so run it anywhere, including a laptop:

```bash
.venv-eval/bin/python -m tools.audit_voices --wake-word "hey seeree" --tts piper \
    --piper <piper-host>:10200 --asr <asr-host>:10300 \
    --out-dir voice_audit_piper
```

or with no venv at all:

```bash
uv run --with numpy --with scipy --with requests \
    audit_voices.py --wake-word "hey seeree" --tts piper \
    --piper <piper-host>:10200 --asr <asr-host>:10300 --out-dir voice_audit_piper
```

Kokoro is the same command with `--tts kokoro --kokoro-url http://<host>:8880`.

Two flags to set deliberately:

- `--piper-speakers` (default 12) caps how many speakers of a multi-speaker model get
  sampled. `en_US-libritts_r-medium` alone carries 904; without the cap the audit
  spends all its time inside one model.
- `--repeats` (default 2) matters more for Piper than for Kokoro. VITS samples noise
  and durations per call, so one speaker measured 0% in one pass and 100% in the
  next. A single rendering is not evidence.

`--voices en_GB-alba-medium,en_US-lessac-medium` audits just those two, which is the
quick way to confirm both services are reachable before committing to 84 voices.

### Reading the result

**It is a screen, not a verdict.** It produces a ranked shortlist to check by ear. A
voice at 100% is probably fine; anything below needs listening to before it is
trusted or excluded — the clips are written to `--out-dir` for exactly that.

It cannot tell a mispronunciation from a strong accent. `en_US-l2arctic-medium` is a
non-native-speaker corpus and flags heavily, but accented renderings of the *correct*
phrase are **good** training data, because real users have accents. Listen before
excluding those.

For Piper the score also means something slightly different than for Kokoro. Because
VITS is stochastic, a speaker's score is a *contamination rate* rather than a verdict:
a voice at 67% puts a bad clip in the corpus one time in three, and belongs on the
exclusion list even though it is sometimes fine.

### What to write down afterwards

Two lists, both keyed per wake word — how a voice handles "seeree" says nothing about
how it would handle another phrase.

**1. `MISPRONOUNCING_PIPER_VOICES` in `piper.py`** (or `MISPRONOUNCING_VOICES` in
`negatives.py` for Kokoro). The voices to exclude.

**Key them per SPEAKER, not per model.** The expectation going in was the opposite —
espeak-ng phonemises per model, so every speaker in a voice gets the same phoneme
string, and it seemed to follow that they would all pronounce it alike. The 2026-09-02
audit says otherwise: `en_US-l2arctic-medium` ran from `:ASI` at 0% to `:PNV` at 100%.
Identical phonemes, different acoustic models, and intelligibility varies by speaker.

**2. `PIPER_VOICE_SEX` in `piper.py` — generate it, do not listen for it.**

It drives the child-range lever, the largest single win in the notebook: a 4-year-old
went from 24% detection to 83% once the corpus stopped being adult-only (run 13).
`add_child_range_copies` picks a stretch ratio from the voice's sex, and Piper voice
names carry no sex marker, so a voice missing from this map is written `piper_pu_*`
and receives **no child-range copy at all** — the lever loses reach in proportion to
how much of the corpus is Piper, silently.

Sex here is only a proxy for F0, and F0 is measurable from the clips the audit already
wrote:

```bash
python -m tools.measure_voice_f0 voice_audit_piper/            # table, with the split
python -m tools.measure_voice_f0 voice_audit_piper/ --python   # paste-ready dict body
```

96 entries is more listening than anyone will actually do, which is the real argument
for measuring it. Validate a new run against the voices whose *name* states the
answer — `hfc_male` 147 Hz, `hfc_female` 268 Hz, `northern_english_male` 117 Hz,
`southern_english_female` 248 Hz. All ten such voices agreed with the 185 Hz split
on the run above.

Leaving a voice out is deliberate rather than lazy: run 12 measured male voices as
"useless above R1.30 (chipmunk)", so shifting by the wrong range produces an artefact,
and training on an artefact teaches the artefact. A missing copy costs coverage; a
wrong one costs correctness.

`train/oww/train.py` prints what fraction of the Piper set has a known sex, and warns when the
mispronunciation list is empty. It will not stop you — the warning exists because a
silent run on a mislabelled corpus is the expensive outcome, not a loud one.
