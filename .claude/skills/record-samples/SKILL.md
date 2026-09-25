---
name: record-samples
description: Guide a user through recording real wake-word voice samples (step 1 of the training pipeline) and check the result is usable before training. Use when the user wants to record samples, add a speaker, capture training audio, collect voice data, check their recordings, or asks why detection is poor for a particular voice.
---

# Recording real wake-word samples

Step 1 of the pipeline. Runs **on the host, not in Docker** — it needs the microphone,
and deliberately avoids the trainer's torch/tensorflow stack.

## You cannot run the recording yourself

The recorder is an interactive loop that waits on ENTER and needs a human to speak into
a microphone. Do not try to drive it — you will block on `input()` and capture silence.

Your job is to prepare the command, hand it to the user, then **verify what came back**.
The verification is the valuable half: bad samples are not recoverable later, and the
user will not know they are bad by listening casually.

## Preflight

```bash
which uv && brew list portaudio >/dev/null 2>&1 && echo "portaudio ok"
```

If PortAudio is missing: `brew install portaudio`. It is needed to build PyAudio, which
is the default and better capture backend — the ffmpeg fallback produced audible clicks.

First run on macOS may sit waiting on a microphone permission prompt for the terminal.
If capture fails immediately, that is the first thing to check.

## The commands to hand over

Device numbering differs per backend, so list with the same `--backend` used to record:

```bash
cd src/record
uv run record_samples.py --list-devices
```

Then, one directory per speaker:

```bash
uv run record_samples.py \
  --wake-word "hey seeree" \
  --device 0 \
  --output-dir data/recordings/samples/speaker1
```

Take-by-take: ENTER arms, "SPEAK NOW!" cues, 2 s captured, levels reported. `q` or Ctrl-C
quits. Numbering resumes from the highest existing index, so sessions can be split across
days, and the recorder refuses to overwrite an existing take.

For a longer stretch, `--continuous 120` records one block and splits it on silence — say
the phrase, pause ~1 s, repeat. Ctrl-C stops early and keeps what was captured. The unsplit
take is kept under `data/recordings/raw/<speaker>/`, so it can be re-cut with
`--resegment <wav> --dry-run` while tuning `--gap-ms`/`--min-ms` without recording again.

## What to tell the user before they start

- **20–50 samples minimum per speaker.** Vary tone, speed, and distance from the mic.
- **Aim for a peak near −12 dBFS.** Absolute level turns out not to matter to the model;
  SNR does, and it degrades detection below about 15 dB. Recording hot buys margin.
- **Level is fixed at capture.** Amplifying a quiet clip raises its noise with it, so a
  quiet session cannot be rescued afterwards.
- **Listen to a few clips early.** The meters measure level and steady noise, so an
  impulsive click passes them: takes measuring SNR 38 dB have sounded plainly wrong.

## More than one speaker matters

`src/train/corpus/augment.py` shifts synthetic clips into the child range specifically because
a run measured a 4-year-old at 24% detection against 97% for the adult, while being 26% of
the real corpus. Under-representation was not the cause — the fundamental sat outside
everything the model had seen. If the wake word needs to work for a child or a very
different voice, real clips from that speaker are worth more than any augmentation.

## Verify before anyone trains on it

Things you *can* do, and should:

```bash
# how many, per speaker
find data/recordings/samples -name '*.wav' | wc -l
ls data/recordings/samples/

# where speech sits in openWakeWord's detection window
cd src/record && uv run python check_alignment.py data/recordings/samples/
```

Trailing silence pushes the phrase earlier than the alignment the model sees when
streaming detection fires, so this is worth running before a multi-hour training run
rather than after it.

Re-read the level warnings from the session output if the user pasted them. `LOW LEVEL`
or `NOISY` on most takes means re-record rather than train — raise the input gain, move
closer, or find a quieter room.

## Holdout

Holdout clips exist to be evaluated against and are **never trained on**. They belong in
`data/recordings/holdout/`, kept out of `data/recordings/samples/` entirely, since the
trainer globs the samples tree recursively for positives.

```bash
cd src/record
uv run record_samples.py --holdout --speaker speaker1
```

`--speaker` picks the subdirectory under whichever of `samples/` or `holdout/` applies,
so the same name means the same person on both sides. Combining `--holdout` with an
`--output-dir` that points outside `holdout/` is rejected rather than silently resolved.

**Record the holdout in the same session as the training clips.** A holdout captured
weeks later on a different mic measures the setup as much as it measures the model.

For run-on clips — the phrase spoken straight into a command — use a `_runon` suffix
on the speaker name (`--holdout --speaker speaker1_runon`). The eval tools key on that
suffix to keep the two sets apart, since a run-on clip scored as if it were the phrase
alone measures the wrong thing. Anything without the suffix is treated as plain.

The eval step then finds all of this on its own: with no `--positives`/`--runon`, it
picks up every held-out speaker directory. Nothing to wire up per speaker.
