#!/usr/bin/env python3
"""Generate a synthetic positive corpus for wake-word evaluation, using an
OpenAI-compatible TTS server (tested against Kokoro-FastAPI).

Read the output carefully, because this corpus is not a generalisation test.
`train.py` generates its positives from every English Kokoro voice at speeds
0.7-1.3, so a plain rendering of the wake phrase is inside the training
distribution and the model has effectively seen it. Detection near 100% on those
clips means the training run worked; it says nothing about a new speaker. The real
recordings in `my_real_samples/` remain the only speaker-generalisation measure.

What this *is* good for is the axes where the corpus can be pushed outside what
training saw, which is why generation is organised as sweeps:

    voices    the phrase across every voice, at a normal speed - the sanity check,
              and a per-voice breakdown showing whether any voice type fails
    speed     deliberately beyond the 0.7-1.3 training range, to find where
              unusually slow or fast delivery stops being recognised
    level     the same clip attenuated toward the level real recordings came in
              at (-22 dBFS), to test whether recording level matters at all
    noise     the same clip mixed with room tone at decreasing SNR
    command   the phrase spoken straight into a command, as one utterance

The `command` sweep is the one worth explaining. `eval_model.py` already tests this
by concatenating a positive recording onto an unrelated command recording, but that
splice has no coarticulation and an audible seam - the phrase ends the way an
isolated phrase ends, then unrelated audio begins. Rendering "hey seeree, what's the
time?" as a single utterance instead gives the prosody of someone actually talking
to a device: the phrase runs into the command, and its final syllable is shaped by
what follows. It is generated in two variants, `cmd_run` with no punctuation and
`cmd_pause` with a comma, because the TTS puts a natural pause in for the comma and
tuning.md found a 300 ms pause was worth six detections to the pre-alignment model.

The commands are the same ones `generate_negatives.py` renders WITHOUT the wake
word, so the two corpora form a matched pair: identical trailing speech, differing
only in whether the wake word precedes it.

Filenames encode the swept variable (`speed_0.60_af_bella.wav`), so results can be
grouped by it afterwards.

Examples
--------
    # everything, against a Kokoro-FastAPI server on the LAN
    python generate_positives.py --wake-word "hey seeree" \\
        --url http://192.168.2.14:8880/v1/audio/speech --out positives_tts

    # just the axis you care about
    python generate_positives.py --wake-word "hey seeree" --sweeps speed

    # see what would be produced without calling the server
    python generate_positives.py --wake-word "hey seeree" --dry-run

Then score with:
    python eval_model.py --model MODEL --positives positives_tts --negatives negatives_tts
"""

import argparse
import concurrent.futures as cf
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000
FULL_SCALE = 32768.0

# A spread of accents and pitches; a single voice would measure that voice.
VOICES = ["af_bella", "af_nicole", "af_sarah", "af_sky", "af_heart", "af_nova",
          "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck",
          "bf_emma", "bf_lily", "bf_alice", "bm_george", "bm_lewis", "bm_daniel"]

# train.py renders positives at speeds drawn from U(0.7, 1.3). Values inside that
# range test nothing the model has not seen; the ones outside are the measurement.
SPEEDS = [0.55, 0.65, 0.75, 1.0, 1.25, 1.4, 1.6]

# Real recordings came in at a median peak of -22 dBFS. -12 is where they should be.
LEVELS_DBFS = [-6, -12, -18, -22, -28, -34]

# SNR against added room tone, in dB.
SNRS_DB = [30, 20, 15, 10, 5]

# Kept identical to generate_negatives.py's COMMAND list, so "wake word + command"
# and "command alone" differ by exactly one thing.
COMMANDS = [
    "what's the time?", "turn on the lights please", "play some music",
    "set a timer for five minutes", "what's the weather like today?",
    "turn the volume down a bit",
]


def synth(text, voice, speed, args):
    """Render one utterance as 16 kHz mono 16-bit PCM."""
    payload = json.dumps({"model": args.model, "input": text, "voice": voice,
                          "response_format": "wav", "speed": round(float(speed), 3)}).encode()
    request = urllib.request.Request(args.url, data=payload, headers={
        "Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"})

    raw = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = response.read()
            break
        except Exception as exc:                                     # noqa: BLE001
            if attempt == args.retries - 1:
                return None, f"{type(exc).__name__}: {exc}"

    scratch = Path(args.out) / f".raw_{voice}_{speed}.wav"
    scratch.write_bytes(raw)
    with warnings.catch_warnings():
        # Streaming servers emit a placeholder RIFF length; the data itself is fine.
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sr, data = wavfile.read(scratch)
    scratch.unlink()

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = resample_poly(data.astype(np.float32), SR, sr)
    return np.clip(data, -32768, 32767).astype(np.int16), None


def set_level(data, target_dbfs):
    """Scale so the peak sits at `target_dbfs`, as a recording at that gain would."""
    peak = float(np.abs(data).max())
    if peak <= 0:
        return data
    return np.clip(data * (10 ** (target_dbfs / 20) * FULL_SCALE / peak),
                   -32768, 32767).astype(np.int16)


def add_noise(data, snr_db, rng):
    """Mix in room tone at a given SNR, measured against the speech RMS."""
    speech_rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if speech_rms <= 0:
        return data
    noise = rng.normal(0, speech_rms / (10 ** (snr_db / 20)), len(data))
    return np.clip(data.astype(np.float64) + noise, -32768, 32767).astype(np.int16)


def plan(args):
    """(filename, voice, speed, post, text) for every clip the sweeps require.

    Filenames are `{sweep}_{value}_{voice}.wav` so eval_model.py --by-group can
    recover the swept variable from the first two fields.
    """
    phrase = args.wake_word
    jobs = []
    if "voices" in args.sweeps:
        for voice in VOICES:
            jobs.append((f"voices_1.00_{voice}.wav", voice, 1.0, None, phrase))
    if "speed" in args.sweeps:
        for speed in SPEEDS:
            for voice in VOICES[:args.voices_per_step]:
                jobs.append((f"speed_{speed:.2f}_{voice}.wav", voice, speed, None, phrase))
    if "level" in args.sweeps:
        for dbfs in LEVELS_DBFS:
            for voice in VOICES[:args.voices_per_step]:
                jobs.append((f"level_{dbfs:+03d}_{voice}.wav", voice, 1.0,
                             ("level", dbfs), phrase))
    if "noise" in args.sweeps:
        for snr in SNRS_DB:
            for voice in VOICES[:args.voices_per_step]:
                jobs.append((f"noise_{snr:02d}_{voice}.wav", voice, 1.0,
                             ("noise", snr), phrase))
    if "command" in args.sweeps:
        # The comma is the whole difference between the two variants: the TTS reads
        # it as a pause, which is the interaction style tuning.md found the model
        # used to depend on.
        for variant, template in (("run", "{phrase} {command}"),
                                  ("pause", "{phrase}, {command}")):
            for i, command in enumerate(COMMANDS):
                for voice in VOICES[:args.voices_per_step]:
                    text = template.format(phrase=phrase, command=command)
                    jobs.append((f"cmd_{variant}_{i:02d}_{voice}.wav", voice, 1.0,
                                 None, text))
    return jobs


def produce(job, args, rng):
    filename, voice, speed, post, text = job
    data, error = synth(text, voice, speed, args)
    if data is None:
        return None, f"FAIL {filename}: {error}"

    if post and post[0] == "level":
        data = set_level(data, post[1])
    elif post and post[0] == "noise":
        # Set a realistic level first, so the SNR is not measured against a clip
        # sitting at full scale.
        data = add_noise(set_level(data, -12), post[1], rng)

    wavfile.write(Path(args.out) / filename, SR, data)
    peak = 20 * np.log10(max(float(np.abs(data).max()), 1) / FULL_SCALE)
    return filename, f"{filename:<34}{len(data) / SR:5.2f}s  peak {peak:>6.1f} dBFS"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wake-word", required=True, help="The phrase to render")
    p.add_argument("--url", default="http://localhost:8880/v1/audio/speech",
                   help="OpenAI-compatible speech endpoint (default: %(default)s)")
    p.add_argument("--api-key", default="not-needed",
                   help="bearer token; Kokoro-FastAPI ignores it")
    p.add_argument("--model", default="kokoro", help="TTS model name")
    p.add_argument("--out", default="positives_tts", help="output directory")
    p.add_argument("--sweeps", nargs="+",
                   default=["voices", "speed", "level", "noise", "command"],
                   choices=["voices", "speed", "level", "noise", "command"],
                   help="which sweeps to generate")
    p.add_argument("--voices-per-step", type=int, default=6,
                   help="voices per point in the speed/level/noise sweeps "
                        "(default: %(default)s)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests; keep modest, the server does the work")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be generated without calling the server")
    args = p.parse_args()

    jobs = plan(args)
    if args.dry_run:
        for filename, voice, speed, post, text in jobs:
            note = f"  {post[0]}={post[1]}" if post else ""
            print(f"  {filename:<34}{voice:<12} speed={speed:.2f}{note}   \"{text}\"")
        print(f"\n{len(jobs)} clips across {len(args.sweeps)} sweep(s) "
              "(nothing written; drop --dry-run to generate)")
        return

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    print(f'generating {len(jobs)} positives for "{args.wake_word}" -> {out}')

    rng = np.random.default_rng(0)
    written, failures = 0, []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(produce, job, args, rng) for job in jobs]
        for future in cf.as_completed(futures):
            filename, line = future.result()
            if filename is None:
                failures.append(line)
            else:
                written += 1

    for line in failures:
        print(" ", line)
    print(f"\n{written} written, {len(failures)} failed")
    if written:
        print("\nNOTE: these are training-distribution clips. High detection here means")
        print("the training run worked, not that the model generalises to new speakers.")
        print("The speed points outside 0.7-1.3, and the level/noise sweeps, are the")
        print("parts that test something training did not already cover.")
        print(f"\nevaluate with:\n  python eval_model.py --model MODEL --positives {out}")


if __name__ == "__main__":
    main()
