#!/usr/bin/env python3
"""Generate a synthetic positive corpus for wake-word evaluation, speaking the repo's
TTS protocol (tts-service/tts_protocol) to whatever engine the URL points at.

Read the output carefully, because this corpus is not a generalisation test.
`train.py` generates its positives from every English Kokoro voice at speeds 0.7-1.3,
so a plain rendering of the wake phrase is inside the training distribution and the
model has effectively seen it. Detection near 100% on those clips means the training
run worked; it says nothing about a new speaker. The held-out recordings in
`data/recordings/holdout/` remain the only speaker-generalisation measure.

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
the tuning log found a 300 ms pause was worth six detections to the pre-alignment model.

The commands are the same ones `generate_negatives.py` renders WITHOUT the wake
word, so the two corpora form a matched pair: identical trailing speech, differing
only in whether the wake word precedes it.

Filenames encode the swept variable (`speed_0.60_af_bella.wav`), so results can be
grouped by it afterwards.

Examples
--------
    # everything, against the MLX engine on the Mac (or the Docker kokoro service)
    python -m eval.generate_positives --wake-word "hey seeree" \\
        --url tcp://127.0.0.1:8900

    # Piper instead of Kokoro (protocol port 8898; the voice list is then the
    # audited selection from train/corpus/piper.py, so its exclusion tables apply)
    python -m eval.generate_positives --wake-word "hey seeree" \\
        --tts piper --piper-url tcp://127.0.0.1:8898

    # just the axis you care about
    python -m eval.generate_positives --wake-word "hey seeree" --sweeps speed

    # see what would be produced without calling the server
    python -m eval.generate_positives --wake-word "hey seeree" --dry-run

`--url` accepts ONLY the `tcp://` protocol form, for the reason spelled out in
generate_negatives.py's docstring: the old `http://...` and `mlx://` forms used
to mean different backends with different audio. `--dry-run` constructs the
engine but lists instead of rendering.

Then score with, noting that --positives here OVERRIDES the held-out recordings the
gates normally run on - this measures the sweeps, not speaker generalisation:
    python -m eval.eval_model --model MODEL --positives data/corpus/eval/positives_tts
"""

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Runnable as `python eval/src/generate_positives.py` as well as `python -m
# eval.generate_positives`: the module form has the `eval` package importable,
# the plain-path form only has this directory on sys.path, so try both.
try:
    from eval import paths
except ImportError:
    import paths

sys.path.insert(0, str(paths.REPO_ROOT))
# The TTS protocol package lives at tts-service/tts_protocol/ at the repo root;
# the hyphens in both directory names mean that string is not importable, so its
# parent goes on sys.path. The engines are NOT here - they run as separate
# servers that this script only speaks to over TCP.
sys.path.insert(0, str(paths.REPO_ROOT / "tts-service" / "tts_protocol"))

from tts_protocol import TtsClient  # noqa: E402
from train.corpus.piper import select_piper_voices  # noqa: E402
from wordlists import load_voice_holdout, voice_holdout_path  # noqa: E402

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

# The voice-holdout ranking set (--voice-holdout): the held-out axis is the
# VOICE, so every other axis stays inside the training distribution - train.py
# renders positives at speeds drawn from U(0.7, 1.3), so this grid (its
# inclusive bounds and the midpoints between them) keeps speed in-distribution
# while still giving each voice five renders. A synthetic voice is not a
# person: this set ranks sweep points (low variance, a real n), it does not
# stand in for the speaker-generalisation gates on the real holdout.
HOLDOUT_SPEEDS = [0.7, 0.85, 1.0, 1.15, 1.3]

# SNR against added room tone, in dB.
SNRS_DB = [30, 20, 15, 10, 5]

# Kept identical to generate_negatives.py's COMMAND list, so "wake word + command"
# and "command alone" differ by exactly one thing.
COMMANDS = [
    "what's the time?", "turn on the lights please", "play some music",
    "set a timer for five minutes", "what's the weather like today?",
    "turn the volume down a bit",
]


def _holdout_voices(args, engine, selection, holdout):
    """The holdout SET itself, checked against the live `selection`.

    The exclusion contract runs the OTHER way here than in the corpus builders:
    they DROP the holdout entries from the catalog, this renderer TAKES them -
    the set is rendered from the held-out voices, and the live catalog is still
    the source of truth, so a tracked entry the selection no longer carries is
    a stale list and an error here, not a skip (improvement.md P1.2).
    """
    entries = holdout.get(engine) or []
    if not entries:
        sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) reserves no "
                 f"{engine} voices - nothing to render")
    if engine == "kokoro":
        offered = set(selection)
        missing = [e for e in entries if e not in offered]
    else:
        # (voice, speaker-or-None) pairs; a bare voice (speaker None) must match
        # at least one speaker of that model the selection carries.
        missing = [e for e in entries
                   if not any(p[0] == e[0] and (e[1] is None or p[1] == e[1])
                              for p in selection)]
    if missing:
        what = "voice(s)" if engine == "kokoro" else "(voice, speaker) pair(s)"
        sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) names {engine} "
                 f"{what} the live catalog does not carry: {missing}. Update the "
                 f"tracked list to match the catalog, rather than rendering a set "
                 f"whose holdout cannot be enforced.")
    shown = [e if not isinstance(e, tuple) else f"{e[0]}:{e[1]}"
             if e[1] is not None else e[0] for e in entries]
    print(f"  voice holdout: this set renders the {len(entries)} held-out {engine} "
          f"voice(s) (of {len(selection)} offered): {', '.join(shown)}")
    return list(entries)


def set_manifest(out, args, written, holdout):
    """Label the rendered directory so nothing can read it as a real-speaker gate.

    The directory name (voice_holdout_tts) and the filename prefix (holdout_)
    already say what the set is; this file says it deliberately - a future run
    that points eval_model.py --positives at it by mistake still has to read
    the label first.
    """
    import json
    from datetime import datetime, timezone
    manifest = {
        "set": "voice-holdout synthetic ranking set (improvement.md P1.2)",
        "what_it_is": ("Positives rendered ONLY from the voices "
                       "wordlists/voice_holdout.yaml holds out of every corpus "
                       "build: voice-disjoint from training, every other axis "
                       "inside the training distribution (speeds 0.7-1.3, plain "
                       "phrase)."),
        "what_it_is_not": ("A speaker-generalisation gate. A synthetic voice is "
                           "not a person; this set is a low-variance RANKING "
                           "signal for sweep points. The gates stay on the real "
                           "held-out recordings in data/recordings/holdout/, and "
                           "the top sweep points go there, not here."),
        "wake_word": args.wake_word,
        "tts": args.tts,
        "holdout_file": str(voice_holdout_path()),
        "holdout": {k: (v if not v or not isinstance(v[0], tuple)
                        else [x if not isinstance(x, tuple)
                             else (f"{x[0]}:{x[1]}" if x[1] else x[0])
                             for x in v]) for k, v in (holdout or {}).items()},
        "speeds": HOLDOUT_SPEEDS,
        "voices_rendered": sorted({vtag(v) for v in args.voices}),
        "clips_written": written,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = Path(out) / "set.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{path} (the set's label - read it before scoring)")


def vtag(voice):
    """Filename-safe tag for a voice: the id itself for Kokoro, voice_speaker for
    Piper multi-speaker models (the same convention the corpus naming uses)."""
    if isinstance(voice, tuple):
        return f"{voice[0]}_{voice[1]}" if voice[1] is not None else voice[0]
    return voice


def synth(text, voice, speed, args):
    """Render one utterance as 16 kHz mono 16-bit PCM.

    The engine returns 16 kHz int16 for both Kokoro and Piper, so the old
    decode-resample step is gone; the retry stays. Piper raises on transport
    failure, Kokoro returns None - same loop, one shape.
    """
    audio, error = None, ""
    for attempt in range(args.retries):
        try:
            audio = args.engine.render(voice, text, round(float(speed), 3))
        except Exception as exc:                                     # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        else:
            if audio is not None:
                break
    return audio, error


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
    voices = args.voices
    jobs = []
    if "voices" in args.sweeps:
        for voice in voices:
            jobs.append((f"voices_1.00_{vtag(voice)}.wav", voice, 1.0, None, phrase))
    if "speed" in args.sweeps:
        for speed in SPEEDS:
            for voice in voices[:args.voices_per_step]:
                jobs.append((f"speed_{speed:.2f}_{vtag(voice)}.wav", voice, speed, None, phrase))
    if "level" in args.sweeps:
        for dbfs in LEVELS_DBFS:
            for voice in voices[:args.voices_per_step]:
                jobs.append((f"level_{dbfs:+03d}_{vtag(voice)}.wav", voice, 1.0,
                             ("level", dbfs), phrase))
    if "noise" in args.sweeps:
        for snr in SNRS_DB:
            for voice in voices[:args.voices_per_step]:
                jobs.append((f"noise_{snr:02d}_{vtag(voice)}.wav", voice, 1.0,
                             ("noise", snr), phrase))
    if "command" in args.sweeps:
        # The comma is the whole difference between the two variants: the TTS reads
        # it as a pause, which is the interaction style the tuning log found the model
        # used to depend on.
        for variant, template in (("run", "{phrase} {command}"),
                                  ("pause", "{phrase}, {command}")):
            for i, command in enumerate(COMMANDS):
                for voice in voices[:args.voices_per_step]:
                    text = template.format(phrase=phrase, command=command)
                    jobs.append((f"cmd_{variant}_{i:02d}_{vtag(voice)}.wav", voice, 1.0,
                                 None, text))
    if "holdout" in args.sweeps:
        # Every voice, every in-distribution speed: the held-out variable is the
        # voice, and all five renders per voice count toward its n. Filenames
        # carry the swept value (the speed), so --by-group can still separate
        # the points.
        for speed in HOLDOUT_SPEEDS:
            for voice in voices:
                jobs.append((f"holdout_{speed:.2f}_{vtag(voice)}.wav", voice, speed,
                             None, phrase))
    return jobs


def render_jobs(jobs, args):
    """{(voice, speed, text): audio} for every distinct utterance the jobs need.

    Grouped by (voice, speed): a timestamp engine joins the group's distinct texts
    into ONE request and splits it back on word timestamps - the same win the
    corpus generator measures (a short Kokoro request is ~3/4 fixed overhead; see
    tts_protocol/engine.py's batch()). An engine without word timestamps (Piper)
    renders per clip, in voice-outer order: its server serves one request at a
    time, so order is a correctness property, not a preference
    (tts-service/engines/piper). Deduplication is by text, so a
    phrase rendered at 1.0 for the voices sweep is not rendered again for the
    level sweep.
    """
    groups = {}
    for _filename, voice, speed, _post, text in jobs:
        groups.setdefault((voice, speed), []).append(text)

    results = {}
    for (voice, speed), texts in groups.items():
        unique = list(dict.fromkeys(texts))
        if args.engine.supports_timestamps and len(unique) > 1:
            rendered = args.engine.batch(voice, speed, unique)
        else:
            rendered = [(args.engine.render(voice, t, speed), None) for t in unique]
        for text, (audio, _ts) in zip(unique, rendered):
            if audio is not None:
                results[(voice, speed, text)] = audio
    return results


def produce(job, args, rng, rendered):
    filename, voice, speed, post, text = job
    data = rendered.get((voice, speed, text))
    if data is None:
        return None, (f"FAIL {filename}: "
                      f"{args.engine.server_engine or args.engine.name} "
                      f"returned no audio")

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
    p.add_argument("--voice-holdout", action="store_true",
                   help="Render the voice-HOLDOUT synthetic ranking set instead of "
                        "the training-distribution sanity corpus: every clip from "
                        "the voices wordlists/voice_holdout.yaml reserves out of "
                        "every corpus build (the catalog is checked live, and a "
                        "held-out voice it no longer offers is an error, not a "
                        "skip), at speeds inside the 0.7-1.3 training range. "
                        "Output goes to data/corpus/eval/voice_holdout_tts by "
                        "default and is reported as a ranking signal, never as "
                        "a speaker-generalisation gate (improvement.md P1.2)")
    p.add_argument("--url", default="tcp://127.0.0.1:8900",
                   help="Kokoro protocol server: a tcp:// spec (the MLX engine's"
                        " default port on a Mac, the Docker kokoro service's 8899 on"
                        " the box). Only tcp:// is accepted (default: %(default)s)")
    p.add_argument("--tts", default="kokoro", choices=["kokoro", "piper"],
                   help="TTS engine (default: %(default)s)")
    p.add_argument("--piper-url", default="tcp://127.0.0.1:8898",
                   help="Piper protocol server tcp:// spec, used with --tts piper"
                        " (default: %(default)s)")
    p.add_argument("--max-speakers", type=int, default=12,
                   help="Piper: cap on speakers per multi-speaker model "
                        "(default: %(default)s)")
    p.add_argument("--api-key", default="not-needed",
                   help="ignored; kept so old invocations keep working (the engine "
                        "owns the request shape now)")
    p.add_argument("--model", default="kokoro", help="ignored; see --api-key")
    p.add_argument("--out", default=str(paths.POSITIVES_DIR),
                   help="output directory (default: %(default)s)")
    p.add_argument("--sweeps", nargs="+",
                   default=["voices", "speed", "level", "noise", "command"],
                   choices=["voices", "speed", "level", "noise", "command", "holdout"],
                   help="which sweeps to generate (--voice-holdout defaults to "
                        "holdout: every held-out voice at every in-distribution speed)")
    p.add_argument("--voices-per-step", type=int, default=6,
                   help="voices per point in the speed/level/noise sweeps "
                        "(default: %(default)s)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests; keep modest, the server does the work")
    p.add_argument("--timeout", type=float, default=120,
                   help="accepted for old invocations; the engine owns its request "
                        "timeout now (tts-service)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be generated without calling the server")
    args = p.parse_args()

    # --voice-holdout takes over two defaults: the voice SET (the tracked
    # holdout list, not the in-distribution VOICES list) and the OUTPUT
    # (its own directory, so the existing positives/negatives corpora stay
    # byte-for-byte untouched). Both only when the user has not set them.
    if args.voice_holdout:
        if args.out == str(paths.POSITIVES_DIR):
            args.out = str(paths.EVAL_CORPUS_DIR / "voice_holdout_tts")
        if args.sweeps == ["voices", "speed", "level", "noise", "command"]:
            args.sweeps = ["holdout"]

    # Engine construction is offline (no I/O), so it happens before the dry-run
    # even though voice SELECTION for Piper needs the live catalog.
    holdout = load_voice_holdout() if args.voice_holdout else None
    if args.voice_holdout and not holdout:
        # The no-op rule applies to the TRAINERS (no tracked file: train on the
        # whole catalog, print a note). It does not apply here: with no reserved
        # voices there is nothing to render, and falling back to the in-corpus
        # VOICES list would be a training-distribution measurement wearing the
        # holdout's label - the one outcome this whole item exists to prevent.
        sys.exit(f"ERROR: --voice-holdout needs the tracked list at "
                 f"{voice_holdout_path()}, which is absent or empty. The trainers "
                 f"treat that as a no-op (a fresh checkout predating the file "
                 f"still trains); the ranking set cannot - create the file first.")
    if args.tts == "piper":
        args.engine = TtsClient(args.piper_url)
        if args.dry_run:
            if holdout is not None:
                if holdout.get("piper"):
                    args.voices = list(holdout["piper"])
                    print(f"  NOTE: --dry-run - the {len(args.voices)} held-out (voice, speaker) "
                          f"pairs are taken as written; the live selection check happens on "
                          f"the real render")
                else:
                    sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) reserves "
                             f"no Piper pairs - nothing to render for --tts piper")
            else:
                args.voices = ["<live Piper catalog>"]
        else:
            # The trainer's audited selection: drops the voices whose pronunciation
            # of this wake word failed the audit, so the corpus cannot contain
            # mislabelled phrases - the failure mode the Kokoro side spent eleven
            # runs discovering (train/corpus/piper.py).
            selection = select_piper_voices(args.piper_url, args.wake_word,
                                            max_speakers=args.max_speakers)
            if not selection:
                sys.exit("ERROR: no usable Piper voices")
            if holdout is not None:
                # The set is the held-out pairs themselves (checked against the
                # audited selection: a pair the audit dropped cannot render the
                # eval set either, so the tracked list must move, not the audit).
                args.voices = _holdout_voices(args, "piper", selection, holdout)
            else:
                args.voices = selection
        if not args.voices:
            sys.exit("ERROR: no usable Piper voices")
    else:
        args.engine = TtsClient(args.url)
        if holdout is not None and not holdout.get("kokoro"):
            sys.exit(f"ERROR: the voice holdout ({voice_holdout_path()}) reserves no "
                     f"Kokoro voices - nothing to render for --tts kokoro")
        if holdout is not None and holdout.get("kokoro"):
            # The live catalog is the source of truth, but --dry-run's contract
            # is to call no server at all: without one the list is taken as
            # written and the check is deferred to the real render, which does
            # it unconditionally (a stale list there is a sys.exit, not a skip).
            if args.dry_run:
                args.voices = list(holdout["kokoro"])
                print(f"  NOTE: --dry-run - the {len(args.voices)} holdout voices "
                      f"are taken as written; the live catalog check happens on "
                      f"the real render")
            else:
                catalog = args.engine.voices()
                args.voices = _holdout_voices(args, "kokoro", catalog, holdout)
        else:
            args.voices = VOICES
        if not args.voices:
            sys.exit("ERROR: no usable Kokoro voices")

    jobs = plan(args)
    if args.dry_run:
        for filename, voice, speed, post, text in jobs:
            note = f"  {post[0]}={post[1]}" if post else ""
            print(f"  {filename:<44}{vtag(voice):<16} speed={speed:.2f}{note}   \"{text}\"")
        print(f"\n{len(jobs)} clips across {len(args.sweeps)} sweep(s) "
              "(nothing written; drop --dry-run to generate)")
        return

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    # The catalog reply carries the engine's own name (wire.py): a URL pointed
    # at the wrong server says so at the top of the run, not at its end. One
    # cheap exchange, cached for life.
    args.engine.voices()
    print(f'generating {len(jobs)} positives for "{args.wake_word}" -> {out} '
          f"({args.engine.server_engine or args.engine.name}, "
          f"{len(args.voices)} voices)" + ("  [VOICE-HOLDOUT set]"
                                          if args.voice_holdout else ""))

    # Render first, in engine order (voice-outer for Piper), then post-process in
    # parallel as before - the sweeps differ only in post-processing, so one
    # rendering per distinct (voice, speed, text) serves them all.
    rendered = render_jobs(jobs, args)

    rng = np.random.default_rng(0)
    written, failures = 0, []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(produce, job, args, rng, rendered) for job in jobs]
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
        if args.voice_holdout:
            set_manifest(out, args, written, holdout)
            print("\nNOTE: this is the voice-HOLDOUT synthetic ranking set, not the")
            print("training-distribution corpus: every clip is a voice the corpus")
            print("builders never train on, at speeds inside the 0.7-1.3 training")
            print("range. A synthetic voice is not a person - this ranks sweep")
            print("points (low variance, a real n); the speaker-generalisation")
            print("gates stay on data/recordings/holdout/, where the top points go.")
            print(f"\nevaluate with (labelled block, never merged into the gates):\n"
                  f"  python eval_model.py --model MODEL --voice-holdout-set {out}")
        else:
            print("\nNOTE: these are training-distribution clips. High detection here means")
            print("the training run worked, not that the model generalises to new speakers.")
            print("The speed points outside 0.7-1.3, and the level/noise sweeps, are the")
            print("parts that test something training did not already cover.")
            print(f"\nevaluate with:\n  python eval_model.py --model MODEL --positives {out}")


if __name__ == "__main__":
    main()
