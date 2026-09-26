#!/usr/bin/env python3
"""Generate a targeted negative corpus for wake-word evaluation, speaking the repo's
TTS protocol (src/tts-service/tts_protocol) to whatever engine the URL points at.

A hundred random sentences would mostly measure nothing: a wake-word model that is
already quiet on ordinary speech scores zero on all of them. The useful negatives are
the ones that probe the decision the model actually makes — words that *begin* like
the wake phrase and then continue into something else, and the wake word's own first
syllable attached to a different ending.

The corpus is therefore grouped into categories, and results should be read per
category rather than pooled. Measured against `hey_seeree.onnx`, the informative rows
were `extend` (13/20 false accepts) and `hey_other` (5/12); `general` was 0/36 and is
the background rate.

Categories:
    extend      the phrase, then the word keeps going ("hey serious", "hey series")
    running     wake-word-like sounds inside ordinary speech, with no "hey"
    hey_other   "hey" followed by something else ("hey Sarah", "hey Cindy")
    command     bare commands with no wake word, to check speech alone cannot trip it
    other_ww    other assistants' wake words
    general     ordinary conversation, for a baseline false-accept rate

THE PHRASES ARE NOT IN THIS FILE. They live in src/wordlists/<wake_word>.yaml, because
they are the one part of this pipeline that does not transfer between wake words:
"hey serious" probes the boundary of "hey seeree" and says nothing about "okay
jarvis". Hardcoding them here made the repo look general while being about one
phrase. Write a new one with the `write-wordlists` skill.

A copy of this script also lives in the openWakeWord repo (`scripts/`). It has now
diverged: that copy keeps its phrases inline and has no wordlists package.

Examples
--------
    # from inside the eval container, against the MLX engine on the Mac host
    python -m eval.generate_negatives \\
        --url tcp://host.docker.internal:8900

    # against the Docker kokoro service on the training box (protocol port 8899)
    python -m eval.generate_negatives --url tcp://kokoro:8899

    # Piper instead of Kokoro (protocol port 8898; the voice list is then the
    # audited selection from src/train/corpus/piper.py, so its exclusion tables apply)
    python -m eval.generate_negatives --tts piper --piper-url tcp://127.0.0.1:8898

    # see what would be produced without calling the server
    python -m eval.generate_negatives --dry-run

    # top up one category after editing its wordlist
    python -m eval.generate_negatives --categories extend

`--url` accepts ONLY the `tcp://` protocol form: the old `http://...` server URL
and `mlx://` forms are rejected at the probe with an explanation, because they
used to mean different backends with different audio and a silent misread rendered
a whole corpus from the wrong one. The engines that publish a protocol port are
listed in src/tts-service/README.md (the MLX engine on 8900, the Docker kokoro on
8899, Piper on 8898).

Then score a model against the result with `eval_model.py --negatives ...`, reading
the output per category rather than pooled — the corpus is adversarial by
construction, so a pooled false-accept rate means little.

Three of the six categories — `extend`, `running`, `hey_other` — are built from the
wake word's own consonants and vowels and carry nearly all of the signal. The other
three transfer unchanged.
"""

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Runnable as `python src/eval/src/generate_negatives.py` as well as `python -m
# eval.generate_negatives`. The module form has the `eval` package importable;
# the plain-path form only has this directory on sys.path, so try both. Then put
# the repo root on sys.path for `wordlists` - paths.py finds the root.
try:
    from eval import paths
except ImportError:
    try:
        import paths  # plain-path form: this directory is on sys.path
    except ImportError:
        # The copy of this script that lives in the openWakeWord repo, where there
        # is no package and no data/ layout to write into.
        paths = None

if paths is not None:
    sys.path.insert(0, str(paths.REPO_ROOT / "src"))
    # The TTS protocol package lives at src/tts-service/tts_protocol/; the hyphens
    # in both directory names mean that string is not importable, so its parent
    # (src/) goes on sys.path. The engines are NOT here -
    # they run as separate servers that this script only speaks to over TCP.
    sys.path.insert(0, str(paths.REPO_ROOT / "src" / "tts-service" / "tts_protocol"))
    import wordlists  # noqa: E402
    from tts_protocol import TtsClient  # noqa: E402
    from train.corpus.piper import select_piper_voices  # noqa: E402
    DEFAULT_OUT = str(paths.NEGATIVES_DIR)
else:
    wordlists = None  # type: ignore
    TtsClient = None  # type: ignore
    select_piper_voices = None
    DEFAULT_OUT = "negatives_tts"

SR = 16000  # openWakeWord operates on 16 kHz mono audio

# A spread of accents and pitches; a single voice would measure that voice, not the model.
VOICES = ["af_bella", "af_nicole", "af_sarah", "af_sky", "af_heart", "af_nova",
          "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck",
          "bf_emma", "bf_lily", "bf_alice", "bm_george", "bm_lewis", "bm_daniel"]


def build_corpus(categories, phrases):
    out = []
    for name in categories:
        out += [(name, text) for text in phrases.get(name, [])]
    return out


def vtag(voice):
    """Filename-safe tag for a voice: the id itself for Kokoro,
    voice_speaker for Piper multi-speaker models. The same convention as
    generate_positives.vtag and the corpus naming - the client's voice is a
    (voice, speaker) tuple for Piper, and interpolating it raw put the tuple
    repr into the filename."""
    if isinstance(voice, tuple):
        return f"{voice[0]}_{voice[1]}" if voice[1] is not None else voice[0]
    return voice


def synth(index, category, text, voice, args):
    """Render one utterance and write it as a 16 kHz mono 16-bit WAV.

    The voice is passed in rather than derived from the index: main groups
    the jobs voice-outer (see there), which is a correctness property, not a
    scheduling preference.
    """
    rng = np.random.default_rng(index)
    speed = round(float(rng.uniform(*args.speed_range)), 3)

    # The engine returns 16 kHz int16 for both Kokoro and Piper, so the old
    # decode-resample step is gone; the retry stays - both engines fail transiently.
    # Piper raises on transport errors; Kokoro returns None. Same shape, one loop.
    audio, err = None, ""
    for attempt in range(args.retries):
        try:
            audio = args.engine.render(voice, text, speed)
        except Exception as exc:                                     # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        else:
            if audio is not None:
                break
    if audio is None:
        return None, (f"FAIL {category}_{index:03d}: "
                      f"{args.engine.server_engine or args.engine.name}"
                      f"{': ' + err if err else ''}")

    name = f"{category}_{index:03d}_{vtag(voice)}.wav"
    wavfile.write(Path(args.out) / name, SR, audio)
    return name, f"{name:<34} {len(audio)/SR:5.2f}s  speed={speed:<5} {text[:44]}"


def make_engine(args):
    # One engine here: a ~100-clip eval corpus has no business driving the pool
    # machinery, which exists to keep a whole server farm busy on a 5000-clip
    # corpus. The URL is a protocol spec - see the module docstring for why
    # only tcp:// is accepted.
    if args.tts == "piper":
        return TtsClient(args.piper_url)
    return TtsClient(args.url)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="output directory for the WAVs (default: %(default)s, where "
                        "the eval tools look for them)")
    p.add_argument("--wake-word", default="hey seeree",
                   help="picks src/wordlists/<wake_word>.yaml (default: %(default)s)")
    p.add_argument("--wordlist", default=None,
                   help="explicit path to a wordlist YAML, instead of --wake-word")
    p.add_argument("--categories", nargs="+", default=list(wordlists.EVAL_CATEGORIES),
                   choices=list(wordlists.EVAL_CATEGORIES),
                   help="which categories to generate")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests; keep modest, the server is doing the work")
    p.add_argument("--speed-range", type=float, nargs=2, default=(0.88, 1.18),
                   metavar=("MIN", "MAX"), help="speaking-rate jitter")
    p.add_argument("--timeout", type=float, default=120,
                   help="accepted for old invocations; the engine owns its request "
                        "timeout now (tts-service)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be generated without calling the server")
    args = p.parse_args()

    try:
        data = wordlists.load(args.wake_word, path=args.wordlist)
    except wordlists.WordlistError as exc:
        sys.exit(f"ERROR: {exc}")
    phrases = wordlists.eval_categories(data, args.categories)
    print(f"wordlist {data['_path']} for {data['wake_word']!r}: "
          + ", ".join(f"{k} {len(v)}" for k, v in phrases.items()))

    corpus = build_corpus(args.categories, phrases)
    if args.dry_run:
        shown = VOICES if args.tts != "piper" else ["<from live Piper catalog>"]
        for i, (category, text) in enumerate(corpus):
            print(f"  {category}_{i:03d}_{shown[i % len(shown)]:<34} {text}")
        print(f"\n{len(corpus)} utterances across {len(args.categories)} categories "
              f"(nothing written; drop --dry-run to generate)")
        return

    if wordlists is None or TtsClient is None:
        sys.exit("ERROR: this copy of the script has no repo layout to import the "
                 "TTS layer from - run it from the lva-wakeword-trainer checkout")

    args.engine = make_engine(args)
    if args.tts == "piper":
        # The trainer's audited selection: drops the voices whose pronunciation of
        # this wake word failed the audit, so the eval corpus cannot contain
        # mislabelled phrases - the failure mode the Kokoro side spent eleven runs
        # discovering (src/train/corpus/piper.py).
        args.voices = select_piper_voices(args.piper_url, args.wake_word,
                                          max_speakers=args.max_speakers)
        if not args.voices:
            sys.exit("ERROR: no usable Piper voices")
    else:
        args.voices = VOICES

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    # The catalog reply carries the engine's own name (wire.py): a URL pointed
    # at the wrong server (a Piper port, a stale process) says so at the top
    # of the run, not at its end. One cheap exchange, cached for life.
    args.engine.voices()
    print(f"generating {len(corpus)} negatives -> {out} "
          f"({args.engine.server_engine or args.engine.name}, "
          f"{len(args.voices)} voices)")

    written, failures = 0, []
    # Voice-outer, the way generate_positives.render_jobs orders the same
    # server: the protocol serves one request at a time (the lock in
    # tts_protocol/server.py), and Piper holds one voice resident - a voice
    # swap costs a reload, and the old per-clip rotation swapped on nearly
    # every clip. Each voice's group is fully done before the next starts, so
    # the order holds even though workers within a group run in parallel.
    # The voice assignment itself stays index-based, so filenames are stable.
    by_voice = {}
    for i, (c, t) in enumerate(corpus):
        by_voice.setdefault(args.voices[i % len(args.voices)], []).append((i, c, t))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for voice in args.voices:
            jobs = by_voice.get(voice, [])
            if not jobs:
                continue
            futures = [pool.submit(synth, i, c, t, voice, args) for (i, c, t) in jobs]
            for future in cf.as_completed(futures):
                name, line = future.result()
                if name is None:
                    failures.append(line)
                else:
                    written += 1

    for line in failures:
        print(" ", line)
    print(f"\n{written} written, {len(failures)} failed")
    if written:
        print(f"evaluate with:\n  eval_model.py --model MODEL --negatives {out}")


if __name__ == "__main__":
    main()
