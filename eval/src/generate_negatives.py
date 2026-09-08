#!/usr/bin/env python3
"""Generate a targeted negative corpus for wake-word evaluation, using an
OpenAI-compatible TTS server (tested against Kokoro-FastAPI).

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

THE PHRASES ARE NOT IN THIS FILE. They live in wordlists/<wake_word>.yaml, because
they are the one part of this pipeline that does not transfer between wake words:
"hey serious" probes the boundary of "hey seeree" and says nothing about "okay
jarvis". Hardcoding them here made the repo look general while being about one
phrase. Write a new one with the `write-wordlists` skill.

A copy of this script also lives in the openWakeWord repo (`scripts/`). It has now
diverged: that copy keeps its phrases inline and has no wordlists package.

Examples
--------
    # from inside the eval container, against kokoro on the host
    python -m eval.generate_negatives \\
        --url http://host.docker.internal:8880/v1/audio/speech

    # against a Kokoro-FastAPI server elsewhere on the LAN
    python -m eval.generate_negatives \\
        --url http://192.168.2.14:8880/v1/audio/speech

    # see what would be produced without calling the server
    python -m eval.generate_negatives --dry-run

    # top up one category after editing its wordlist
    python -m eval.generate_negatives --categories extend

Then score a model against the result with `eval_model.py --negatives ...`, reading
the output per category rather than pooled — the corpus is adversarial by
construction, so a pooled false-accept rate means little.

Three of the six categories — `extend`, `running`, `hey_other` — are built from the
wake word's own consonants and vowels and carry nearly all of the signal. The other
three transfer unchanged.
"""

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# Runnable as `python eval/src/generate_negatives.py` as well as `python -m
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
    sys.path.insert(0, str(paths.REPO_ROOT))
    import wordlists  # noqa: E402
    DEFAULT_OUT = str(paths.NEGATIVES_DIR)
else:
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


def synth(index, category, text, args):
    """Render one utterance and write it as a 16 kHz mono 16-bit WAV."""
    rng = np.random.default_rng(index)
    voice = VOICES[index % len(VOICES)]
    speed = round(float(rng.uniform(*args.speed_range)), 3)

    payload = json.dumps({"model": args.model, "input": text, "voice": voice,
                          "response_format": "wav", "speed": speed}).encode()
    request = urllib.request.Request(args.url, data=payload, headers={
        "Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"})

    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = response.read()
            break
        except Exception as exc:                                     # noqa: BLE001
            if attempt == args.retries - 1:
                return None, f"FAIL {category}_{index:03d}: {type(exc).__name__}: {exc}"

    scratch = Path(args.out) / f".raw_{index:03d}.wav"
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
    data = np.clip(data, -32768, 32767).astype(np.int16)

    name = f"{category}_{index:03d}_{voice}.wav"
    wavfile.write(Path(args.out) / name, SR, data)
    return name, f"{name:<34} {len(data)/SR:5.2f}s  speed={speed:<5} {text[:44]}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8880/v1/audio/speech",
                   help="OpenAI-compatible speech endpoint (default: %(default)s)")
    p.add_argument("--api-key", default="not-needed",
                   help="bearer token; Kokoro-FastAPI ignores it (default: %(default)s)")
    p.add_argument("--model", default="kokoro", help="TTS model name")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="output directory for the WAVs (default: %(default)s, where "
                        "the eval tools look for them)")
    p.add_argument("--wake-word", default="hey seeree",
                   help="picks wordlists/<wake_word>.yaml (default: %(default)s)")
    p.add_argument("--wordlist", default=None,
                   help="explicit path to a wordlist YAML, instead of --wake-word")
    p.add_argument("--categories", nargs="+", default=list(wordlists.EVAL_CATEGORIES),
                   choices=list(wordlists.EVAL_CATEGORIES),
                   help="which categories to generate")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel requests; keep modest, the server is doing the work")
    p.add_argument("--speed-range", type=float, nargs=2, default=(0.88, 1.18),
                   metavar=("MIN", "MAX"), help="speaking-rate jitter")
    p.add_argument("--timeout", type=float, default=120)
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
        for i, (category, text) in enumerate(corpus):
            print(f"  {category}_{i:03d}_{VOICES[i % len(VOICES)]:<12} {text}")
        print(f"\n{len(corpus)} utterances across {len(args.categories)} categories "
              f"(nothing written; drop --dry-run to generate)")
        return

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    print(f"generating {len(corpus)} negatives -> {out}")

    written, failures = 0, []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(synth, i, c, t, args) for i, (c, t) in enumerate(corpus)]
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
