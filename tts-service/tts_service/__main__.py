"""Command-line entry point: render text with any registered engine and measure it.

This exists so a new machine - or a new engine - can be checked in five seconds
without writing Python, and so the README's launch recipes are runnable as-is:

    # list what an engine offers
    python -m tts_service --tts kokoro-mlx --list-voices
    python -m tts_service --tts 127.0.0.1:10200 --list-voices
    python -m tts_service --tts http://kokoro:8880 --list-voices

    # render: one clip to a file, or several into a directory
    python -m tts_service --tts kokoro-mlx --voice af_bella -o hey.wav "hey seeree"
    python -m tts_service --tts kokoro-mlx --voice af_bella --batch 10 --out-dir out/ \\
        "hey seeree" "hey serious" "hey series" "hey Sarah" "hey Cindy"

    # piper takes (voice, speaker) pairs; --speaker is optional
    python -m tts_service --tts 127.0.0.1:10200 --voice en_US-lessac-medium "hello"
    python -m tts_service --tts 127.0.0.1:10200 --voice en_US-libritts_r-medium \\
        --speaker 12 -o s12.wav "hey seeree"

    # the tts-service (python -m tts_service.server): one port, whatever engine it hosts
    python -m tts_service --tts tcp://127.0.0.1:8899 --voice af_bella -o svc.wav "hey seeree"

    # speed, applied the way each engine applies it (Kokoro's parameter, WSOLA for Piper)
    python -m tts_service --tts kokoro-mlx --voice af_bella --speed 1.6 -o fast.wav "hey seeree"

A run prints one line per clip and a summary with the measured ms/clip - the same
numbers the corpus generator's progress bar shows - so a machine that looks slow
says so before you spend an hour on a corpus.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.io.wavfile

from . import Client, engines_from_spec
from .audio import SR
from .engine import Engine


def _engine_for(engines, args):
    for e in engines:
        ok, why = e.available()
        if ok:
            return e
        print(f"warning: {e.name} not usable: {why}", file=sys.stderr)
    print("error: no usable engines", file=sys.stderr)
    sys.exit(1)


def _voice(engine, args):
    if isinstance(engine, Engine) and engine.name == "piper":
        return (args.voice, args.speaker) if args.speaker is not None else args.voice
    return args.voice


def _list_voices(engine, args):
    try:
        if engine.name == "piper":
            voices = engine.voices(max_speakers=args.max_speakers)
        else:
            voices = engine.voices()
    except SystemExit:
        raise
    except Exception as e:
        print(f"error: {engine.name} returned no voices: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(voices)} voices:")
    for v in voices:
        print(f"  {v}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tts_service", description=__doc__)
    p.add_argument("--tts", default="kokoro-mlx",
                   help="engine spec: kokoro-mlx | http(s) URL(s) | tcp://host:port (tts-service) | "
                        "piper://host:port | host:port (default kokoro-mlx)")
    p.add_argument("--voice", help="voice name (Kokoro voice id, or Piper voice name)")
    p.add_argument("--speaker", default=None,
                   help="Piper speaker for multi-speaker models (e.g. 12)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="delivery rate; 1.6 = 1.6x faster (default 1.0)")
    p.add_argument("--batch", type=int, default=1,
                   help="max texts per request for timestamp engines (default 1)")
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--max-speakers", type=int, default=0,
                   help="cap on Piper speakers per multi-speaker model (default: all)")
    p.add_argument("-o", dest="out_file", default=None, help="output .wav (one text)")
    p.add_argument("--out-dir", default=None, help="output directory (several texts)")
    p.add_argument("texts", nargs="*", help="text to render")
    args = p.parse_args(argv)

    engines = engines_from_spec(args.tts)
    engine = _engine_for(engines, args)

    if args.list_voices:
        _list_voices(engine, args)
        return 0

    if not args.texts:
        p.error("give at least one text, or --list-voices")
    if args.out_file and args.out_dir:
        p.error("-o and --out-dir are mutually exclusive")
    if args.out_file and len(args.texts) != 1:
        p.error("-o takes exactly one text; use --out-dir for several")

    voice = _voice(engine, args)
    clips = []
    total_ms = 0.0

    def record(name, text):
        nonlocal total_ms
        t0 = time.time()
        audio = engine.render(voice, text, args.speed)
        ms = (time.time() - t0) * 1000
        total_ms += ms
        if audio is None:
            print(f"{name}: FAILED ({engine.name})")
            return None
        print(f"{name}: {len(audio) / SR:.3f}s audio in {ms:.0f} ms")
        return audio

    def write(path, audio):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        scipy.io.wavfile.write(path, SR, audio)
        clips.append(path)

    if args.batch > 1 and engine.supports_timestamps and len(args.texts) > 1:
        # The batched path: chunks of --batch texts per joined request.
        out_dir = args.out_dir or "tts-service-out"
        for start in range(0, len(args.texts), args.batch):
            chunk = args.texts[start:start + args.batch]
            t0 = time.time()
            results = engine.batch(voice, args.speed, chunk)
            wall_ms = (time.time() - t0) * 1000
            total_ms += wall_ms
            ok = 0
            for i, (audio, _ts) in enumerate(results):
                name = f"{out_dir}/clip-{start + i:03d}.wav"
                if audio is None:
                    print(f"{name}: FAILED in batch ({engine.name})")
                    continue
                print(f"{name}: {len(audio) / SR:.3f}s ({wall_ms / len(chunk):.0f} ms/clip)")
                write(name, audio)
                ok += 1
            if ok < len(chunk):
                print(f"warning: {len(chunk) - ok}/{len(chunk)} clips from this batch "
                      f"were unlocatable in the joined rendering")
    else:
        for i, text in enumerate(args.texts):
            audio = record(f"clip-{i:03d}", text)
            if audio is not None:
                write(args.out_file or
                      os.path.join(args.out_dir or "tts-service-out", f"clip-{i:03d}.wav"),
                      audio)

    if total_ms:
        per = total_ms / len(args.texts)
        print(f"\n{len(clips)}/{len(args.texts)} clips written, "
              f"{per:.0f} ms/clip end to end "
              f"({'batch' if args.batch > 1 else 'serial'} x{args.batch}, {engine.name})")
    if not clips:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
