"""Check a TTS server in five seconds: list its catalog, or render through it.

    python -m tts_protocol --server tcp://127.0.0.1:8899 --list-voices
    python -m tts_protocol --server tcp://127.0.0.1:8899 --voice af_bella -o hey.wav "hey seeree"
    python -m tts_protocol --server tcp://127.0.0.1:8898 --voice en_US-lessac-medium --speaker 12 -o s12.wav "hey seeree"
    python -m tts_protocol --server tcp://127.0.0.1:8899 --voice af_bella --batch 5 --out-dir out/ "a" "b" "c" "d" "e"

One line per clip, then a measured ms/clip summary - the same number the
corpus generator's progress bar shows, so a slow machine or a wedged server
says so before an hour is spent on a corpus. A batch groups the texts by the
protocol's join/split (engine.Engine.batch): one request, split on word
timestamps - or one request per text when the server declared no timestamps.
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

import scipy.io.wavfile

from .client import TtsClient, TtsProtocolError
from .audio import SR


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tts_protocol", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True,
                   help="the TTS server: tcp://host:port")
    p.add_argument("--list-voices", action="store_true",
                   help="print the catalog and exit")
    p.add_argument("--voice", help="voice id, or a Piper voice name with --speaker")
    p.add_argument("--speaker", help="Piper speaker, for multi-speaker voices")
    p.add_argument("--speed", type=float, default=1.0,
                   help="delivery rate (default: %(default)s)")
    p.add_argument("-o", "--out", help="write the single clip to this file")
    p.add_argument("--batch", type=int, default=1,
                   help="group this many texts into one join/split request "
                        "(default: 1)")
    p.add_argument("--out-dir", help="batch mode: write clips into this directory")
    p.add_argument("texts", nargs="*", help="the text(s) to render")
    args = p.parse_args(argv)

    client = TtsClient(args.server)
    ok, why = client.available()
    if not ok:
        print(f"ERROR: {args.server} is not usable: {why}", file=sys.stderr)
        return 1
    print(f"{args.server}: {client.name}, timestamps={client.supports_timestamps}, "
          f"speaker={client.speaker_voices}")

    voices = client.voices()
    if args.list_voices:
        for v in voices:
            print(" ", v if not isinstance(v, tuple) else f"{v[0]}:{v[1]}")
        return 0

    if not args.texts:
        p.error("no texts to render")

    def voice_arg():
        if not args.voice:
            p.error("--voice is required to render")
        if args.speaker is not None:
            return (args.voice, args.speaker)
        return args.voice

    t0 = time.monotonic()
    clips = 0
    for i in range(0, len(args.texts), args.batch):
        group = args.texts[i:i + args.batch]
        if args.batch == 1:
            results = [client.timed_render(voice_arg(), group[0], args.speed)]
        else:
            results = client.batch(voice_arg(), args.speed, group)
        for text, (audio, _ts) in zip(group, results):
            if audio is None:
                print(f"  MISS  {len(group) and text[:44]!r} (server rendered nothing)")
                continue
            clips += 1
            if args.batch == 1 and args.out:
                scipy.io.wavfile.write(args.out, SR, audio)
                print(f"  {args.out}  {len(audio) / SR:5.3f}s")
            elif args.out_dir:
                name = f"clip_{uuid.uuid4().hex}.wav"
                Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                scipy.io.wavfile.write(str(Path(args.out_dir) / name), SR, audio)
                print(f"  {name}  {len(audio) / SR:5.3f}s  {text[:44]}")
            else:
                print(f"  {len(audio) / SR:5.3f}s  {text[:44]}")

    dt = time.monotonic() - t0
    try:
        n = clips or len(args.texts)
    except ZeroDivisionError:
        n = 1
    print(f"{clips}/{len(args.texts)} clips in {dt * 1000:.0f} ms "
          f"({dt * 1000 / max(n, 1):.0f} ms/clip end to end)")
    return 0 if clips else 2


if __name__ == "__main__":
    raise SystemExit(main())
