#!/usr/bin/env python3
"""How many clips per second does one TTS server actually produce?

WHY THIS EXISTS. The answer decides how the corpus gets generated, and the two
engines here answer it in opposite directions:

  - Kokoro is single-threaded. One process pins one core, the GPU sits at 21%, and
    concurrent requests just queue (4 client threads: 15.0 it/s against 14.1
    sequential). More INSTANCES scale, roughly linearly, until the cores run out.
  - Piper is not. onnxruntime parallelises across cores, and one instance measured
    980% CPU - ten cores. A second instance was 0.88x, slower than one, because the
    two contend for the cores the first was already using.

Guessing which pattern an engine follows gets it backwards, so measure. The sweep
here is designed to tell them apart: if throughput is flat while latency grows
linearly with client threads, requests are queueing behind a serialised stage and
more clients will never help. Whether more INSTANCES help is the second question,
and --instances answers it.

    python bench_tts.py --engine piper --port 10200
    python bench_tts.py --engine piper --instances 10200 10201
    python bench_tts.py --engine kokoro --url http://localhost:8880

Run it on the machine that will generate the corpus. Numbers from a laptop - worse,
from an emulated architecture - are a floor, not a capacity plan.
"""

import argparse
import concurrent.futures as cf
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The phrase and a run-on, because synthesis cost scales with output length and a
# corpus is made of both.
PHRASES = ["hey seeree", "hey seeree what is on tonight"]


def piper_caller(host, port, voice):
    from train.corpus.piper import piper_render

    def call(i):
        return piper_render(host, port, voice, None, PHRASES[i % len(PHRASES)], 1.0)
    return call


def kokoro_caller(url, voice):
    import io
    import numpy as np
    import requests
    import scipy.io.wavfile

    def call(i):
        r = requests.post(f"{url}/v1/audio/speech", timeout=120, json={
            "model": "kokoro", "voice": voice, "response_format": "wav",
            "input": PHRASES[i % len(PHRASES)]})
        r.raise_for_status()
        _, data = scipy.io.wavfile.read(io.BytesIO(r.content))
        return np.asarray(data)
    return call


def timed(call, i):
    t = time.perf_counter()
    audio = call(i)
    return time.perf_counter() - t, len(audio) / 16000


def run(call, n, workers):
    t0 = time.perf_counter()
    if workers == 1:
        rows = [timed(call, i) for i in range(n)]
    else:
        with cf.ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(lambda i: timed(call, i), range(n)))
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in rows)
    return n / wall, sum(r[1] for r in rows) / wall, lat


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=("piper", "kokoro"), default="piper")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=10200, help="piper Wyoming port")
    p.add_argument("--url", default="http://localhost:8880", help="kokoro base URL")
    p.add_argument("--voice", default=None,
                   help="default: en_US-lessac-medium (piper) / af_bella (kokoro)")
    p.add_argument("--clips", type=int, default=24, help="clips per measurement")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="client-thread counts to sweep")
    p.add_argument("--instances", nargs="+", default=None, metavar="HOST:PORT",
                   help="piper: several instances, to test whether adding instances "
                        "adds throughput. Takes 'host:port' or a bare port (which "
                        "uses --host). Both forms matter: from the host the "
                        "instances are localhost:10200 and localhost:10201, but on "
                        "the compose network they are piper:10200 and piper2:10200 "
                        "- the 10201 mapping does not exist inside it.")
    args = p.parse_args()

    def split(spec):
        host, sep, port = str(spec).rpartition(":")
        return (host if sep else args.host), int(port)

    if args.engine == "piper":
        voice = args.voice or "en_US-lessac-medium"
        make = lambda host, port: piper_caller(host, port, voice)
        target = f"piper {args.host}:{args.port} voice={voice}"
    else:
        voice = args.voice or "af_bella"
        make = lambda _host, _port: kokoro_caller(args.url, voice)
        target = f"kokoro {args.url} voice={voice}"

    call = make(args.host, args.port)
    print(f"{target}\nwarming up...")
    try:
        for _ in range(3):
            call(0)
    except (OSError, ConnectionError) as e:
        # A raw gaierror/ECONNREFUSED here is nearly always one of two setup
        # mistakes, and the traceback says neither of them out loud.
        sys.exit(
            f"\nCould not reach {args.host}:{args.port} - {e}\n\n"
            "  * Is the service up?  `docker compose ps` should show it running.\n"
            "    Start it with `docker compose up -d piper` (add `--profile bench`\n"
            "    and piper2 for the --instances test). Note `docker compose down\n"
            "    <service>` tears down the network too - use `stop`.\n"
            "  * Are you using the right names for where this is running?\n"
            "    Inside the compose network:  --host piper   --instances piper:10200 "
            "piper2:10200\n"
            "    From the host:               --host localhost "
            "--instances localhost:10200 localhost:10201\n")

    print(f"\n{args.clips} clips per row\n")
    print(f"{'mode':<16}{'clips/s':>9}{'RTF':>8}{'med ms':>9}{'p90 ms':>9}")
    baseline = None
    for w in args.threads:
        rate, rtf, lat = run(call, args.clips, w)
        baseline = baseline or rate
        label = "sequential" if w == 1 else f"{w} client threads"
        print(f"{label:<16}{rate:>9.2f}{rtf:>8.1f}"
              f"{statistics.median(lat) * 1000:>9.0f}"
              f"{lat[int(0.9 * len(lat))] * 1000:>9.0f}")

    print("\nFlat clips/s with latency growing in proportion to threads means the\n"
          "server serialises: more client threads will not help.")

    if args.instances and args.engine == "piper":
        targets = [split(s) for s in args.instances]
        calls = [make(h, prt) for h, prt in targets]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(len(calls)) as ex:
            share = args.clips // len(calls)
            futs = [ex.submit(lambda c=c: [c(i) for i in range(share)]) for c in calls]
            for f in futs:
                f.result()
        rate = (share * len(calls)) / (time.perf_counter() - t0)
        shown = ", ".join(f"{h}:{prt}" for h, prt in targets)
        print(f"\n{len(calls)} instances ({shown}), 1 thread each: "
              f"{rate:.2f} clips/s = {rate / baseline:.2f}x one instance sequential")
        print("Below ~1.0x the instances are contending for the same cores; run one.")


if __name__ == "__main__":
    main()
