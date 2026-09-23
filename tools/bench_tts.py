#!/usr/bin/env python3
"""How many clips per second does one TTS server actually produce?

WHY THIS EXISTS. The answer decides how the corpus gets generated, and the
engines answer it in opposite directions. The numbers below predate the
tts-service split (they were the reason for batching and for the instance
topology), and they were all measured against the raw engines - but they
still describe the engines themselves, because the protocol adds one line of
JSON per exchange, measured at under 1 ms against 100-500 ms renders.

  - Kokoro is single-threaded. One process pins one core, the GPU sits at 21%, and
    concurrent requests just queue (4 client threads: 15.0 it/s against 14.1
    sequential). More INSTANCES scale, roughly linearly, until the cores run out.

    EXCEPT ON METAL, WHERE INSTANCES STOP SCALING: a second MPS instance measured
    9.07 clips/s against 8.45, i.e. nothing, with throughput flat from 1 to 8 client
    threads while latency grew in proportion. Two processes share one GPU and
    serialise on it. On CUDA start two; on Metal start one.

    WHAT THIS TOOL CANNOT TELL YOU, AND IT MATTERS. It renders in the mode you
    tell it to. Batching changes the RANKING, not just the numbers. Measured on
    an M1 Max, Kokoro-FastAPI v0.8.1 throughout, "hey seeree" as plain positives:

                             unbatched        batched     batching gain
        host,   cpu           249 ms/clip      88 ms/clip     2.83x
        host,   mps           110 ms/clip     161 ms/clip     0.68x   <- a LOSS
        docker, cpu           852 ms/clip     323 ms/clip     2.64x

    Unbatched, Metal looks 2.25x better than CPU and this tool will say so. Batched,
    host CPU wins outright at 88 ms/clip and Metal is third. The cause is that
    Kokoro-FastAPI keeps the ISTFT layers on CPU while the rest runs on Metal, and
    ISTFT cost is linear in audio length - so joining ~10 texts into one utterance
    moves ten times the work into the one stage that is not on the GPU.

    So: measure in the mode you will run in. A sweep here that contradicts a real
    corpus run is not noise, it is this.

    The other surprise in that table is the docker row. It is native arm64 with all
    10 CPUs - not emulated - and still 3.7x slower than the same version on the
    host, because the image ships a generic linux/arm64 torch while the host wheel
    uses Accelerate. On Apple Silicon the win is leaving the container, not reaching
    the GPU.
  - Piper is not. onnxruntime parallelises across cores, and one instance measured
    980% CPU - ten cores (this short-phrase workload; on the longer oww wordlist
    phrases the same box measures 462% per instance, and a fleet of 2-8 still
    cannot beat one - SPEED.md "Piper fleet", 2026-09-22). A second instance was
    0.88x, slower than one, because the two contend for cores the first was
    already using.

Guessing which pattern an engine follows gets it backwards, so measure. The sweep
here is designed to tell them apart: if throughput is flat while latency grows
linearly with client threads, requests are queueing behind a serialised stage and
more clients will never help. Whether more INSTANCES help is the second question,
and --instances answers it.

The tool speaks the TTS protocol (tts-service/), so it benches the servers the
corpus actually uses - the per-engine protocol servers on 8898 (piper) and
8899/8901 (the two docker kokoro instances; the Mac's in-process mlx one
defaults to 8900) - and it can batch, because that is how the corpus renders.
A raw `http://` URL is also accepted for the un-wrapped Kokoro-FastAPI
(scripts/start-kokoro-host.sh, or the fastapi port of the docker image): that
mode is unbatched only, one clip per request, which is what all the historical
tables above are.

    python bench_tts.py --url tcp://127.0.0.1:8898
    python bench_tts.py --url tcp://127.0.0.1:8900 --batch 16
    python bench_tts.py --url http://127.0.0.1:8880
    python bench_tts.py --instances tcp://127.0.0.1:8898 tcp://127.0.0.1:8899

Run it on the machine that will generate the corpus. Numbers from a laptop -
worse, from an emulated architecture - are a floor, not a capacity plan.
"""

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts-service" / "tts_protocol"))

from tts_protocol import TtsClient  # noqa: E402

# The phrase and a run-on, because synthesis cost scales with output length and a
# corpus is made of both.
PHRASES = ["hey seeree", "hey seeree what is on tonight"]


def _raw_http_voices(spec):
    import requests
    raw = requests.get(spec.rstrip("/") + "/v1/audio/voices", timeout=30)
    raw.raise_for_status()
    return [v["id"] if isinstance(v, dict) else v for v in raw.json().get("voices", [])]


def sweep_one(spec: str, voice: str, batch: int, threads: int, clips: int):
    """Render `clips` of text against one server with `threads` client threads
    and return (clips/s, median ms/clip, p95 ms/clip, engine_name)."""
    if spec.startswith("http"):
        # Unwrapped Kokoro-FastAPI: one clip per request, no batching (the raw
        # API has no join; the historical tables are all in this mode).
        import requests
        def one(i):
            t0 = time.time()
            r = requests.post(spec.rstrip("/") + "/v1/audio/speech",
                              json={"model": "kokoro", "voice": voice,
                                    "input": PHRASES[i % len(PHRASES)],
                                    "speed": 1.0, "response_format": "wav"},
                              timeout=120)
            r.raise_for_status()
            latencies.append(time.time() - t0)
        work = list(range(clips))
        engine = "kokoro-fastapi (raw)"
    else:
        c = TtsClient(spec)
        c.voices()  # populates server_engine; cached thereafter
        # The corpus's batching: N texts joined into ONE render, split on word
        # timestamps. batch <= 1 must take the plain render() instead - a
        # one-element batch still goes through Engine.batch's join, which
        # appends "." to the text (engine.py), so it is not the request
        # render() would send.
        size = max(batch, 1)
        # min() makes the final chunk the remainder, so work sums to exactly
        # `clips`. Without it every batched run renders one full-size chunk
        # past the request - 64 clips for 60, 112 for 100 at batch 16 -
        # while rate divides by the requested count, reading throughput
        # 7-12% low. This is the tool that feeds SPEED.md; do not bias the
        # numbers it publishes.
        work = [[i for i in range(i, min(i + size, clips))]
                for i in range(0, clips, size)]

        def one(chunk):
            t0 = time.time()
            texts = [PHRASES[i % len(PHRASES)] for i in chunk]
            if len(chunk) == 1:
                c.render(voice, texts[0], 1.0)
            else:
                c.batch(voice, 1.0, texts)
            # per-request wall, divided back to a per-clip number so both
            # modes report the same unit
            latencies.append((time.time() - t0) / len(chunk))
        engine = c.server_engine or "?"
    latencies = []

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, work))
    wall = time.time() - t0
    latencies.sort()
    return (clips / wall,
            latencies[len(latencies) // 2] * 1000,
            latencies[int(len(latencies) * 0.95)] * 1000, engine)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", type=str,
                    help="tcp:// URL of one protocol server, or a raw http://"
                         " Kokoro-FastAPI (unbatched mode)")
    ap.add_argument("--instances", nargs="+", metavar="URL",
                    help="URLs of several servers (instance sweep)")
    ap.add_argument("--batch", type=int, default=1,
                    help="texts joined per render, as the corpus does "
                         "(1 = one text per request; protocol servers only; "
                         "latency percentiles have one sample per render, "
                         "so batched runs need more --clips for a meaningful "
                         "p95)")
    ap.add_argument("--threads", type=int, default=1,
                    help="client threads per server - the queueing axis")
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--voice", type=str, default=None,
                    help="voice name; defaults to the server's first one")
    args = ap.parse_args()

    specs = args.instances or []
    if args.url:
        specs.insert(0, args.url)
    if not specs:
        ap.error("give --url or --instances")

    for spec in specs:
        if spec.startswith("http"):
            try:
                voices = _raw_http_voices(spec)
            except Exception as e:
                print(f"{spec}: UNREACHABLE ({type(e).__name__}: {e})")
                continue
        else:
            c = TtsClient(spec)
            try:
                voices = c.voices()
            except Exception as e:
                print(f"{spec}: UNREACHABLE ({type(e).__name__}: {e})")
                continue
        voice = args.voice or (voices[0] if voices else None)
        if voice is None:
            print(f"{spec}: no voices")
            continue
        rate, med, p95, engine = sweep_one(spec, voice, args.batch, args.threads,
                                           args.clips)
        mode = ("unbatched (raw http)" if spec.startswith("http")
                else f"batch{args.batch}" if args.batch > 1 else "unbatched")
        print(f"{spec}  [{engine}, voice={voice}, {args.threads} thread(s), {mode}]")
        print(f"    {rate:6.2f} clips/s   median {med:8.0f} ms   p95 {p95:8.0f} ms")


if __name__ == "__main__":
    main()
