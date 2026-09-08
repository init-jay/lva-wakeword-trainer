"""Server pool, probe, job runner, and the batched corpus generator.

Everything here moved VERBATIM from train/corpus/kokoro.py (the pre-2026-09-08
git-HEAD version, which the working tree matched - recovered against that .pyc
before this rewrite): KokoroPool, probe_kokoro_servers, run_jobs, and the body of
generate_kokoro_samples. Sixteen tuning runs are calibrated against that behaviour,
and this file is where the calibrated text lives: same bucketing, same per-sample
speed draw, same per-batch server choice, same weights, same tqdm accounting, same
file names.

The only addition is the `Client` class: the same generator, but taking engines
instead of raw URLs so the in-process MLX backend (engines/kokoro_mlx.py) and the
TCP service (engines/tcp.py) can run the identical path. `Client.from_pool` is a
1:1 mapping of the pool's URLs, so `Client.from_pool(pool).generate(...)` is the
git-HEAD `generate_kokoro_samples(pool, ...)` with `pool.next()` swapped for
`client.next()`.

The URL functions (kokoro_tts, kokoro_tts_timed, kokoro_tts_batch,
get_kokoro_voices, phrase_end_sample) live in engines/kokoro_http.py; they are
still the dispatch layer for `mlx://` and are what probe and run-on depend on.
"""
import concurrent.futures as cf
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile
from tqdm import tqdm

from . import engines
from .engines import kokoro_mlx
from .engines.kokoro_http import get_kokoro_voices, kokoro_tts, kokoro_tts_timed


class KokoroPool:
    """Hands out TTS server URLs round-robin, so several servers share the work.

    Worth the trouble because one Kokoro process is single-threaded: measured at
    101.8% CPU (exactly one core) while the GPU sat at 21% and VRAM at 1.5 of 24 GB.
    Extra client threads against one server therefore just queue - four workers
    against one server measured 15.0 it/s versus 14.1 sequential. Extra *processes*
    are what scale, until the cores run out.

    Round-robin per job rather than a static split by index, so a thread that draws
    a slow server holds it longer and completes fewer jobs. That balancing is weak -
    it does not make a slow server free.

    MEASURED: with batching on, adding two remote servers to the two local ones made
    the run SLOWER, so the local pair alone is the better configuration here.
    Batching amortises latency, not bandwidth: it pays the ~119 ms fixed cost once
    per batch instead of once per clip, but the bytes returned are unchanged, and
    a batch of 16 sends back ~640 KB instead of ~40 KB. The local containers talk
    over the Docker bridge with no physical network; a remote server over a
    bandwidth-limited link gains nothing from batching and then takes an equal share
    of the jobs. Measure before adding servers rather than assuming they help.
    """

    def __init__(self, urls):
        self.urls = [u.strip().rstrip("/") for u in urls if u.strip()]
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        if len(self.urls) == 1:
            return self.urls[0]
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url

    def __len__(self):
        return len(self.urls)


def probe_kokoro_servers(pool: KokoroPool) -> list:
    """Report each server, drop the dead ones, and return the voices they all share.

    Servers are only interchangeable if they agree on the voice list - asking a
    server for a voice it lacks fails that job - so the intersection is used.
    Timestamp support is probed too: a server without /dev/captioned_speech pushes
    run-on clips onto the degraded phrase-alone estimate, which is the exact bug
    that cost two training runs, and it would otherwise do so silently.
    """
    # In-process: no servers to probe, no intersection to take, and timestamps are
    # always available - so this reduces to asking the backend what it has.
    if any(kokoro_mlx.is_mlx_url(u) for u in pool.urls):
        if len(pool.urls) > 1:
            print("  NOTE: mlx:// is in-process; extra URLs in the pool are ignored.")
        return get_kokoro_voices(pool.urls[0])

    voices_per_server = []
    for url in pool.urls:
        try:
            r = requests.get(f"{url}/v1/audio/voices", timeout=10)
            voices = r.json().get("voices", [])
            voices = [v["id"] if isinstance(v, dict) else v for v in voices]
            english = {v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))}
        except Exception as e:
            print(f"  {url}: UNREACHABLE ({e})")
            voices_per_server.append(set())
            continue

        _, timestamps = kokoro_tts_timed(url, sorted(english)[0], "test", 1.0)
        timed = "timestamps" if timestamps else "NO timestamps (run-ons degraded)"
        print(f"  {url}: {len(english)} English voices, {timed}")
        voices_per_server.append(english)

    live = [(u, v) for u, v in zip(pool.urls, voices_per_server) if v]
    if not live:
        print("ERROR: no usable Kokoro servers")
        sys.exit(1)

    pool.urls = [u for u, _ in live]
    shared = set.intersection(*[v for _, v in live])
    union = set.union(*[v for _, v in live])
    if shared != union:
        print(f"  NOTE: servers disagree on {len(union - shared)} voice(s); "
              f"using the {len(shared)} they share")
    return sorted(shared)


def run_jobs(jobs, worker, desc: str, workers: int, weights=None):
    """Run `worker` over `jobs` in a thread pool, with a progress bar.

    Threads rather than processes because every job is a blocking HTTP request to
    the TTS server - the work happens there, not here. The pool size is really a
    concurrency limit on the server: measured throughput rises from ~7 to ~11.5
    calls/s going from 1 to 4 workers and is flat at 8, so the server saturates
    early and more workers would only queue.
    """
    def guarded(job):
        # One transient failure must not abandon a batch that takes tens of minutes.
        # pool.map re-raises on iteration, so swallow here and count it as a miss;
        # the caller already reports success against the job total.
        try:
            return worker(job)
        except Exception:
            return 0

    # Workers return either a bool (one clip) or a count (a batch of clips);
    # int() covers both, since int(True) is 1.
    #
    # `weights` is how many clips each job produces. Without it the bar would count
    # REQUESTS once batching is on, so "12 it/s" would mean 12 batches - roughly
    # 190 clips/s - and look like a slowdown against the pre-batching 28 clips/s. The
    # bar counts clips either way.
    weights = weights or [1] * len(jobs)
    success = 0
    with tqdm(total=sum(weights), desc=desc, unit="clip") as pbar:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for weight, result in zip(weights, pool.map(guarded, jobs)):
                success += int(result)
                pbar.update(weight)
    return success


class Client:
    """Engines plus the batched generation they share.

    Build one of two ways:

        Client.from_pool(pool)          # after probe_kokoro_servers; 1:1 with pool.urls
        Client.from_spec(spec)          # an engines_from_spec string, e.g. "kokoro-mlx"

    `next()` alternates engines exactly the way KokoroPool.next() alternated URLs
    (locked round-robin, per job rather than per clip), so a two-server deployment
    renders half of each batch on each one, as before.

    `generate` IS the git-HEAD generate_kokoro_samples, with `pool.next()` replaced
    by `self.next()` and the URL functions replaced by Engine.render / Engine.batch.
    `speeds` is the grid the per-sample draw comes from (the caller's - the
    openWakeWord trainer passes PLAIN_SPEED_GRID; the draw order is the calibrated
    one: one np.random.choice per (voice, sample-index), voice-major).
    """

    def __init__(self, engine_list: list, pool: KokoroPool = None):
        self.engines = engine_list
        self._pool = pool
        self._i = 0
        self._lock = threading.Lock()

    @classmethod
    def from_pool(cls, pool: KokoroPool) -> "Client":
        return cls(engines.engines_from_spec(",".join(pool.urls)), pool=pool)

    @classmethod
    def from_spec(cls, spec: str) -> "Client":
        return cls(engines.engines_from_spec(spec))

    def next(self):
        """The next engine, round-robin - KokoroPool.next() semantics."""
        if len(self.engines) == 1:
            return self.engines[0]
        with self._lock:
            engine = self.engines[self._i % len(self.engines)]
            self._i += 1
            return engine

    def __len__(self):
        return len(self.engines)

    def generate(self, voices: list, output_dir, samples_per_voice: int,
                 texts: list, desc: str, speeds, workers: int = 2,
                 batch: int = 16) -> int:
        """Render the `desc` stage of a corpus across this client's engines.

        The body of the old generate_kokoro_samples (verbatim above the URL
        substitution). Clips are rendered in batches sharing one voice and speed,
        which is a ~5x speedup (182 ms/clip individually, ~37 ms at a batch of
        16-32) because three quarters of a short request is fixed overhead.
        `batch=1` renders each clip in its own request, which is the pre-batching
        behaviour.

        Returns the number of clips written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # The job list is built up front, in one thread. Drawing the speed here rather
        # than inside the worker keeps the corpus a function of the seed alone, instead
        # of depending on the order threads happen to run in.
        from collections import defaultdict
        buckets = defaultdict(list)
        total = 0
        for v, voice in enumerate(voices):
            for i in range(samples_per_voice):
                # Offset each voice's starting point in the wordlist. Without it every
                # voice renders texts[0:samples_per_voice], so a list longer than
                # samples_per_voice never gets past its own beginning - which the test
                # sets hit as soon as the negative list grew past samples_per_voice//10.
                text = texts[(v * samples_per_voice + i) % len(texts)]
                speed = float(np.random.choice(speeds))
                buckets[(voice, speed)].append(text)
                total += 1

        # One job per batch: a (voice, speed) group sliced into chunks.
        jobs = []
        for (voice, speed), group in buckets.items():
            for i in range(0, len(group), batch):
                jobs.append((voice, speed, group[i:i + batch]))

        def render(job):
            voice, speed, group = job
            engine = self.next()
            if batch == 1:
                data = engine.render(voice, group[0], speed)
                results = [(data, None)]
            else:
                results = engine.batch(voice, speed, group)

            written = 0
            for data, _ in results:
                if data is None:
                    continue
                # The voice goes in the name so add_child_range_copies can pick a
                # stretch ratio from its sex, and so a bad voice can be traced later.
                scipy.io.wavfile.write(
                    str(output_dir / f"kokoro_{voice}_{uuid.uuid4().hex}.wav"), 16000, data)
                written += 1
            return written

        written = run_jobs(jobs, render, desc, workers * len(self.engines),
                           weights=[len(g) for _, _, g in jobs])

        print(f"  Generated {written}/{total} samples "
              f"({len(jobs)} request(s), batch {batch})")
        return written
