"""The Kokoro TTS client: server pool, voice catalog, rendering, batching.

Extracted from train/oww/train.py as a verbatim move, not a rewrite - sixteen tuning
runs are calibrated against that behaviour and anything that looks like it wants
tidying probably encodes a measured result (see the module notes throughout). The
reason it moved: the microWakeWord corpus was Piper-only because this client lived
inside the openWakeWord trainer, which the corpus layer cannot import without
dragging in that whole pipeline. Both trainers now import it from here - the same
"shared code, separate output" split as the rest of this package.

WHAT MOVED AND WHAT STAYED:

  * moved: KokoroPool, probe_kokoro_servers, get_kokoro_voices,
    kokoro_tts / kokoro_tts_timed / kokoro_tts_batch, phrase_end_sample,
    generate_kokoro_sample (a dead single-clip helper kept for reference),
    run_jobs and generate_kokoro_samples.
  * stayed in train/oww/train.py: generate_runon_samples. Its cut logic needs
    RUNON_SPEEDS and RUNON_TAIL_MS, which stay openWakeWord-local for now
    (RUNON_TAIL_MS carries the run-4-through-7 history), and microWakeWord has no
    run-on positives yet. It moves here when it needs to, with those constants.

"mlx://" is a URL that means "in this process": kokoro_tts and kokoro_tts_timed
dispatch on it rather than plumbing a backend argument, because a URL is what
every call site already threads around - the measurement for that backend is in
corpus/kokoro_mlx.py.
"""
import base64
import concurrent.futures as cf
import io
import sys
import threading
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile
from tqdm import tqdm

from . import kokoro_mlx
from .positives import PLAIN_SPEEDS, PLAIN_SPEED_GRID

# Same filter train/oww/train.py has set since before this file existed: the TTS
# reads below occasionally hit a partial read and urllib3 warns, once per clip.
warnings.filterwarnings("ignore", message="Reached EOF prematurely")


class KokoroPool:
    """Hands out TTS server URLs round-robin, so several servers share the work.

    Worth the trouble because one Kokoro process is single-threaded: measured at
    101.8% CPU (exactly one core) while the GPU sat at 21% and VRAM at 1.5 of 24 GB.
    Extra client threads against one server therefore just queue - four workers
    against one server measured 15.0 it/s versus 14.1 sequential. Extra *processes*
    are what scale, until the cores run out.

    Round-robin per job rather than a static split by index, so a thread that draws
    a slow server holds it longer and completes fewer jobs. That balancing is weak,
    though - it does not make a slow server free.

    MEASURED: with batching on, adding two remote servers to the two local ones made
    the run SLOWER, so the local pair alone is the better configuration here.
    Batching amortises latency, not bandwidth: it pays the ~119 ms fixed cost once
    per batch instead of once per clip, but the bytes returned are unchanged, and a
    batch of 16 sends back ~640 KB instead of ~40 KB. The local containers talk over
    the Docker bridge with no physical network; a remote server over a
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


def get_kokoro_voices(kokoro_url: str) -> list:
    """Get all available English voices from Kokoro."""
    if kokoro_mlx.is_mlx_url(kokoro_url):
        ok, why = kokoro_mlx.available()
        if not ok:
            print(f"ERROR: {kokoro_url} requested but MLX is not usable: {why}")
            sys.exit(1)
        # Loading the model here rather than lazily on the first clip, so a failure
        # lands before the run prints its plan - the same reason the HTTP path probes
        # the server up front instead of discovering it is down mid-corpus.
        english = kokoro_mlx.voices()
        print(f"Kokoro voices available: {len(english)} (MLX, in-process)")
        if len(english) < 40:
            print(f"  NOTE: the HTTP service offers 42 English voices; this offers "
                  f"{len(english)}. Voice diversity is a corpus lever - see "
                  f"corpus/kokoro_mlx.py.")
        return english
    try:
        r = requests.get(f"{kokoro_url}/v1/audio/voices", timeout=5)
        voices = r.json().get("voices", [])
        # Filter to English voices (a = American, b = British)
        voices = [v["id"] if isinstance(v, dict) else v for v in voices]
        english = [v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))]
        print(f"Kokoro voices available: {len(english)}")
        return english
    except Exception as e:
        print(f"ERROR: Cannot connect to Kokoro at {kokoro_url}: {e}")
        print("Make sure Kokoro is running:")
        print("  docker compose up -d kokoro kokoro2")
        print("  # on the training server, for the GPU image:")
        print("  docker compose -f docker-compose.yml -f docker-compose.cuda.yml \\")
        print("      up -d kokoro kokoro2")
        sys.exit(1)


def kokoro_tts(kokoro_url: str, voice: str, text: str, speed: float):
    """Render one utterance as 16 kHz int16 audio, or None on failure.

    "mlx://" means in-process rather than over HTTP - see corpus/kokoro_mlx.py. It is
    dispatched on the URL rather than plumbed through as a separate backend argument
    because a URL is what every call site already threads around; this keeps
    KokoroPool, generate_kokoro_samples and generate_runon_samples untouched.
    """
    if kokoro_mlx.is_mlx_url(kokoro_url):
        return kokoro_mlx.render(voice, text, speed)
    try:
        r = requests.post(
            f"{kokoro_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed
            },
            timeout=30
        )
        if r.status_code != 200:
            return None

        sr, data = scipy.io.wavfile.read(io.BytesIO(r.content))
        if data.ndim > 1:
            data = data[:, 0]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(data) * 16000 / sr)
            data = resample(data, num_samples)

        return np.clip(data, -32768, 32767).astype(np.int16)
    except Exception:
        return None


def kokoro_tts_timed(kokoro_url: str, voice: str, text: str, speed: float):
    """Render an utterance and return (16 kHz int16 audio, word timestamps).

    Kokoro-FastAPI's /dev/captioned_speech returns per-word start/end times
    alongside the audio, which is what makes an exact cut possible for the run-on
    positives: it says precisely where the wake word ends inside the utterance,
    instead of that having to be inferred from a separate phrase-alone rendering.

    Returns (None, None) if the endpoint is unavailable, so callers can fall back.
    """
    if kokoro_mlx.is_mlx_url(kokoro_url):
        return kokoro_mlx.render_timed(voice, text, speed)
    try:
        r = requests.post(
            f"{kokoro_url}/dev/captioned_speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed,
                "stream": False,
                "return_timestamps": True,
            },
            timeout=60
        )
        if r.status_code != 200:
            return None, None

        payload = r.json()
        sr, data = scipy.io.wavfile.read(io.BytesIO(base64.b64decode(payload["audio"])))
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            from scipy.signal import resample
            data = resample(data, int(len(data) * 16000 / sr))
        return np.clip(data, -32768, 32767).astype(np.int16), payload.get("timestamps")
    except Exception:
        return None, None


def kokoro_tts_batch(kokoro_url: str, voice: str, texts: list, speed: float):
    """Render several utterances in ONE request and split them apart.

    Measured against a Kokoro-FastAPI server: a single "hey seeree" request costs
    ~119 ms of fixed overhead plus ~42 ms per second of audio, so for a phrase under
    a second, THREE QUARTERS of the request is overhead. Batching amortises it.

    In isolation a batch of 16-32 reaches ~37 ms/clip against 182 individually (5x),
    but real batches are smaller: buckets hold `samples_per_voice / len(grid)` clips,
    about 9.5 for plain positives at the defaults. Measured end to end at that shape
    it is 137 -> 42 ms/clip, a 3.3x speedup. A coarser speed grid would enlarge the
    buckets but measured no faster (39 ms/clip at 0.10 steps), so the finer grid is
    kept for the extra speed diversity.

    The split is exact, not energy-based: /dev/captioned_speech returns per-word
    start and end times, so each utterance is cut at its own word boundaries. The
    texts are joined with ". " and the punctuation tokens are filtered back out of
    the timestamp list.

    Every text in a batch shares one voice and one speed - that is what makes it a
    single forward pass - so callers must group by (voice, speed) before calling.

    Returns a list of (audio, timestamps) in the order given, with None for any
    utterance whose words could not be located. Falls back to nothing: a caller
    seeing None should render that one individually.
    """
    if not texts:
        return []

    joined = ". ".join(t.rstrip(".") for t in texts) + "."
    data, timestamps = kokoro_tts_timed(kokoro_url, voice, joined, speed)
    if data is None or not timestamps:
        return [(None, None)] * len(texts)

    # Punctuation arrives as its own token; drop it so word indices line up.
    words = [t for t in timestamps if str(t.get("word", "")).strip(".,!?;:")]

    out, cursor = [], 0
    pad = int(16000 * 30 / 1000)
    for text in texts:
        n_words = len(text.split())
        if cursor + n_words > len(words):
            out.append((None, None))
            continue
        span = words[cursor:cursor + n_words]
        cursor += n_words

        start = max(0, int(span[0]["start_time"] * 16000) - pad)
        end = min(len(data), int(span[-1]["end_time"] * 16000) + pad)
        if end <= start:
            out.append((None, None))
            continue

        # Re-base the timestamps so they read as if this clip were rendered alone -
        # phrase_end_sample and the run-on cut both index from the clip's own start.
        rebased = [{"word": t.get("word"),
                    "start_time": t["start_time"] - start / 16000,
                    "end_time": t["end_time"] - start / 16000} for t in span]
        out.append((data[start:end], rebased))
    return out


def phrase_end_sample(timestamps, wake_word: str, sr: int = 16000):
    """Sample index where the wake word ends, or None if the words do not line up.

    Verified rather than assumed: the timestamps are matched against the words of
    the wake phrase before their times are used. A mismatch (different tokenisation,
    a normalisation rule splitting a word) would otherwise cut at the wrong place
    silently, and a wrong cut here is what broke the alignment last time.
    """
    if not timestamps:
        return None

    strip = str.maketrans("", "", ".,!?;:\"'")
    expected = [w.translate(strip).lower() for w in wake_word.split()]
    got = [str(t.get("word", "")).translate(strip).lower()
           for t in timestamps[:len(expected)]]
    if got != expected:
        return None

    end = timestamps[len(expected) - 1].get("end_time")
    return int(end * sr) if end else None


def generate_kokoro_sample(kokoro_url: str, voice: str, text: str, output_dir: Path,
                           speed: float = None) -> bool:
    """Generate a single Kokoro TTS sample."""
    if speed is None:
        speed = np.random.uniform(*PLAIN_SPEEDS)
    data = kokoro_tts(kokoro_url, voice, text, speed)
    if data is None:
        return False
    filename = f"kokoro_{uuid.uuid4().hex}.wav"
    scipy.io.wavfile.write(str(output_dir / filename), 16000, data)
    return True


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
    # REQUESTS once batching is on, so "12 it/s" would mean 12 batches - roughly 190
    # clips/s - and look like a slowdown against the pre-batching 28 clips/s. The bar
    # counts clips either way.
    weights = weights or [1] * len(jobs)
    success = 0
    with tqdm(total=sum(weights), desc=desc, unit="clip") as pbar:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for weight, result in zip(weights, pool.map(guarded, jobs)):
                success += int(result)
                pbar.update(weight)
    return success


def generate_kokoro_samples(pool: "KokoroPool", voices: list, output_dir: Path,
                            samples_per_voice: int, texts: list, desc: str,
                            workers: int = 2, batch: int = 16):
    """Generate Kokoro samples for all voices.

    Clips are rendered in batches sharing one voice and speed, which is a ~5x
    speedup (182 ms/clip individually, ~37 ms at a batch of 16-32) because three
    quarters of a short request is fixed overhead. `batch=1` renders each clip in
    its own request, which is the pre-batching behaviour.
    """
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
            speed = float(np.random.choice(PLAIN_SPEED_GRID))
            buckets[(voice, speed)].append(text)
            total += 1

    # One job per batch: a (voice, speed) group sliced into chunks.
    jobs = []
    for (voice, speed), group in buckets.items():
        for i in range(0, len(group), batch):
            jobs.append((voice, speed, group[i:i + batch]))

    def render(job):
        voice, speed, group = job
        url = pool.next()
        if batch == 1:
            data = kokoro_tts(url, voice, group[0], speed)
            results = [(data, None)]
        else:
            results = kokoro_tts_batch(url, voice, group, speed)

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

    written = run_jobs(jobs, render, desc, workers * len(pool),
                       weights=[len(g) for _, _, g in jobs])

    print(f"  Generated {written}/{total} samples "
          f"({len(jobs)} request(s), batch {batch})")
    return written

